from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd

TRADING_DAYS = 252
MAX_EXPOSURE = 0.30
COST_BPS = 5
VOL_WINDOW = 20
TARGET_ANN_VOL = 0.15
MIN_DAILY_VOL = 1e-6

DEFAULT_FAST = 10
DEFAULT_SLOW = 50


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain Date column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated()]
    return df


def generate_signals(data: pd.DataFrame) -> pd.Series:
    if data is None or len(data) == 0:
        return pd.Series(0, index=data.index, dtype="int8")
    if "Close" in data.columns:
        close = pd.to_numeric(data["Close"], errors="coerce")
    else:
        close_cols = [c for c in data.columns if c.startswith("Close_")]
        if len(close_cols) == 0:
            raise ValueError(
                "Data must contain 'Close' or 'Close_<SYMBOL>' column")
        close = pd.to_numeric(data[close_cols[0]], errors="coerce")
    ma_fast = close.rolling(DEFAULT_FAST).mean()
    ma_slow = close.rolling(DEFAULT_SLOW).mean()
    pos = pd.Series(0, index=data.index, dtype="int8")
    pos[ma_fast > ma_slow] = 1
    pos[1.2 * ma_fast < ma_slow] = -1
    return pos.fillna(0).clip(-1, 1).astype("int8")


def generate_signals_with_symbol(df: pd.DataFrame, symbol: str, fast: int, slow: int) -> pd.Series:
    if fast >= slow:
        raise ValueError("Fast SMA must be less than Slow SMA")
    close_col = f"Close_{symbol}"
    if close_col not in df.columns:
        raise ValueError(f"{close_col} not found")
    close = pd.to_numeric(df[close_col], errors="coerce")
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    pos = pd.Series(0, index=df.index, dtype=float)
    pos[ma_fast > ma_slow] = 1.0
    pos[1.2 * ma_fast < ma_slow] = -1.0
    return pos.fillna(0.0)


def strategy_pipeline(
    df: pd.DataFrame,
    raw_position: pd.Series,
    symbol: str,
    *,
    max_exposure: float = MAX_EXPOSURE,
    cost_bps: float = COST_BPS,
    vol_window: int = VOL_WINDOW,
    target_ann_vol: float = TARGET_ANN_VOL,
) -> tuple[pd.Series, pd.Series, pd.Series]:

    close = pd.to_numeric(df[f"Close_{symbol}"], errors="coerce").astype(float)
    returns = close.pct_change().fillna(0.0)

    vol = returns.rolling(vol_window).std()
    vol = vol.replace(0.0, np.nan).fillna(
        method="bfill").fillna(method="ffill")
    vol = vol.clip(lower=MIN_DAILY_VOL)

    target_daily_vol = target_ann_vol / np.sqrt(TRADING_DAYS)
    vol_scale = (target_daily_vol /
                 vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    effective_exposure = raw_position.astype(float) * vol_scale
    effective_exposure = effective_exposure.clip(
        lower=-max_exposure, upper=max_exposure)
    effective_exposure = effective_exposure.fillna(0.0)

    turnover = effective_exposure.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 10_000.0)

    strat_returns = returns * effective_exposure.shift(1).fillna(0.0) - cost
    equity = (1.0 + strat_returns).cumprod()

    return effective_exposure, strat_returns, equity


def compute_metrics(df: pd.DataFrame, raw_position: pd.Series, symbol: str) -> dict:
    _, strat_returns, equity = strategy_pipeline(df, raw_position, symbol)

    total_return = equity.iloc[-1] - 1.0

    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_dd = float(drawdown.min())

    mean = float(strat_returns.mean())
    std = float(strat_returns.std())

    sharpe = (mean / std) * np.sqrt(TRADING_DAYS) if std > 0 else 0.0

    periods = len(equity)
    ann_return = float(
        equity.iloc[-1] ** (TRADING_DAYS / periods) - 1.0) if periods > 1 else 0.0

    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.inf
    win_rate = float((strat_returns > 0).mean())

    exposure = float((raw_position != 0).mean())
    avg_abs_exposure = float(
        np.abs(strategy_pipeline(df, raw_position, symbol)[0]).mean())

    return {
        "Sharpe Ratio": sharpe,
        "Total Return": total_return,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
        "Signal Exposure": exposure,
        "Avg Abs Exposure": avg_abs_exposure,
    }


def build_trades(df: pd.DataFrame, raw_position: pd.Series, symbol: str) -> pd.DataFrame:
    close = pd.to_numeric(df[f"Close_{symbol}"], errors="coerce").astype(float)
    effective_exposure, strat_returns, _ = strategy_pipeline(
        df, raw_position, symbol)

    trades = []
    in_trade = False
    side = 0.0
    entry_date = None
    entry_price = None

    trade_equity = 1.0
    peak_equity = 1.0
    max_dd_trade = 0.0

    for i in range(len(raw_position)):
        date = raw_position.index[i]
        sig = float(raw_position.iloc[i])
        price = float(close.iloc[i])

        daily_r = float(strat_returns.iloc[i])

        if in_trade:
            trade_equity *= (1.0 + daily_r)
            peak_equity = max(peak_equity, trade_equity)
            dd = (trade_equity / peak_equity) - 1.0
            max_dd_trade = min(max_dd_trade, dd)

        if (not in_trade) and sig != 0.0:
            in_trade = True
            side = sig
            entry_date = date
            entry_price = price
            trade_equity = 1.0
            peak_equity = 1.0
            max_dd_trade = 0.0

        elif in_trade and (sig == 0.0 or sig != side):
            exit_date = date
            exit_price = price

            trade_side = "LONG" if side > 0 else "SHORT"
            trade_return = trade_equity - 1.0

            trades.append({
                "Side": trade_side,
                "EntryDate": entry_date,
                "EntryPrice": entry_price,
                "ExitDate": exit_date,
                "ExitPrice": exit_price,
                "PnL": trade_return,
                "Trade Max DD": max_dd_trade,
            })

            in_trade = False

    return pd.DataFrame(trades)


class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", padding=4)
        self.title("Trading Bot")
        self.geometry("1050x700")

        self.df = None
        self.trades = None
        self.metrics = None

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.path_var = tk.StringVar()
        ttk.Label(top, text="CSV:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.path_var, width=70).pack(side=tk.LEFT)
        ttk.Button(top, text="Browse", command=self.browse).pack(
            side=tk.LEFT, padx=6)

        controls = ttk.Frame(self, padding=10)
        controls.pack(fill=tk.X)

        self.symbol_var = tk.StringVar(value="QQQ")
        ttk.Label(controls, text="Symbol:").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.symbol_var,
            values=["QQQ", "SPY"],
            width=6,
            state="readonly"
        ).pack(side=tk.LEFT, padx=6)

        self.fast_var = tk.StringVar(value="10")
        self.slow_var = tk.StringVar(value="50")

        ttk.Label(controls, text="Fast SMA:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(controls, textvariable=self.fast_var,
                  width=6).pack(side=tk.LEFT)

        ttk.Label(controls, text="Slow SMA:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(controls, textvariable=self.slow_var,
                  width=6).pack(side=tk.LEFT)

        ttk.Button(controls, text="Run", command=self.run).pack(
            side=tk.LEFT, padx=10)

        self.metrics_text = tk.Text(self, height=10)
        self.metrics_text.pack(fill=tk.X, padx=10)

        self.tree = ttk.Treeview(
            self,
            columns=("Side", "EntryDate", "EntryPrice", "ExitDate",
                     "ExitPrice", "PnL", "Trade Max DD"),
            show="headings"
        )

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        try:
            self.path_var.set(path)
            self.df = load_csv(path)
            messagebox.showinfo("Loaded", "CSV loaded")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load CSV first")
            return

        try:
            symbol = self.symbol_var.get()
            fast = int(self.fast_var.get())
            slow = int(self.slow_var.get())

            raw_position = generate_signals_with_symbol(
                self.df, symbol, fast, slow)
            self.metrics = compute_metrics(self.df, raw_position, symbol)
            self.trades = build_trades(self.df, raw_position, symbol)

            self.show_metrics()
            self.show_trades()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_metrics(self):
        self.metrics_text.delete("1.0", tk.END)
        for k, v in self.metrics.items():
            if any(x in k for x in ["Return", "Drawdown", "Rate", "Exposure"]):
                self.metrics_text.insert(tk.END, f"{k}: {v:.2%}\n")
            else:
                self.metrics_text.insert(tk.END, f"{k}: {v:.4f}\n")

    def show_trades(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        if self.trades is None or self.trades.empty:
            return

        for _, r in self.trades.iterrows():
            self.tree.insert("", tk.END, values=(
                r["Side"],
                r["EntryDate"],
                round(float(r["EntryPrice"]), 2),
                r["ExitDate"],
                round(float(r["ExitPrice"]), 2),
                f"{float(r['PnL']):.2%}",
                f"{float(r['Trade Max DD']):.2%}",
            ))

if __name__ == "__main__":
    TradingBotGUI().mainloop()