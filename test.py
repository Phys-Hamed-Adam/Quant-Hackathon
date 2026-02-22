"""
LEFS Quant Hackathon — Evaluation Script

Usage: python test.py

Participants submit submissions/strategy.py with:
    generate_signals(data: pd.DataFrame) -> pd.Series of numeric values
"""

import pandas as pd
import numpy as np
import json
import os
import time
import importlib.util
import matplotlib.pyplot as plt

TRADING_DAYS = 252
TRANSACTION_COST = 0.0005
MAX_SIGNAL_TIME_SEC = 10


def load_data(filepath: str, ticker: str | None = "SPY") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "Date"

    new_columns = []
    for col in df.columns:
        parts = col.rsplit("_", 1)  # e.g. "Close_QQQ" -> ["Close", "QQQ"]
        if len(parts) == 2:
            ohlcv, tk = parts
            new_columns.append((tk, ohlcv))
        else:
            new_columns.append((col, ""))

    df.columns = pd.MultiIndex.from_tuples(new_columns, names=["Ticker", "OHLCV"])

    required = {"Open", "High", "Low", "Close", "Volume"}
    tickers = list(df.columns.get_level_values(0).unique())

    # Validate columns exist for each ticker (optional but keeps the original spirit)
    for tk in tickers:
        ticker_cols = set(df[tk].columns)
        if not required.issubset(ticker_cols):
            raise ValueError(f"Ticker {tk} missing OHLCV columns. Found: {list(ticker_cols)}")

    df = df.ffill().dropna()

    # Choose which ticker to convert to single OHLCV
    if ticker is None:
        ticker = "SPY" if "SPY" in tickers else sorted(tickers)[0]
    if ticker not in tickers:
        raise ValueError(f"Requested ticker {ticker} not found. Available: {tickers}")

    # Return single-ticker OHLCV with plain columns
    single = df[ticker].copy()
    single = single[list(required)]  # enforce column order if you want
    single.columns.name = None
    return single


def load_strategy(module_path):
    """Dynamically import the participant's strategy file at runtime."""
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Strategy not found: {module_path}")

    spec = importlib.util.spec_from_file_location("strategy", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_signals"):
        raise AttributeError("Strategy must define generate_signals(data).")
    return module


def validate_signals(signals, data):
    if not isinstance(signals, pd.Series):
        raise TypeError("generate_signals must return a pandas Series.")
    if not signals.index.equals(data.index):
        raise ValueError("Signal index must match data index.")
    if signals.isna().any():
        raise ValueError("Signals contain NaN values.")
    if not np.isfinite(signals).all():
        raise ValueError("Signals must be finite numbers.")
    return signals.astype(float)


def backtest(data, positions):
    # single-ticker returns
    returns = data["Close"].pct_change()

    shifted = positions.shift(1)  # apply signals next day to avoid lookahead
    aligned = pd.concat([returns, shifted], axis=1).dropna()
    returns = aligned.iloc[:, 0]
    shifted = aligned.iloc[:, 1]

    # deduct cost whenever position changes
    costs = shifted.diff().abs().fillna(0) * TRANSACTION_COST
    return shifted * returns - costs


def calculate_metrics(returns):
    vol = np.std(returns)
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS / len(returns)) - 1

    equity = (1 + returns).cumprod()
    max_drawdown = ((equity - equity.cummax()) / equity.cummax()).min()

    sharpe = (np.mean(returns) / vol) * np.sqrt(TRADING_DAYS) if vol > 1e-8 else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
        "win_rate": float((returns > 0).mean()),
        "num_days": len(returns),
    }


def run_evaluation(strategy_path, test_data_path, ticker="SPY"):
    data = load_data(test_data_path, ticker=ticker)
    strategy = load_strategy(strategy_path)

    t0 = time.perf_counter()
    signals = strategy.generate_signals(data)
    signal_time = time.perf_counter() - t0

    if signal_time > MAX_SIGNAL_TIME_SEC:
        raise RuntimeError(f"Too slow: {signal_time:.2f}s (limit {MAX_SIGNAL_TIME_SEC}s)")

    signals = validate_signals(signals, data)
    returns = backtest(data, signals)
    metrics = calculate_metrics(returns)
    metrics["signal_time_sec"] = round(signal_time, 4)

    # equity curve
    equity = (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values, linewidth=1.5)
    plt.title(f"Equity Curve - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.grid(True, alpha=0.3)
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/equity_curve_{ticker}.png", dpi=100, bbox_inches="tight")
    plt.close()

    with open("results/results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("--- Results ---")
    print(f"  Ticker:        {ticker}")
    print(f"  Total Return:  {metrics['total_return']:.2%}")
    print(f"  Annual Return: {metrics['annualized_return']:.2%}")
    print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:  {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate:      {metrics['win_rate']:.2%}")
    print(f"  Signal Speed:  {signal_time:.4f}s")

if __name__ == "__main__":
    run_evaluation("submissions/strategy.py", "")
