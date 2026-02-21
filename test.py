"""
LEFS Quant Hackathon — Evaluation Script

Usage: python test.py

Participants submit submissions/strategy.py with:
    generate_signals(data: pd.DataFrame) -> pd.Series of {-1, 0, 1}
"""

import pandas as pd
import numpy as np
import json
import os
import time
import importlib.util

TRADING_DAYS = 252              # used to annualize metrics
TRANSACTION_COST = 0.0005       # 5 bps per trade
MAX_SIGNAL_TIME_SEC = 10        # strategy must finish within this


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data not found: {filepath}")

    # Handle yfinance's multi-row header format (Price/Ticker/Date rows)
    probe = pd.read_csv(filepath, nrows=5, header=None)
    if probe.iloc[0, 0] == "Price":
        df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
        df.columns = df.columns.get_level_values(0)
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    df.index.name = "Date"

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OHLCV columns. Found: {list(df.columns)}")

    return df.ffill().dropna()


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
    if not signals.isin([-1, 0, 1]).all():
        raise ValueError("Signals must only contain -1, 0, or 1.")
    return signals.astype(int)


def backtest(data, positions):
    returns = data["Close"].pct_change()
    shifted = positions.shift(1)  # apply signals next day to avoid lookahead

    aligned = pd.concat([returns, shifted], axis=1).dropna()
    returns, shifted = aligned.iloc[:, 0], aligned.iloc[:, 1]

    # deduct cost whenever position changes
    costs = shifted.diff().abs().fillna(0) * TRANSACTION_COST
    return shifted * returns - costs


def calculate_metrics(returns):
    vol = np.std(returns)
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS / len(returns)) - 1

    equity = (1 + returns).cumprod()  # cumulative wealth curve
    max_drawdown = ((equity - equity.cummax()) / equity.cummax()).min()  # worst peak-to-trough

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


def run_evaluation(strategy_path, test_data_path):
    data = load_data(test_data_path)
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

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("--- Results ---")
    print(f"  Total Return:  {metrics['total_return']:.2%}")
    print(f"  Annual Return: {metrics['annualized_return']:.2%}")
    print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:  {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate:      {metrics['win_rate']:.2%}")
    print(f"  Signal Speed:  {signal_time:.4f}s")


if __name__ == "__main__":
    run_evaluation("submissions/strategy.py", "data/exxonmobil_xom.csv")
