import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# 1. Load the Master Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spy_xgb_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # Core Indicators
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std() 
    df["Volatility_30"] = df["Returns"].rolling(30).std()
    # Benchmark volatility for regime detection
    df["Vol_Benchmark"] = df["Volatility_30"].rolling(252).mean() 
    
    df["SMA_20"] = df["Close"].rolling(20).mean() 
    df["SMA_50"] = df["Close"].rolling(50).mean()
    
    # ATR for stop loss
    df["Prev_close"] = df["Close"].shift(1) 
    df["TR"] = df[["High", "Low", "Prev_close"]].max(axis=1) - df[["High", "Low", "Prev_close"]].min(axis=1)
    df["Atr_14"] = df["TR"].rolling(14).mean()
    
    # Momentum & Divergence
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    
    # Structural Features
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"] 
    df["Resistance"] = df["High"].rolling(20).mean()
    df["Support"] = df["Low"].rolling(20).min()
    
    return df

def generate_signals(data: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(0, index=data.index)

    df = calculate_technical_indicators(data)
    features = [
        "Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", 
        "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", 
        "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS",
        "Resistance", "Support"
    ]
    
    X = df[features].fillna(0).values
    probs = model.predict_proba(X)[:, 1]
    
    # --- DYNAMIC ALPHA SCALING ---
    # We compare current volatility to a long-term benchmark.
    # If market is quiet, we lower the threshold to 0.501 to catch early trends.
    # If market is wild, we raise it to 0.52 to avoid getting chopped.
    vol_ratio = df["Volatility_30"] / df["Vol_Benchmark"].fillna(df["Volatility_30"].mean())
    dynamic_thresh = np.where(vol_ratio < 1.0, 0.501, 0.515)
    
    signals = np.where(probs > dynamic_thresh, 1, np.where(probs < (1 - dynamic_thresh), -1, 0))
    
    # --- TREND CONTINUATION LOGIC ---
    # Prevents early exits in the first half: Stay in if the 20-day trend is healthy
    # even if a single-day probability dips.
    strong_trend = (df["SMA_20"] > df["SMA_50"])
    signals[(signals == 1) & (df["Close"] < df["SMA_20"]) & (~strong_trend)] = 0

    # ATR-BASED TRAILING STOP (Reduced to 3.5x for steadier curve)
    final_signals = signals.copy()
    stop_price = 0.0
    in_pos = 0 

    for i in range(1, len(data)):
        curr_price = df["Close"].iloc[i]
        curr_atr = df["Atr_14"].iloc[i]
        
        if in_pos == 1:
            # Trailing stop that tightens slightly in low vol
            multiplier = 3.5 if vol_ratio.iloc[i] > 1.0 else 2.5
            new_stop = curr_price - (curr_atr * multiplier)
            stop_price = max(stop_price, new_stop)
            if curr_price < stop_price:
                final_signals[i] = 0
                in_pos = 0
        elif in_pos == -1:
            multiplier = 3.5 if vol_ratio.iloc[i] > 1.0 else 2.5
            new_stop = curr_price + (curr_atr * multiplier)
            stop_price = min(stop_price, new_stop) if stop_price != 0 else new_stop
            if curr_price > stop_price:
                final_signals[i] = 0
                in_pos = 0
        else:
            in_pos = signals[i]
            mult = 3.5 if vol_ratio.iloc[i] > 1.0 else 2.5
            stop_price = curr_price - (curr_atr * mult) if in_pos == 1 else curr_price + (curr_atr * mult)

    return pd.Series(final_signals, index=data.index).astype(int)