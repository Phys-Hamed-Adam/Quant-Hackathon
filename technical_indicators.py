#technical indicators
# volatility will be using 10 day and 30 day window rolling
# returns calculated from closing price
# moving averages using slow and fast paced, trend sentiment detection
# previous close for Break-of-Structure to determine next movement and ATR
# Divergences to help detect areas of reversion 
# Rolling window score for momentum
# fibonacci levels to further enhance identification of areas of interest
# using a 10/20 day window support and resistance

import ta  #technical indicator library
import numpy as np
import pandas as pd 
import os

PATH = "data/cleaned.csv"
OUTPUT = "data/"

def add_ta(df: pd.DataFrame = None) -> pd.DataFrame:
    cloned = df.copy() #copy the dataframe to avoid overwriting previous
    
    #basic trend and volatility indicators
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(window=10).std() #10 and 30 day deviation of returns
    df["Volatility_30"] = df["Returns"].rolling(window=30).std()
    
    df["SMA_20"] = df["Close"].rolling(window=20).mean() #mean simple moving average length 20, 20 days
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    df["Prev_close"] = df["Close"].shift(1) #go back 1 candle or X candles
    df["Atr_high_low"] = df["High"] - df["Low"] 
    df["Atr_high_close"] = (df["High"] - df["Prev_close"]).abs()
    df["Atr_low_close"] = (df["Low"] - df["Prev_close"]).abs()
    df["Atr"] = df[["Atr_high_low", "Atr_high_close", "Atr_low_close"]].max(axis=1)
    df["Atr_14"] = df["Atr"].rolling(window=14).mean() #using the average true range of 14 days
    df.drop(["Atr_high_low", "Atr_high_close", "Atr_low_close", "Atr"], axis=1, inplace=True)
    #drop the redundant data that we do not need overwise dimensionality confusion issues
    
    #rolling window of data
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    #divergences to track the change from moving average to closing price
    
    #break of structure 
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"] #[1, 0, -1] for trend sentiment
    
    #finally swing highs and swing lows for final confirmation
    df["Resistance"] = df["High"].rolling(window=20).mean()
    df["Support"] = df["Low"].rolling(window=20).min()
    
    return df

def clean_after_inclusion(df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = df[c].astype(str).str.replace(",", "").str.strip() #strip of empty commas
            df[c] = pd.to_numeric(df[c], errors="coerce") #convert string to numericals
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df[df["Volume"] > 0] #return information greater than 0 for volume

def prepare_data(df=None):
    if df.index.name != "Date" and "Date" not in df.index.names:
        df = df.set_index(df.columns[0])
    
    df.index = pd.to_datetime(df.index)
    
    spy = [spy for spy in df.columns if "SPY" in spy]
    qqq = [qqq for qqq in df.columns if "QQQ" in qqq] #seperate the data into two files
    
    #remove qqq and spy suffixes to standardise correctly as this labelling format is not correct
    df_spy = df[spy].rename(columns=lambda x: x.replace("_SPY", ""))
    df_qqq = df[qqq].rename(columns=lambda x: x.replace("_QQQ", ""))
    
    df_spy = add_ta(df_spy)
    df_qqq = add_ta(df_qqq)
    
    df_spy = clean_after_inclusion(df_spy) #clean the data of Nans, and empty commas
    df_qqq = clean_after_inclusion(df_qqq)
    
    df_spy.to_csv("data/SPY.csv")
    df_qqq.to_csv("data/QQQ.csv")
    
    print(f"Technical indicators successfully included and stocks seperated!")
    return df_spy, df_qqq

