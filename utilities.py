import os
import yfinance as yf
import pandas as pd

def download_hackathon_data():
    os.makedirs("data", exist_ok=True)
    # training data
    tickers = ["SPY", "QQQ"]
    data = yf.download(tickers, start="2015-01-01", end="2026-02-21")
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    data.to_csv("data/training_data_multi.csv")


def data_cleaning():
    df = pd.read_csv("data/training_data_multi.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df = df.drop_duplicates(subset='Date')

    df = df.dropna()

    invalid_qqq = df[df['High_QQQ'] < df['Low_QQQ']]
    invalid_spy = df[df['High_SPY'] < df['Low_SPY']]
    if not invalid_qqq.empty or not invalid_spy.empty:
        df = df[df['High_QQQ'] >= df['Low_QQQ']]
        df = df[df['High_SPY'] >= df['Low_SPY']]

    price_cols = ['Open_QQQ', 'Open_SPY', 'High_QQQ', 'High_SPY', 'Low_QQQ', 'Low_SPY', 'Close_QQQ', 'Close_SPY']
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df[(df['Volume_QQQ'] > 0) & (df['Volume_SPY'] > 0)]

    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    download_hackathon_data()   
