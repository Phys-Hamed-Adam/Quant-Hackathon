import os
import yfinance as yf

def download_hackathon_data():
    os.makedirs("data", exist_ok=True)
    # training data
    tickers = ["SPY", "QQQ"]
    data = yf.download(tickers, start="2015-01-01", end="2026-02-21")
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    data.to_csv("data/training_data_multi.csv")

download_hackathon_data()
