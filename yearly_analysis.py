import pandas as pd
import numpy as np
from submissions.strategy import generate_signals

def run_yoy_analysis(csv_path="data/SPY.csv"):
    # 1. Load Data
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    
    # 2. Generate Signals and Returns
    signals = generate_signals(data)
    data['Market_Returns'] = data['Close'].pct_change()
    
    # Strategy Returns (1-day shift for execution, 0.05% fee)
    data['Strategy_Returns'] = signals.shift(1) * data['Market_Returns']
    trades = signals.diff().fillna(0).abs()
    data['Strategy_Returns'] = data['Strategy_Returns'] - (trades * 0.0005)

    # 3. Group by Year
    yoy_stats = []
    years = data.index.year.unique()

    for year in years:
        year_data = data[data.index.year == year]
        
        # Calculate Returns
        market_return = (1 + year_data['Market_Returns']).prod() - 1
        strat_return = (1 + year_data['Strategy_Returns']).prod() - 1
        
        # Calculate Sharpe (Annualized)
        if year_data['Strategy_Returns'].std() != 0:
            strat_sharpe = (year_data['Strategy_Returns'].mean() / year_data['Strategy_Returns'].std()) * np.sqrt(252)
        else:
            strat_sharpe = 0
            
        # Calculate Max Drawdown for the year
        cum_ret = (1 + year_data['Strategy_Returns']).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        dd = (cum_ret/peak) - 1
        max_dd = dd.min()

        yoy_stats.append({
            "Year": year,
            "Market Return (%)": round(market_return * 100, 2),
            "Strategy Return (%)": round(strat_return * 100, 2),
            "Alpha (%)": round((strat_return - market_return) * 100, 2),
            "Yearly Sharpe": round(strat_sharpe, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2)
        })

    # 4. Display Table
    df_yoy = pd.DataFrame(yoy_stats)
    print("\n" + "="*70)
    print("YEAR-OVER-YEAR PERFORMANCE COMPARISON")
    print("="*70)
    print(df_yoy.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    run_yoy_analysis()