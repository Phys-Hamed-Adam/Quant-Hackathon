# LEFS Quant Hackathon

Build a trading strategy. Best performance wins. You may integrate machine learning and a graphical user interface to your trading bot. This will significantly help your chances in winning.

## What to do

1. Write your strategy in `submissions/strategy.py`
2. Your file must have one function:

```python
def generate_signals(data: pd.DataFrame) -> pd.Series:
```

3. `data` is a DataFrame with columns: `Open`, `High`, `Low`, `Close`, `Volume`
4. Return a Series with the same index as `data`, containing only `-1`, `0`, or `1`
   - `1` = buy / long
   - `0` = hold / flat
   - `-1` = sell / short

## How to submit

1. Create a branch with your team name:

```bash
git checkout -b your-team-name
```

2. Add your code and push:

```bash
git add .
git commit -m "your-team-name submission"
git push origin your-team-name
```

## Rules
- No NaNs in your output
- Signal index must match the data index exactly
- Do not push to `main` — only push to your team branch

## Scoring

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (primary ranking) |
| Total Return | Cumulative return over the period |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | Annual return / max drawdown |
| Win Rate | % of profitable days |
