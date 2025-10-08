
**Project-specific README.md** (example for pairs trading)
```markdown
# Pairs Trading Strategy

A statistical arbitrage strategy that identifies and trades cointegrated pairs of stocks.

## Strategy Overview

1. **Pair Selection**: Find cointegrated pairs using the Augmented Dickey-Fuller test
2. **Hedge Ratio**: Calculate optimal hedge ratio using OLS regression
3. **Trading Signals**: Enter when z-score exceeds ±2.0, exit at mean reversion
4. **Risk Management**: Position sizing with transaction costs (10 bps)

## Usage

```python
from pairs_trading import PairsTradingBacktest

# Initialize backtest
backtest = PairsTradingBacktest(transaction_cost=0.001)

# Run complete analysis
tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
results, metrics = backtest.run_complete_backtest(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
