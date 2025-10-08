import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PairsTradingBacktest:
    def __init__(self, transaction_cost=0.001):  # 10 bps
        self.transaction_cost = transaction_cost
        self.portfolio = {}
        
    def get_data(self, tickers, start_date, end_date):
        """Fetch and clean price data"""
        print("Downloading data...")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data = data.dropna()
        return data
    
    def find_cointegrated_pairs(self, data, threshold=0.05):
        """Find statistically significant cointegrated pairs"""
        n = len(data.columns)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = data.iloc[:, i]
                stock2 = data.iloc[:, j]
                
                # Cointegration test
                score, pvalue, _ = coint(stock1, stock2)
                
                if pvalue < threshold:
                    # Calculate hedge ratio
                    model = sm.OLS(stock1, sm.add_constant(stock2))
                    results = model.fit()
                    hedge_ratio = results.params[1]
                    
                    pairs.append({
                        'stock1': data.columns[i],
                        'stock2': data.columns[j],
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'spread': stock1 - hedge_ratio * stock2
                    })
        
        return sorted(pairs, key=lambda x: x['pvalue'])
    
    def calculate_zscore(self, series, window=30):
        """Calculate rolling z-score"""
        return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()
    
    def backtest_pair(self, data, pair, entry_z=2.0, exit_z=0.5, lookback=30):
        """Backtest a single pair"""
        stock1, stock2 = pair['stock1'], pair['stock2']
        hedge_ratio = pair['hedge_ratio']
        
        # Calculate spread and z-score
        spread = data[stock1] - hedge_ratio * data[stock2]
        z_scores = self.calculate_zscore(spread, lookback)
        
        # Initialize positions and tracking
        positions = pd.DataFrame(index=data.index)
        positions['z_score'] = z_scores
        positions['position_stock1'] = 0
        positions['position_stock2'] = 0
        positions['cash'] = 0
        positions['portfolio_value'] = 0
        
        cash = 100000  # Starting capital
        position1 = 0  # Shares of stock1
        position2 = 0  # Shares of stock2
        
        for i, (date, row) in enumerate(positions.iterrows()):
            current_z = row['z_score']
            price1 = data[stock1].iloc[i]
            price2 = data[stock2].iloc[i]
            
            # Trading logic
            if np.isnan(current_z):
                continue
                
            # Entry signal: spread is too wide
            if position1 == 0 and current_z > entry_z:
                # Short spread: short stock1, long stock2
                position1 = -100
                position2 = int(100 * hedge_ratio)
                cash += position1 * price1 * (1 - self.transaction_cost)
                cash += position2 * price2 * (1 - self.transaction_cost)
                
            elif position1 == 0 and current_z < -entry_z:
                # Long spread: long stock1, short stock2
                position1 = 100
                position2 = -int(100 * hedge_ratio)
                cash += position1 * price1 * (1 - self.transaction_cost)
                cash += position2 * price2 * (1 - self.transaction_cost)
            
            # Exit signal: spread has reverted
            elif position1 != 0 and abs(current_z) < exit_z:
                # Close positions
                cash -= position1 * price1 * (1 + self.transaction_cost)
                cash -= position2 * price2 * (1 + self.transaction_cost)
                position1 = 0
                position2 = 0
            
            # Update portfolio values
            portfolio_value = cash + position1 * price1 + position2 * price2
            positions.loc[date, 'position_stock1'] = position1
            positions.loc[date, 'position_stock2'] = position2
            positions.loc[date, 'cash'] = cash
            positions.loc[date, 'portfolio_value'] = portfolio_value
        
        return positions
    
    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0:
            return {}
            
        total_return = (returns.iloc[-1] / returns.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252/len(returns)) - 1
        volatility = returns.pct_change().std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def run_complete_backtest(self, tickers, start_date, end_date):
        """Run complete pairs trading strategy"""
        # Get data
        data = self.get_data(tickers, start_date, end_date)
        
        # Split data into training and testing
        split_date = data.index[len(data)//2]
        train_data = data[data.index <= split_date]
        test_data = data[data.index > split_date]
        
        print("Finding cointegrated pairs...")
        pairs = self.find_cointegrated_pairs(train_data)
        
        if not pairs:
            print("No cointegrated pairs found!")
            return
        
        print(f"\nTop 5 cointegrated pairs:")
        for i, pair in enumerate(pairs[:5]):
            print(f"{i+1}. {pair['stock1']} - {pair['stock2']} (p-value: {pair['pvalue']:.4f})")
        
        # Backtest best pair on test data
        best_pair = pairs[0]
        print(f"\nBacktesting best pair: {best_pair['stock1']} - {best_pair['stock2']}")
        
        results = self.backtest_pair(test_data, best_pair)
        metrics = self.calculate_metrics(results['portfolio_value'])
        
        # Plot results
        self.plot_results(test_data, results, best_pair)
        
        print("\n=== PERFORMANCE METRICS ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return results, metrics
    
    def plot_results(self, data, results, pair):
        """Plot trading results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot prices
        ax1.plot(data.index, data[pair['stock1']], label=pair['stock1'])
        ax1.plot(data.index, data[pair['stock2']], label=pair['stock2'])
        ax1.set_title('Stock Prices')
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread and z-score
        spread = data[pair['stock1']] - pair['hedge_ratio'] * data[pair['stock2']]
        z_scores = self.calculate_zscore(spread)
        ax2.plot(data.index, spread, label='Spread', color='green')
        ax2b = ax2.twinx()
        ax2b.plot(data.index, z_scores, label='Z-Score', color='red', alpha=0.7)
        ax2.set_title('Spread and Z-Score')
        ax2.legend(loc='upper left')
        ax2b.legend(loc='upper right')
        ax2.grid(True)
        
        # Plot portfolio value
        ax3.plot(results.index, results['portfolio_value'], label='Portfolio Value')
        ax3.set_title('Portfolio Value Over Time')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

# Run the complete pairs trading strategy
if __name__ == "__main__":
    # Example with tech stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    backtest = PairsTradingBacktest(transaction_cost=0.001)
    results, metrics = backtest.run_complete_backtest(tickers, start_date, end_date)
