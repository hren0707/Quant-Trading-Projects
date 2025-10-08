import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class IntradayMomentumStrategy:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def fetch_intraday_data(self, symbol, period='60d', interval='5m'):
        """Fetch intraday data from Yahoo Finance"""
        print(f"Fetching intraday data for {symbol}...")
        stock = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if stock.empty:
            raise ValueError(f"No data found for {symbol}")
            
        # Clean and prepare data
        data = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        
        # Add time-based features
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['day'] = data.index.date
        
        return data
    
    def calculate_momentum_signals(self, data, lookback_periods=[3, 5, 10]):
        """Calculate various momentum indicators"""
        for period in lookback_periods:
            # Price momentum
            data[f'return_{period}'] = data['Close'].pct_change(period)
            
            # Volume momentum
            data[f'volume_ma_{period}'] = data['Volume'].rolling(period).mean()
            data[f'volume_ratio_{period}'] = data['Volume'] / data[f'volume_ma_{period}']
            
            # Volatility (using ATR)
            data[f'atr_{period}'] = self.calculate_atr(data, period)
        
        # RSI
        data['rsi_14'] = self.calculate_rsi(data['Close'], 14)
        
        # MACD
        data['macd'], data['macd_signal'] = self.calculate_macd(data['Close'])
        
        return data
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd, macd_signal
    
    def generate_trading_signals(self, data):
        """Generate buy/sell signals based on momentum"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0
        signals['signal'] = 0
        
        # Define trading conditions
        oversold_condition = (
            (data['rsi_14'] < 30) &
            (data['return_5'] < -0.02) &
            (data['volume_ratio_5'] > 1.5)
        )
        
        overbought_condition = (
            (data['rsi_14'] > 70) &
            (data['return_5'] > 0.02) &
            (data['volume_ratio_5'] > 1.5)
        )
        
        # Generate signals
        signals.loc[oversold_condition, 'signal'] = 1  # Buy
        signals.loc[overbought_condition, 'signal'] = -1  # Sell
        
        # Position management (no shorting for simplicity)
        current_position = 0
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            if row['signal'] == 1 and current_position == 0:
                # Enter long position
                signals.loc[timestamp, 'position'] = 1
                current_position = 1
            elif row['signal'] == -1 and current_position == 1:
                # Exit long position
                signals.loc[timestamp, 'position'] = 0
                current_position = 0
            else:
                # Maintain current position
                signals.loc[timestamp, 'position'] = current_position
        
        return signals
    
    def backtest_strategy(self, data, signals):
        """Backtest the momentum strategy"""
        # Merge signals with price data
        backtest_data = data.copy()
        backtest_data['position'] = signals['position']
        backtest_data['signal'] = signals['signal']
        
        # Calculate returns
        backtest_data['market_returns'] = backtest_data['Close'].pct_change()
        backtest_data['strategy_returns'] = backtest_data['position'].shift(1) * backtest_data['market_returns']
        
        # Apply transaction costs
        trade_entries = backtest_data['position'].diff().fillna(0) != 0
        backtest_data.loc[trade_entries, 'strategy_returns'] -= self.transaction_cost
        
        # Calculate cumulative returns
        backtest_data['cumulative_market'] = (1 + backtest_data['market_returns']).cumprod()
        backtest_data['cumulative_strategy'] = (1 + backtest_data['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        backtest_data['portfolio_value'] = self.initial_capital * backtest_data['cumulative_strategy']
        
        return backtest_data
    
    def calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = returns.iloc[-1] / returns.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 * 6.5) - 1  # Intraday adjustment
        
        # Risk metrics
        volatility = returns.pct_change().std() * np.sqrt(252 * 6.5)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and profit factor
        strategy_returns = returns.pct_change().dropna()
        winning_trades = strategy_returns > 0
        win_rate = winning_trades.mean()
        
        gross_profit = strategy_returns[winning_trades].sum()
        gross_loss = strategy_returns[~winning_trades].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(strategy_returns),
            'winning_trades': winning_trades.sum()
        }
    
    def run_complete_analysis(self, symbol='SPY'):
        """Run complete intraday momentum analysis"""
        # Fetch and prepare data
        data = self.fetch_intraday_data(symbol)
        
        # Calculate technical indicators
        data = self.calculate_momentum_signals(data)
        
        # Generate trading signals
        signals = self.generate_trading_signals(data)
        
        # Run backtest
        results = self.backtest_strategy(data, signals)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results['cumulative_strategy'])
        
        # Plot results
        self.plot_results(results, metrics, symbol)
        
        # Print performance report
        self.print_performance_report(metrics)
        
        return results, metrics
    
    def plot_results(self, results, metrics, symbol):
        """Plot comprehensive backtest results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot cumulative returns
        ax1.plot(results.index, results['cumulative_market'], label='Buy & Hold', linewidth=2)
        ax1.plot(results.index, results['cumulative_strategy'], label='Momentum Strategy', linewidth=2)
        ax1.set_title(f'{symbol} - Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True)
        
        # Plot RSI with overbought/oversold levels
        ax2.plot(results.index, results['rsi_14'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # Plot positions
        ax3.plot(results.index, results['position'], label='Position', color='orange', linewidth=2)
        ax3.set_title('Trading Positions (1=Long, 0=Flat)')
        ax3.set_ylabel('Position')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True)
        
        # Plot volume
        ax4.bar(results.index, results['Volume'], alpha=0.3, color='gray', label='Volume')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(results.index, results['volume_ratio_5'], color='red', label='Volume Ratio')
        ax4.set_title('Volume Analysis')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_report(self, metrics):
        """Print detailed performance report"""
        print("\n" + "="*50)
        print("INTRADAY MOMENTUM STRATEGY PERFORMANCE REPORT")
        print("="*50)
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print("="*50)

# Run the complete intraday momentum strategy
if __name__ == "__main__":
    strategy = IntradayMomentumStrategy(initial_capital=100000, transaction_cost=0.001)
    
    # Analyze SPY (S&P 500 ETF)
    results, metrics = strategy.run_complete_analysis('SPY')
    
    # You can also test with other symbols:
    # results, metrics = strategy.run_complete_analysis('QQQ')  # NASDAQ 100
    # results, metrics = strategy.run_complete_analysis('IWM')   # Russell 2000
