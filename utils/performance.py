import pandas as pd
import numpy as np
from typing import Dict

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic returns
        metrics['total_return'] = (returns.iloc[-1] / returns.iloc[0]) - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252/len(returns)) - 1
        
        # Volatility and risk
        daily_returns = returns.pct_change().dropna()
        metrics['volatility'] = daily_returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        winning_trades = daily_returns > 0
        metrics['win_rate'] = winning_trades.mean()
        
        return metrics
    
    @staticmethod
    def generate_report(metrics: Dict) -> str:
        """Generate formatted performance report"""
        report = []
        report.append("PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Total Return: {metrics['total_return']:>10.2%}")
        report.append(f"Annual Return: {metrics['annual_return']:>9.2%}")
        report.append(f"Volatility: {metrics['volatility']:>12.2%}")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:>10.2f}")
        report.append(f"Max Drawdown: {metrics['max_drawdown']:>9.2%}")
        report.append(f"Win Rate: {metrics['win_rate']:>14.2%}")
        report.append("=" * 50)
        
        return "\n".join(report)
