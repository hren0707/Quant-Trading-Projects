import yfinance as yf
import pandas as pd
from typing import List, Optional

class DataLoader:
    """Utility class for loading and managing financial data"""
    
    @staticmethod
    def load_daily_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load daily adjusted close prices"""
        print(f"Downloading data for {len(tickers)} tickers...")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data.dropna()
    
    @staticmethod
    def load_intraday_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
        """Load intraday data"""
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data.dropna()
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
