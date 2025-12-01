import numpy as np
import pandas as pd
from typing import Tuple

class DataLoader:
    """股票數據加載器"""

    def __init__(self, stock_symbols: list, start_date: str, end_date: str):
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
    
    def load_stock_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加載股票數據
        
        :return: (stock_data, technical_indicators, fundamental_data)
                形狀：
                - stock_data: (num_steps, num_stocks, 5) - OHLCV
                - technical_indicators: (num_steps, num_stocks, num_indicators)
                - fundamental_data: (num_steps, num_stocks, num_fundamentals)
        """
        
        # 加載原始數據
        stock_df = self.load_ohclv_data()           # DataFrame
        technical_df = self.load_technical_indicators()  # DataFrame
        fundamental_df = self.load_fundamental_data()    # DataFrame
        
        # ← 轉換為 numpy 數組並重新整形
        stock_data = self._reshape_ohlcv_data(stock_df)
        tech_indicators = self._reshape_indicator_data(technical_df)
        fund_data = self._reshape_fundamental_data(fundamental_df)
        
        return stock_data, tech_indicators, fund_data
    
    def _reshape_ohlcv_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        重新整形 OHLCV 數據
        
        輸入 DataFrame 格式（example）：
        date       ticker  open  high  low   close volume
        2010-01-04 AAPL   7.62  7.66  7.53  7.56   493729400
        2010-01-04 JPM   41.28 41.74 41.00 41.60  197039800
        ...
        
        輸出形狀：(num_steps, num_stocks, 5)
        其中最後一維是 [Close, High, Low, Open, Volume]
        """
        # 確定日期和股票
        dates = sorted(df['date'].unique())
        tickers = sorted(df['ticker'].unique())
        
        num_steps = len(dates)
        num_stocks = len(tickers)
        
        # 初始化數組
        data = np.zeros((num_steps, num_stocks, 5), dtype=np.float32)
        
        # 填充數據
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            for j, ticker in enumerate(tickers):
                ticker_data = date_df[date_df['ticker'] == ticker]
                if not ticker_data.empty:
                    # [Close, High, Low, Open, Volume]
                    data[i, j, 0] = ticker_data['close'].values[0]
                    data[i, j, 1] = ticker_data['high'].values[0]
                    data[i, j, 2] = ticker_data['low'].values[0]
                    data[i, j, 3] = ticker_data['open'].values[0]
                    data[i, j, 4] = ticker_data['volume'].values[0]
        
        return data
    
    def _reshape_indicator_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        重新整形技術指標數據
        
        輸出形狀：(num_steps, num_stocks, num_indicators)
        """
        if df.empty:
            return np.zeros((1, len(self.stock_symbols), 1), dtype=np.float32)
        
        dates = sorted(df['date'].unique())
        tickers = sorted(df['ticker'].unique())
        
        # 假設技術指標列（除了 date 和 ticker）
        indicator_cols = [col for col in df.columns if col not in ['date', 'ticker']]
        
        num_steps = len(dates)
        num_stocks = len(tickers)
        num_indicators = len(indicator_cols)
        
        data = np.zeros((num_steps, num_stocks, num_indicators), dtype=np.float32)
        
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            for j, ticker in enumerate(tickers):
                ticker_data = date_df[date_df['ticker'] == ticker]
                if not ticker_data.empty:
                    for k, col in enumerate(indicator_cols):
                        data[i, j, k] = ticker_data[col].values[0] if col in ticker_data.columns else 0.0
        
        return data
    
    def _reshape_fundamental_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        重新整形基本面數據
        
        輸出形狀：(num_steps, num_stocks, num_fundamentals)
        """
        if df.empty:
            return np.zeros((1, len(self.stock_symbols), 1), dtype=np.float32)
        
        dates = sorted(df['date'].unique())
        tickers = sorted(df['ticker'].unique())
        
        fundamental_cols = [col for col in df.columns if col not in ['date', 'ticker']]
        
        num_steps = len(dates)
        num_stocks = len(tickers)
        num_fundamentals = len(fundamental_cols)
        
        data = np.zeros((num_steps, num_stocks, num_fundamentals), dtype=np.float32)
        
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            for j, ticker in enumerate(tickers):
                ticker_data = date_df[date_df['ticker'] == ticker]
                if not ticker_data.empty:
                    for k, col in enumerate(fundamental_cols):
                        data[i, j, k] = ticker_data[col].values[0] if col in ticker_data.columns else 0.0
        
        return data
    
    def load_ohclv_data(self, data_dir: str = './data/stock_data') -> pd.DataFrame:
        """
        加載 OHLCV 數據
        
        :return: DataFrame with columns [date, ticker, open, high, low, close, volume]
        """
        all_stock_data = []
        for ticker in self.stock_symbols:
            try:
                file = f"{data_dir}/{ticker}_{self.start_date}_{self.end_date}.csv"
                df = pd.read_csv(file, parse_dates=['date'])
                df['ticker'] = ticker  # ← 添加 ticker 列
                all_stock_data.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found for {ticker}")
        
        if all_stock_data:
            return pd.concat(all_stock_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def load_technical_indicators(self, data_dir: str = './data/indicators') -> pd.DataFrame:
        """加載技術指標數據"""
        all_tech_data = []
        for ticker in self.stock_symbols:
            try:
                file = f"{data_dir}/{ticker}_indicators.csv"
                df = pd.read_csv(file, parse_dates=['date'])
                df['ticker'] = ticker
                all_tech_data.append(df)
            except FileNotFoundError:
                print(f"Warning: Indicators file not found for {ticker}")
        
        if all_tech_data:
            return pd.concat(all_tech_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def load_fundamental_data(self, data_dir: str = './data/fundamentals') -> pd.DataFrame:
        """加載基本面數據"""
        all_fund_data = []
        for ticker in self.stock_symbols:
            try:
                file = f"{data_dir}/{ticker}_fundamentals.csv"
                df = pd.read_csv(file, parse_dates=['date'])
                df['ticker'] = ticker
                all_fund_data.append(df)
            except FileNotFoundError:
                print(f"Warning: Fundamentals file not found for {ticker}")
        
        if all_fund_data:
            return pd.concat(all_fund_data, ignore_index=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def normalize_data(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        使用 Min-Max Scaler 標準化數據
        
        公式: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
        
        :param data: 輸入數據
        :param feature_range: 縮放範圍 (default: (0, 1))
        :return: 標準化後的數據
        """
        data_min = np.nanmin(data, axis=0, keepdims=True)
        data_max = np.nanmax(data, axis=0, keepdims=True)
        
        # 避免除以零
        data_range = data_max - data_min
        data_range[data_range == 0] = 1e-8
        
        # Min-Max 縮放到 [0, 1]
        data_normalized = (data - data_min) / data_range
        
        # 如果需要不同的範圍，進一步縮放
        min_val, max_val = feature_range
        data_scaled = data_normalized * (max_val - min_val) + min_val
        
        return np.nan_to_num(data_scaled, nan=0.0).astype(np.float32)
    
    @staticmethod
    def normalize_data_zscore(data: np.ndarray) -> np.ndarray:
        """Z-Score 標準化（備用方法）"""
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
        return ((data - mean) / (std + 1e-8)).astype(np.float32)