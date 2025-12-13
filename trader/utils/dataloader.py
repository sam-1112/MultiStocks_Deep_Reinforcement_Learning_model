import numpy as np
import pandas as pd
from typing import Tuple, Optional

class DataLoader:
    """股票數據加載器"""

    def __init__(self, stock_symbols: list, start_date: str, end_date: str):
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self._daily_dates = None
    
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
        
        # ← 統一基本面欄位
        # 提取所有可用的基本面欄位
        all_fund_cols = set()
        for ticker in self.stock_symbols:
            ticker_data = fundamental_df[fundamental_df['ticker'] == ticker]
            numeric_cols = ticker_data.select_dtypes(include=[np.number]).columns.tolist()
            all_fund_cols.update(numeric_cols)
        
        # 確保所有票券都有這些欄位（缺失的填 NaN 再填 0）
        common_fund_cols = sorted(list(all_fund_cols))
        for col in common_fund_cols:
            if col not in fundamental_df.columns:
                fundamental_df[col] = np.nan
        
        fundamental_df = fundamental_df.fillna(0)

        # ← 轉換為 numpy 數組並重新整形
        stock_data = self._reshape_ohlcv_data(stock_df)
        tech_indicators = self._reshape_indicator_data(technical_df)
        fund_data = self._reshape_fundamental_data(fundamental_df)
        
        return stock_data, tech_indicators, fund_data
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        尋找日期欄位名稱
        
        Args:
            df: DataFrame
        
        Returns:
            日期欄位名稱，如果找不到則返回 None
        """
        possible_date_cols = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 
                              'datetime', 'DateTime', 'time', 'Time']
        for col in possible_date_cols:
            if col in df.columns:
                return col
        return None
    
    def _standardize_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        標準化日期欄位名稱為 'date'
        
        Args:
            df: 原始 DataFrame
        
        Returns:
            標準化後的 DataFrame
        """
        date_col = self._find_date_column(df)
        if date_col is None:
            print(f"Warning: No date column found. Available columns: {df.columns.tolist()}")
            return df
        
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
        
        # 確保日期格式正確
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
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
        if df.empty:
            return np.zeros((1, len(self.stock_symbols), 5), dtype=np.float32)
        
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
        
        if num_indicators == 0:
            return np.zeros((num_steps, num_stocks, 1), dtype=np.float32)
        
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
        
        if num_fundamentals == 0:
            return np.zeros((num_steps, num_stocks, 1), dtype=np.float32)
        
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
                df = pd.read_csv(file)
                df = self._standardize_date_column(df)
                df['ticker'] = ticker  # ← 添加 ticker 列
                all_stock_data.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found for {ticker}: {file}")
        
        if all_stock_data:
            result = pd.concat(all_stock_data, ignore_index=True)
            # 儲存日線日期供後續使用
            self._daily_dates = pd.DatetimeIndex(sorted(result['date'].unique()))
            return result
        else:
            return pd.DataFrame()

    def load_technical_indicators(self, data_dir: str = './data/indicators') -> pd.DataFrame:
        """加載技術指標數據"""
        all_tech_data = []
        for ticker in self.stock_symbols:
            try:
                file = f"{data_dir}/{ticker}_{self.start_date}_{self.end_date}.csv"
                df = pd.read_csv(file)
                df = self._standardize_date_column(df)
                df['ticker'] = ticker
                all_tech_data.append(df)
            except FileNotFoundError:
                print(f"Warning: Indicators file not found for {ticker}")
        
        if all_tech_data:
            return pd.concat(all_tech_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def load_fundamental_data(self, data_dir: str = './data/fundamentals/daily_aligned') -> pd.DataFrame:
        """
        加載基本面數據
        
        預設從 daily_aligned 目錄讀取對齊後的資料
        """
        all_fund_data = []
        for ticker in self.stock_symbols:
            try:
                df = None
                try:
                    df = pd.read_csv(f"{data_dir}/{ticker}_{self.start_date}_{self.end_date}.csv")
                except FileNotFoundError:
                    print(f"Info: {ticker}_fundamentals_daily.csv not found, trying alternative filenames.")
                
                if df is not None:
                    df = self._standardize_date_column(df)
                    
                    # 如果沒有 ticker 欄位，添加它
                    if 'ticker' not in df.columns:
                        df['ticker'] = ticker
                    
                    all_fund_data.append(df)
                else:
                    print(f"Warning: Fundamentals file not found for {ticker}")
                    
            except Exception as e:
                print(f"Warning: Error loading fundamentals for {ticker}: {e}")
        
        if all_fund_data:
            return pd.concat(all_fund_data, ignore_index=True)
        else:
            print("Warning: No fundamental data loaded. Returning empty DataFrame.")
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
        # 先將 NaN 和 Inf 替換為 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 檢查數據是否全為 0 或空
        if data.size == 0:
            return data.astype(np.float32)
        
        # 檢查是否全為相同值（避免除以零）
        if np.all(data == data.flat[0]):
            return np.zeros_like(data, dtype=np.float32)
        
        # 計算 min 和 max
        data_min = np.min(data, axis=0, keepdims=True)
        data_max = np.max(data, axis=0, keepdims=True)
        
        # 避免除以零
        data_range = data_max - data_min
        # 將範圍為 0 的設為 1，避免除以零
        data_range = np.where(data_range == 0, 1.0, data_range)
        
        # Min-Max 縮放到 [0, 1]
        data_normalized = (data - data_min) / data_range
        
        # 如果需要不同的範圍，進一步縮放
        min_val, max_val = feature_range
        data_scaled = data_normalized * (max_val - min_val) + min_val
        
        # 最後確保沒有 NaN
        data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data_scaled.astype(np.float32)
    
    @staticmethod
    def normalize_data_zscore(data: np.ndarray) -> np.ndarray:
        """Z-Score 標準化（備用方法）"""
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
        return ((data - mean) / (std + 1e-8)).astype(np.float32)

    def _get_daily_dates(self, stock_df: pd.DataFrame) -> pd.DatetimeIndex:
        """從股票數據獲取日線日期"""
        if stock_df.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(stock_df['date'].unique()))

    def load_fundamental_data_aligned_ma(
        self, 
        data_dir: str = './data/fundamentals',
        ma_window: int = 63,
        method: str = 'ema'
    ) -> pd.DataFrame:
        """
        加載並對齊基本面數據到日線，使用移動平均展平
        
        Args:
            data_dir: 基本面資料目錄
            ma_window: 移動平均窗口（預設 63 天，約一季）
            method: 移動平均方法
                - 'sma': 簡單移動平均
                - 'ema': 指數移動平均（推薦）
                - 'wma': 加權移動平均
                - 'interpolate': 線性內插後再平滑
        
        Returns:
            對齊並平滑後的日線基本面 DataFrame
        """
        if self._daily_dates is None:
            raise ValueError("請先加載 OHLCV 數據以獲取日線日期")
        
        from trader.data.fundamentals import FundamentalData
        
        print(f"\n{'='*60}")
        print(f"對齊基本面資料到日線 (使用 {method.upper()} 移動平均)")
        print(f"{'='*60}")
        print(f"移動平均窗口: {ma_window} 天")
        print(f"日線日期範圍: {self._daily_dates.min()} ~ {self._daily_dates.max()}")
        print()
        
        all_aligned = []
        
        for ticker in self.stock_symbols:
            try:
                fund_data = FundamentalData(
                    ticker=ticker,
                    start_date=str(self._daily_dates.min().date()),
                    end_date=str(self._daily_dates.max().date()),
                    data_dir=data_dir
                )
                
                aligned = fund_data.align_to_daily_with_ma(
                    self._daily_dates,
                    ma_window=ma_window,
                    method=method
                )
                
                if not aligned.empty:
                    all_aligned.append(aligned)
                    
                    # 計算平滑效果統計
                    numeric_cols = aligned.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # 計算變化的標準差（越小表示越平滑）
                        changes_std = aligned[numeric_cols].diff().std().mean()
                        print(f"  ✓ {ticker}: {len(aligned)} 筆 (變化標準差: {changes_std:.4f})")
                else:
                    print(f"  ⚠ {ticker}: 無有效資料")
                    
            except FileNotFoundError:
                print(f"  ⚠ {ticker}: 檔案不存在")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
        if all_aligned:
            result = pd.concat(all_aligned, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"總計: {len(result)} 筆資料")
            print(f"{'='*60}\n")
            return result
        else:
            print("警告: 無法載入任何基本面資料")
            return pd.DataFrame()
    
    def load_stock_data_with_ma_fundamentals(
        self,
        ma_window: int = 63,
        ma_method: str = 'ema'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加載股票數據，基本面使用移動平均展平
        
        Args:
            ma_window: 移動平均窗口
            ma_method: 移動平均方法
        
        Returns:
            (stock_data, technical_indicators, fundamental_data)
        """
        # 加載 OHLCV
        stock_df = self.load_ohclv_data()
        self._daily_dates = self._get_daily_dates(stock_df)
        
        # 加載技術指標
        technical_df = self.load_technical_indicators()
        
        # 加載基本面（使用移動平均）
        fundamental_df = self.load_fundamental_data_aligned_ma(
            ma_window=ma_window,
            method=ma_method
        )
        
        # 轉換為 numpy 數組
        stock_data = self._reshape_ohlcv_data(stock_df)
        tech_indicators = self._reshape_indicator_data(technical_df)
        fund_data = self._reshape_fundamental_data(fundamental_df)
        
        return stock_data, tech_indicators, fund_data