"""
Data module for loading and processing stock market data including OHLCV.
"""

import os
from time import sleep
import pandas as pd
import yfinance as yf

class StockDataModule:
    def __init__(self, tickers, start_date, end_date, data_dir='data/stock_data'):
        """
        Initializes the StockDataModule.
        
        Args:
            tickers (list): List of stock tickers to download data for.
            start_date (str): Start date for data in 'YYYY-MM-DD' format.
            end_date (str): End date for data in 'YYYY-MM-DD' format.
            data_dir (str): Directory to save/load the stock data.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self):
        """
        Downloads stock data for the specified tickers and date range.
        Saves the data as CSV files in the specified data directory.
        """
        failed_downloads = []

        for ticker in self.tickers:
            file_path = os.path.join(self.data_dir, f"{ticker}_{self.start_date}_{self.end_date}.csv")
            if not os.path.exists(file_path):
                try:
                    sleep(10)  # 避免過度請求
                    print(f"Downloading data for {ticker}...")
                    data = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
                    
                    # 檢查是否下載到數據
                    if data.empty:
                        print(f"No data available for {ticker}")
                        failed_downloads.append(ticker)
                        continue
                    
                    # 處理多層列索引，將列名簡化
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    
                    # 重設索引，將 Date 變成一般欄位
                    data.reset_index(inplace=True)
                    
                    # 建立欄位映射字典，排除 Adj Close
                    column_mapping = {}
                    for col in data.columns:
                        col_lower = col.lower()
                        if col_lower in ['date', 'datetime'] or 'date' in col_lower:
                            column_mapping[col] = 'date'
                        elif 'open' in col_lower:
                            column_mapping[col] = 'open'
                        elif 'high' in col_lower:
                            column_mapping[col] = 'high'
                        elif 'low' in col_lower:
                            column_mapping[col] = 'low'
                        elif 'close' in col_lower and 'adj' not in col_lower:
                            column_mapping[col] = 'close'
                        elif 'volume' in col_lower:
                            column_mapping[col] = 'volume'
                    
                    # 重新命名欄位
                    data = data.rename(columns=column_mapping)
                    
                    # 確保我們有必要的欄位
                    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    
                    if missing_columns:
                        print(f"Missing required columns for {ticker}: {missing_columns}")
                        failed_downloads.append(ticker)
                        continue
                    
                    # 選擇需要的欄位並重新排序，排除 Adj Close
                    data = data[required_columns]
                    
                    data.to_csv(file_path, index=False)
                    print(f"Successfully saved data for {ticker}")
                    
                except Exception as e:
                    print(f"Error downloading {ticker}: {e}")
                    failed_downloads.append(ticker)
                    continue
            else:
                print(f"Data for {ticker} already exists. Skipping download.")
            
            

    def load_data(self):
        """
        Loads the stock data from CSV files.

        Returns:
            dict: A dictionary with tickers as keys and their corresponding DataFrames as values.
        """
        all_data = {}
        for ticker in self.tickers:
            file_path = os.path.join(self.data_dir, f"{ticker}_{self.start_date}_{self.end_date}.csv")
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                # 確保 date 欄位是 datetime 格式
                data['date'] = pd.to_datetime(data['date'])
                all_data[ticker] = data
            else:
                print(f"No data found for {ticker}. Please download it first.")
        return all_data

    def get_ohlcv(self, ticker):
        data = self.load_data().get(ticker)
        if data is not None:
            return data[['date', 'open', 'high', 'low', 'close', 'volume']]
        else:
            print(f"No OHLCV data available for {ticker}.")
            return None
        

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    
    data_module = StockDataModule(tickers, start_date, end_date)
    data_module.download_data()
    stock_data = data_module.load_data()
    
    for ticker, df in stock_data.items():
        print(f"Data for {ticker}:")
        print(df.head())