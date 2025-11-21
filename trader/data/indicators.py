"""
Technical indicators for stock trading.
"""

import argparse
import os
import talib
import pandas as pd
import yaml
from datamodule import StockDataModule

class TechnicalIndicators:
    def __init__(self, data_dir='data/indicators'):
        """
        Initializes the TechnicalIndicators class.
        
        Args:
            data_dir (str): Directory to save/load the indicator data.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def calculate_macd(self, df):
        """
        Calculates the MACD indicator.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'Close' column.
        
        Returns:
            pd.DataFrame: DataFrame with MACD and Signal line columns added.
        """
        macd, signal, _ = talib.MACD(df['close'])
        df['MACD'] = macd
        # df['Signal_Line'] = signal
        return df
    
    def calculate_rsi(self, df, period=14):
        """
        Calculates the RSI indicator.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'Close' column.
            period (int): Period for RSI calculation.
        
        Returns:
            pd.DataFrame: DataFrame with RSI column added.
        """
        rsi = talib.RSI(df['close'], timeperiod=period)
        df['RSI'] = rsi
        return df
    
    def calculate_cci(self, df, period=14):
        """
        Calculates the CCI indicator.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'high', 'low', 'close' columns.
            period (int): Period for CCI calculation.
        
        Returns:
            pd.DataFrame: DataFrame with CCI column added.
        """
        cci = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
        df['CCI'] = cci
        return df
    
    def calculate_dmi(self, df, period=14):
        """
        Calculates the DMI indicator, authors computed Directional Movement Index(DX) in the paper.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'high', 'low', 'close' columns.
            period (int): Period for DMI calculation.
        
        Returns:
            pd.DataFrame: DataFrame with +DI and -DI columns added.
        """
        df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=period)
        return df
    
    def calculate_turbulence(self, df, ticker, period=14):
        """
        Calculates a simple turbulence index based on price volatility.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'close' column.
            ticker (str): Stock ticker symbol.
            period (int): Period for turbulence calculation.
        
        Returns:
            pd.DataFrame: DataFrame with Turbulence column added.
        """
        try:
            # 使用簡單的波動率指標作為 turbulence 替代
            # 計算收益率
            returns = df['close'].pct_change()
            
            # 計算滾動標準差作為波動率指標
            rolling_std = returns.rolling(window=period).std()
            
            # 計算相對於歷史平均的偏差（turbulence 概念）
            rolling_mean = returns.rolling(window=period*2).mean()
            rolling_std_long = returns.rolling(window=period*2).std()
            
            # 計算標準化的波動率偏差
            turbulence = (rolling_std - rolling_std.rolling(window=period).mean()) / rolling_std.rolling(window=period).std()
            
            # 處理 NaN 值
            turbulence = turbulence.fillna(0)
            
            # 取絕對值並進行縮放
            turbulence = abs(turbulence) * 100
            
            df['Turbulence'] = turbulence
            
            print(f"Turbulence calculated for {ticker}, range: {turbulence.min():.2f} to {turbulence.max():.2f}")
            
        except Exception as e:
            print(f"Error calculating turbulence for {ticker}: {e}")
            # 如果計算失敗，使用更簡單的方法
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=period).std() * 100
            df['Turbulence'] = volatility.fillna(0)
        
        return df
    
    def save_indicators(self, df, ticker):
        """
        Saves the DataFrame with technical indicators to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock price data with indicators.
            ticker (str): Stock ticker symbol.
        """
        if df.isnull().values.any():
            df = df.fillna(0)
        file_path = os.path.join(self.data_dir, f"{ticker}_indicators.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved indicators for {ticker} to {file_path}")
    

def load_default_config(config_path='configs/defaults.yaml'):
    """
    Load default configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using hardcoded defaults")
        return {'tickers': ['AAPL', 'MSFT', 'GOOGL']}
    except Exception as e:
        print(f"Error loading {config_path}: {e}, using hardcoded defaults")
        return {'tickers': ['AAPL', 'MSFT', 'GOOGL']}

def main():
    # 載入預設配置
    config = load_default_config()
    default_tickers = config.get('data', {}).get('ticker_list', [])
    
    parser = argparse.ArgumentParser(description='Calculate technical indicators for stock data')
    
    # 使用 YAML 檔案中的股票清單作為預設值
    parser.add_argument('--tickers', 
                        nargs='+', 
                        default=default_tickers,
                        help=f'List of stock ticker symbols (default: {" ".join(default_tickers)})')
    
    parser.add_argument('--start-date', 
                        type=str, 
                        default=config.get('start_date', '2010-01-01'),
                        help='Start date for data in YYYY-MM-DD format (default: 2010-01-01)')
    
    parser.add_argument('--end-date', 
                        type=str, 
                        default=config.get('end_date', '2023-03-01'),
                        help='End date for data in YYYY-MM-DD format (default: 2023-03-01)')
    
    parser.add_argument('--data-dir', 
                        type=str, 
                        default=config.get('data_dir', 'data/indicators'),
                        help='Directory to save indicator data (default: data/indicators)')
    
    parser.add_argument('--config', 
                        type=str, 
                        default='default.yaml',
                        help='Path to YAML configuration file (default: default.yaml)')
    
    parser.add_argument('--rsi-period', 
                        type=int, 
                        default=config.get('rsi_period', 14),
                        help='Period for RSI calculation (default: 14)')
    
    parser.add_argument('--cci-period', 
                        type=int, 
                        default=config.get('cci_period', 14),
                        help='Period for CCI calculation (default: 14)')
    
    parser.add_argument('--dmi-period', 
                        type=int, 
                        default=config.get('dmi_period', 14),
                        help='Period for DMI calculation (default: 14)')
    
    parser.add_argument('--turbulence-period', 
                        type=int, 
                        default=config.get('turbulence_period', 14),
                        help='Period for Turbulence calculation (default: 14)')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置檔案，重新載入
    if args.config != 'default.yaml':
        config = load_default_config(args.config)
    
    # 使用解析的參數
    tickers = args.tickers
    start_date = args.start_date
    end_date = args.end_date
    
    print(f"Processing tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Using config file: {args.config}")
    
    # 初始化資料模組和技術指標類別
    data_module = StockDataModule(tickers, start_date, end_date)
    data_module.download_data()
    stock_data = data_module.load_data()
    
    tech_ind = TechnicalIndicators(data_dir=args.data_dir)
    
    # 計算並儲存技術指標
    for ticker, df in stock_data.items():
        print(f"\nProcessing {ticker}...")
        df = tech_ind.calculate_macd(df)
        df = tech_ind.calculate_rsi(df, period=args.rsi_period)
        df = tech_ind.calculate_cci(df, period=args.cci_period)
        df = tech_ind.calculate_dmi(df, period=args.dmi_period)
        df = tech_ind.calculate_turbulence(df, ticker, period=args.turbulence_period)
        tech_ind.save_indicators(df, ticker)
        print(f"Technical indicators for {ticker}:")
        print(df[['close', 'MACD', 'RSI', 'CCI', 'DX', 'Turbulence']].tail())

    
    
if __name__ == "__main__":
    main()