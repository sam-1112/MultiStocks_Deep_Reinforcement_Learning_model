from trader.envs.trading_env import TradingEnv
from trader.utils.dataloader import DataLoader
import numpy as np

class EnvironmentFactory:
    """環境工廠"""
    
    @staticmethod
    def create_trading_env(config: dict) -> TradingEnv:
        """
        創建交易環境（統一使用離散動作空間）
        
        :param config: 配置字典
        :return: TradingEnv 實例
        """
        print(f"[EnvironmentFactory] Creating trading environment...")
        
        # 加載數據
        if 'stock_data' not in config:
            print(f"[EnvironmentFactory] Loading stock data...")
            
            stock_symbols = config['stock_symbols']
            start_date = config['start_date']
            end_date = config['end_date']
            
            loader = DataLoader(stock_symbols, start_date, end_date)
            
            # ← 加載數據（返回 np.ndarray）
            stock_data, tech_data, fund_data = loader.load_stock_data()
            
            # ← 驗證返回的是 np.ndarray
            assert isinstance(stock_data, np.ndarray), f"stock_data must be np.ndarray, got {type(stock_data)}"
            assert isinstance(tech_data, np.ndarray), f"tech_data must be np.ndarray, got {type(tech_data)}"
            assert isinstance(fund_data, np.ndarray), f"fund_data must be np.ndarray, got {type(fund_data)}"
            
            # ← 規範化數據
            stock_data = DataLoader.normalize_data(stock_data, feature_range=(-1, 1))
            tech_data = DataLoader.normalize_data(tech_data, feature_range=(-1, 1))
            fund_data = DataLoader.normalize_data(fund_data, feature_range=(-1, 1))
            
            config['stock_data'] = stock_data
            config['technical_indicators'] = tech_data
            config['fundamental_data'] = fund_data
            
            print(f"[EnvironmentFactory] Stock data shape: {stock_data.shape}")
            print(f"[EnvironmentFactory] Technical indicators shape: {tech_data.shape}")
            print(f"[EnvironmentFactory] Fundamental data shape: {fund_data.shape}\n")
        
        print(f"[EnvironmentFactory] Configuration:")
        print(f"  - Stocks: {config['num_stocks']}")
        print(f"  - Initial balance: ${config['initial_balance']:,.0f}")
        print(f"  - Action space: [-{config.get('k', 5)}, ..., 0, ..., {config.get('k', 5)}]")
        print(f"  - Max steps: {config['max_steps']}\n")
        
        return TradingEnv(config)