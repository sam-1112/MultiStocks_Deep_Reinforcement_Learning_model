import numpy as np
from trader.envs.trading_env import TradingEnv
from trader.utils.dataloader import DataLoader


class EnvironmentFactory:
    """環境工廠類別"""
    
    @staticmethod
    def create_trading_env(config: dict) -> TradingEnv:
        """
        創建交易環境
        
        Args:
            config: 環境配置字典，包含：
                - num_stocks: 股票數量
                - stock_symbols: 股票代碼列表
                - initial_balance: 初始資金
                - max_steps: 最大步數
                - start_date: 開始日期
                - end_date: 結束日期
                - transaction_cost: 交易成本
                - seed: 隨機種子
                - agent_type: 代理類型（可選）
        
        Returns:
            TradingEnv 實例
        """
        # 載入數據
        loader = DataLoader(
            stock_symbols=config['stock_symbols'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        stock_data, tech_data, fund_data = loader.load_stock_data()
        
        # 處理 NaN 值
        stock_data = EnvironmentFactory._clean_data(stock_data, "stock_data")
        tech_data = EnvironmentFactory._clean_data(tech_data, "tech_data")
        fund_data = EnvironmentFactory._clean_data(fund_data, "fund_data")
        
        # 標準化數據
        stock_data = DataLoader.normalize_data(stock_data)
        tech_data = DataLoader.normalize_data(tech_data)
        fund_data = DataLoader.normalize_data(fund_data)
        
        # 最終驗證
        assert not np.isnan(stock_data).any(), "stock_data contains NaN values after cleaning"
        assert not np.isnan(tech_data).any(), "tech_data contains NaN values after cleaning"
        assert not np.isnan(fund_data).any(), "fund_data contains NaN values after cleaning"
        
        # 打印數據形狀
        print(f"[EnvironmentFactory] Data shapes:")
        print(f"  - stock_data: {stock_data.shape}")
        print(f"  - tech_data: {tech_data.shape}")
        print(f"  - fund_data: {fund_data.shape}")
        
        # 建立環境配置字典
        env_config = {
            'num_stocks': config['num_stocks'],
            'initial_balance': config['initial_balance'],
            'max_steps': config['max_steps'],
            'transaction_cost': config['transaction_cost'],
            'seed': config.get('seed', 42),
            'agent_type': config.get('agent_type', 'final'),
            'stock_data': stock_data,
            'technical_indicators': tech_data,
            'fundamental_data': fund_data,
            # ★ 新增：模型相關配置
            'model_type': config.get('model_type', 'mlp'),
            'window_size': config.get('window_size', 10),
        }
        
        # 創建環境
        env = TradingEnv(env_config)
        
        return env
    
    @staticmethod
    def _clean_data(data: np.ndarray, name: str) -> np.ndarray:
        """
        清理數據中的 NaN 和 Inf 值
        
        Args:
            data: 輸入數據
            name: 數據名稱（用於日誌）
        
        Returns:
            清理後的數據
        """
        if data is None or data.size == 0:
            print(f"[EnvironmentFactory] Warning: {name} is empty, creating zeros array")
            return np.zeros((1, 1, 1), dtype=np.float32)
        
        # 統計 NaN 和 Inf
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"[EnvironmentFactory] Warning: {name} contains {nan_count} NaN and {inf_count} Inf values, replacing with 0")
        
        # 替換 NaN 和 Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data.astype(np.float32)