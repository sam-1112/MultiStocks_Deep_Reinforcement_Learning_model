"""
A trading environment for reinforcement learning agents.
"""

"""
多股票交易環境實現
"""
import numpy as np
from trader.envs.base_env import BaseTradingEnv

class TradingEnv(BaseTradingEnv):
    """
    多股票交易環境
    
    State: [股票交易數據, 技術指標, 基本面數據, 剩餘資金, 持股張數, 組合價值]
    Action: 每支股票的買賣數量 [-k, ..., k]
    Reward: 組合價值變化 + Sharpe Ratio + 0.9 * 日收益率
    """
    
    def __init__(self, config: dict):
        """
        初始化交易環境
        
        :param config: 配置字典
            必需參數:
            - num_stocks: 股票數量
            - initial_balance: 初始資金
            - stock_data: 股票OHLCV數據
            - technical_indicators: 技術指標
            - fundamental_data: 基本面數據
            
            可選參數:
            - max_steps: 最大步數 (default: 252)
            - k: 動作範圍 (default: 5)
            - transaction_cost: 交易成本 (default: 0.001)
        """
        super().__init__(config)
        self.config = config
    