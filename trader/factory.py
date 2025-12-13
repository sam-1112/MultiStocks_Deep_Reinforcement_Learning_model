"""
Algorithm Factory - 負責創建強化學習演算法
"""

from typing import Optional
import torch.nn as nn


class AlgorithmFactory:
    """演算法工廠類別"""
    
    @staticmethod
    def create(algorithm: str, **kwargs):
        """
        創建強化學習演算法
        
        Args:
            algorithm: 演算法名稱 ('a2c', 'ddpg', 'ddqn')
            **kwargs: 演算法參數
        
        Returns:
            演算法實例
        """
        algorithm = algorithm.lower()
        
        if algorithm == 'a2c':
            from trader.algos.a2c import A2CStrategy
            return A2CStrategy(**kwargs)
        elif algorithm == 'ddpg':
            from trader.algos.ddpg import DDPGStrategy
            return DDPGStrategy(**kwargs)
        elif algorithm == 'ddqn':
            from trader.algos.ddqn import DDQNStrategy
            return DDQNStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


# 為了向後相容，從 model_factory 匯出 ModelFactory
from trader.model_factory import ModelFactory

__all__ = ['AlgorithmFactory', 'ModelFactory']