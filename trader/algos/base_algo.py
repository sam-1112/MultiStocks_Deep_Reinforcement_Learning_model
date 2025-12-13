from abc import ABC, abstractmethod
from trader.model_factory import ModelFactory
class AlgorithmStrategy(ABC):
    """策略模式的抽象基類，定義演算法的接口"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state, noise_scale=0.1):
        """
        根據當前狀態選擇動作
        
        :param state: 當前狀態
        :param noise_scale: 探索參數
        :return: 動作
        """
        pass

    @abstractmethod
    def update_model(self) -> dict:
        """
        更新模型參數（從回放緩衝區批量採樣）
        
        :return: dict - {'critic_loss': float, 'actor_loss': float}
        """
        pass
    
    @abstractmethod
    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗到回放緩衝區"""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """保存模型到指定路徑"""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """從指定路徑加載模型"""
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """返回演算法名稱"""
        pass