from abc import ABC, abstractmethod

class AlgorithmStrategy(ABC):
    """策略模式的抽象基類，定義演算法的接口"""

    @abstractmethod
    def select_action(self, state):
        """根據當前狀態選擇動作"""
        pass

    @abstractmethod
    def update_model(self, state, action, reward, next_state, done):
        """更新模型參數"""
        pass

    @abstractmethod
    def save_model(self, path):
        """保存模型到指定路徑"""
        pass

    @abstractmethod
    def load_model(self, path):
        """從指定路徑加載模型"""
        pass

    @abstractmethod
    def get_algorithm_name(self):
        """返回演算法名稱"""
        pass
        