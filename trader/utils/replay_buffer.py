import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, max_size: int = 100000):
        """
        初始化回放緩衝區
        
        :param max_size: 緩衝區最大容量
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, state, action, reward, next_state, done):
        """
        添加經驗到緩衝區
        
        :param state: 當前狀態
        :param action: 採取的動作
        :param reward: 獲得的獎勵
        :param next_state: 下一狀態
        :param done: 是否結束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """
        從緩衝區中隨機採樣一個批次
        
        :param batch_size: 批次大小
        :return: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"緩衝區大小 {len(self.buffer)} 小於批次大小 {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """返回緩衝區中的經驗數量"""
        return len(self.buffer)
    
    def is_full(self):
        """檢查緩衝區是否已滿"""
        return len(self.buffer) == self.max_size
    
    def clear(self):
        """清空緩衝區"""
        self.buffer.clear()