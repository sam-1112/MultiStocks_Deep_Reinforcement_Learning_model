import numpy as np
import random
import torch
import os
from typing import Optional

class SeedManager:
    """隨機種子管理器 - 確保可重現性"""
    
    @staticmethod
    def set_seed(seed: int = 42):
        """
        設置全局隨機種子（MT19937）
        
        確保以下庫的結果可重現：
        - NumPy (MT19937 生成器)
        - Python random
        - PyTorch CPU & GPU
        - Gym/Gymnasium
        
        :param seed: 隨機種子（0-2^32-1）
        """
        # 驗證種子範圍
        if not isinstance(seed, int) or seed < 0 or seed >= 2**32:
            raise ValueError(f"種子必須是 0 到 {2**32-1} 之間的整數，收到: {seed}")
        
        print(f"[SeedManager] 設置隨機種子: {seed}")
        
        # 1. NumPy MT19937 生成器
        np.random.seed(seed)
        np_generator = np.random.Generator(np.random.MT19937(seed))
        np.random.default_rng(seed)
        
        # 2. Python 內置 random 模塊
        random.seed(seed)
        
        # 3. PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 4. PyTorch 額外設置（確保確定性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 5. 環境變量
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"[SeedManager] 已設置:")
        print(f"  - NumPy MT19937 種子: {seed}")
        print(f"  - Python random 種子: {seed}")
        print(f"  - PyTorch CPU 種子: {seed}")
        if torch.cuda.is_available():
            print(f"  - PyTorch GPU 種子: {seed}")
        print()
    
    @staticmethod
    def get_mt19937_seed(seed: int = 42) -> int:
        """
        獲取 MT19937 種子值
        
        MT19937（Mersenne Twister）是 NumPy 默認的隨機數生成器
        
        :param seed: 輸入種子
        :return: MT19937 種子
        """
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"種子必須是非負整數，收到: {seed}")
        
        # MT19937 的周期是 2^19937-1，但種子空間是 32 位
        # 將種子限制在 [0, 2^32-1] 範圍
        mt_seed = seed % (2**32)
        return mt_seed
    
    @staticmethod
    def generate_random_seeds(n_seeds: int, base_seed: int = 42) -> list:
        """
        生成 n 個不同的 MT19937 種子
        
        用於多個環境實例的初始化
        
        :param n_seeds: 要生成的種子數量
        :param base_seed: 基礎種子
        :return: 種子列表 [seed1, seed2, ..., seedN]
        """
        print(f"[SeedManager] 生成 {n_seeds} 個隨機種子（基礎: {base_seed}）")
        
        rng = np.random.Generator(np.random.MT19937(base_seed))
        seeds = rng.integers(0, 2**32, size=n_seeds)
        
        print(f"[SeedManager] 已生成種子: {seeds}\n")
        return seeds.tolist()
    
    @staticmethod
    def get_seed_info(seed: int) -> dict:
        """
        獲取種子的詳細信息
        
        :param seed: 隨機種子
        :return: dict - 種子信息
        """
        return {
            'seed': seed,
            'seed_hex': hex(seed),
            'seed_binary': bin(seed),
            'max_mt19937_seed': 2**32 - 1,
            'generator': 'MT19937 (Mersenne Twister)',
            'period': '2^19937 - 1',
            'state_size': '19937 bits',
            'word_size': '32 bits'
        }


class EnvironmentSeeder:
    """環境種子管理器 - 為環境 reset 提供種子"""
    
    def __init__(self, base_seed: int = 42):
        """
        初始化環境種子管理器
        
        :param base_seed: 基礎種子
        """
        self.base_seed = base_seed
        self.rng = np.random.Generator(np.random.MT19937(base_seed))
        self.episode_count = 0
    
    def get_reset_seed(self) -> int:
        """
        獲取下一個 reset 的種子
        
        每次調用返回不同的種子（基於 MT19937 生成器）
        
        :return: MT19937 種子
        """
        seed = int(self.rng.integers(0, 2**32))
        self.episode_count += 1
        return seed
    
    def get_deterministic_seed(self, episode: int) -> int:
        """
        獲取確定性種子（基於回合號）
        
        相同的回合號總是返回相同的種子
        
        :param episode: 回合號
        :return: 確定性種子
        """
        # 使用 episode 號作為種子，確保可重現
        rng_deterministic = np.random.Generator(np.random.MT19937(self.base_seed + episode))
        return int(rng_deterministic.integers(0, 2**32))
    
    def reset_episode_count(self):
        """重置回合計數"""
        self.episode_count = 0
    
    def get_episode_count(self) -> int:
        """獲取當前回合數"""
        return self.episode_count