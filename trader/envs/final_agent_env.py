"""
Final Agent 環境包裝器

將 Sub-Agent 的 Q-values 整合到 state 中
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, List

from trader.envs.trading_env import TradingEnv
from trader.parallel_trainer import SubAgentEnsemble


class FinalAgentEnv(gym.Env):
    """
    Final Agent 環境
    
    包裝原始的 TradingEnv，並將 Sub-Agent 的 Q-values 加入 state
    """
    
    def __init__(self, base_env: TradingEnv, sub_agent_ensemble: SubAgentEnsemble):
        """
        初始化 Final Agent 環境
        
        Args:
            base_env: 基礎交易環境
            sub_agent_ensemble: Sub-Agent 集成器
        """
        super().__init__()
        
        self.base_env = base_env
        self.ensemble = sub_agent_ensemble

        # ★ 繼承基礎環境的模型配置
        self.model_type = getattr(base_env, 'model_type', 'mlp')
        self.use_temporal = getattr(base_env, 'use_temporal', False)
        self.window_size = getattr(base_env, 'window_size', 1)
        
        # ★ 修改：state_dim 始終是整數
        self.base_state_dim = base_env.state_dim  # 這是整數
        self._q_values_dim = self._compute_q_values_dim()
        self.state_dim = self.base_state_dim + self._q_values_dim  # 這也是整數
        
        # ★ 觀察空間根據是否時序決定形狀
        if self.use_temporal:
            # 時序模式：(window_size, state_dim)
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.state_dim),
                dtype=np.float32
            )
        else:
            # 平面模式：(state_dim,)
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            )
        
        # ★★★ 動態計算 Q-values 維度 ★★★
        # 先獲取一個樣本 state 來計算實際的 Q-values 維度
        # self._q_values_dim = self._compute_q_values_dim()
        # self.state_dim = self.base_state_dim + self._q_values_dim
        
        # 繼承基礎環境的屬性
        self.action_dim = base_env.action_dim
        self.num_stocks = base_env.num_stocks
        self.k = getattr(base_env, 'k', 1)
        self.initial_balance = base_env.initial_balance
        self.max_steps = base_env.max_steps
        self.transaction_cost = base_env.transaction_cost
        
        # 定義觀察空間
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.state_dim,),
        #     dtype=np.float32
        # )
        
        # 繼承動作空間
        self.action_space = base_env.action_space
        
        print(f"[FinalAgentEnv] 初始化完成 (模型: {self.model_type.upper()})")
        if self.use_temporal:
            print(f"  - 時序模式")
            print(f"  - Window size: {self.window_size}")
            print(f"  - Base state dim: {self.base_state_dim}")
            print(f"  - Q-values dim: {self._q_values_dim}")
            print(f"  - Total state dim: {self.state_dim}")
            print(f"  - Observation shape: {(self.window_size, self.state_dim)}")
        else:
            print(f"  - 平面模式")
            print(f"  - Base state dim: {self.base_state_dim}")
            print(f"  - Q-values dim: {self._q_values_dim}")
            print(f"  - Total state dim: {self.state_dim}")
        print(f"  - Action dim: {self.action_dim}")
    
    def _augment_state(self, base_state: np.ndarray) -> np.ndarray:
        """
        將 Sub-Agent 的 Q-values 加入 state
        
        Args:
            base_state: 基礎環境的狀態向量
                - 平面模式: (base_state_dim,)
                - 時序模式: (window_size, base_feature_dim)
        
        Returns:
            augmented_state: 增強後的狀態向量
                - 平面模式: (base_state_dim + q_values_dim,)
                - 時序模式: (window_size, base_feature_dim + q_values_dim)
        """
        if self.use_temporal:
            # ★ 時序模式
            # base_state 形狀: (window_size, base_feature_dim)
            
            # 使用最新的狀態（最後一個時間步）來獲取 Q-values
            latest_state = base_state[-1]
            q_values = self.ensemble.get_q_values(latest_state)
            
            # 確保維度正確
            if len(q_values) != self._q_values_dim:
                if len(q_values) < self._q_values_dim:
                    q_values = np.pad(q_values, (0, self._q_values_dim - len(q_values)))
                else:
                    q_values = q_values[:self._q_values_dim]
            
            # 標準化
            if np.std(q_values) > 1e-8:
                q_values = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-8)
            
            # 將 Q-values 擴展到所有時間步
            # (q_values_dim,) -> (window_size, q_values_dim)
            q_values_expanded = np.tile(q_values, (self.window_size, 1))
            
            # 拼接：(window_size, base_feature_dim + q_values_dim)
            augmented_state = np.concatenate([base_state, q_values_expanded], axis=1)
            
        else:
            # ★ 平面模式（原有邏輯）
            q_values = self.ensemble.get_q_values(base_state)
            
            # 確保 Q-values 維度正確
            if len(q_values) != self._q_values_dim:
                if len(q_values) < self._q_values_dim:
                    q_values = np.pad(q_values, (0, self._q_values_dim - len(q_values)))
                else:
                    q_values = q_values[:self._q_values_dim]
            
            # 標準化
            if np.std(q_values) > 1e-8:
                q_values = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-8)
            
            # 合併 state
            augmented_state = np.concatenate([base_state, q_values])
        
        return augmented_state.astype(np.float32)

    def _compute_q_values_dim(self) -> int:
        """
        計算 Q-values 的實際維度
        
        透過實際呼叫 ensemble.get_q_values() 來確定維度
        """
        # 方法 1：使用 ensemble 的估計方法
        estimated_dim = self.ensemble.get_q_values_dim()
        
        # 方法 2：使用實際的 dummy state 測試
        try:
            dummy_state = np.zeros(self.base_state_dim, dtype=np.float32)
            actual_q_values = self.ensemble.get_q_values(dummy_state)
            actual_dim = len(actual_q_values)
            
            if actual_dim != estimated_dim:
                print(f"[FinalAgentEnv] ⚠️ Q-values 維度不一致:")
                print(f"    預估: {estimated_dim}, 實際: {actual_dim}")
                print(f"    使用實際維度: {actual_dim}")
            
            return actual_dim
        except Exception as e:
            print(f"[FinalAgentEnv] ⚠️ 無法計算實際 Q-values 維度: {e}")
            print(f"    使用預估維度: {estimated_dim}")
            return estimated_dim
    
    @property
    def q_values_dim(self) -> int:
        """獲取 Q-values 維度"""
        return self._q_values_dim
    
    def reset(self, seed: Optional[int] = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """重置環境"""
        base_state, info = self.base_env.reset(seed=seed, options=options)
        augmented_state = self._augment_state(base_state)
        
        # 加入 Sub-Agent 的動作建議到 info
        info['sub_agent_actions'] = self.ensemble.get_ensemble_actions(base_state)
        info['base_state_dim'] = self.base_state_dim
        info['q_values_dim'] = self._q_values_dim
        
        return augmented_state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行一步"""
        base_next_state, reward, done, truncated, info = self.base_env.step(action)
        augmented_state = self._augment_state(base_next_state)
        
        # 加入 Sub-Agent 的動作建議到 info
        info['sub_agent_actions'] = self.ensemble.get_ensemble_actions(base_next_state)
        
        return augmented_state, reward, done, truncated, info
    
    def render(self):
        """渲染環境"""
        return self.base_env.render()
    
    def close(self):
        """關閉環境"""
        return self.base_env.close()
    
    # 代理基礎環境的屬性
    @property
    def stock_data(self):
        return self.base_env.stock_data
    
    @property
    def technical_indicators(self):
        return getattr(self.base_env, 'technical_indicators', None)
    
    @property
    def fundamental_data(self):
        return getattr(self.base_env, 'fundamental_data', None)
    
    @property
    def balance(self):
        return self.base_env.balance
    
    @property
    def holdings(self):
        return self.base_env.holdings
    
    @property
    def portfolio_value(self):
        return self.base_env.portfolio_value