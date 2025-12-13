import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from trader.utils.replay_buffer import ReplayBuffer
from trader.algos.base_algo import AlgorithmStrategy
from trader.model_factory import ModelFactory
from trader.models.self_attention import (
    ContinuousAttentionActor
)

class DDPGStrategy(AlgorithmStrategy):
    """
    DDPG 演算法 - 離散動作空間 {-1, 0, 1}
    
    Actor 輸出連續值 [-1, 1]，然後離散化為 {-1, 0, 1}
    支持自注意力機制增強 Actor Network（僅 Final Agent）
    """
    
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64,
                 model_type='mlp', hidden_dim=128, n_layers=2,
                 use_attention: bool = False, num_heads: int = 4,
                 attention_type: str = 'simple', configs: dict = None, **kwargs):
        """
        初始化 DDPG
        
        :param state_dim: 狀態維度
        :param action_dim: 動作維度（股票數量）
        :param actor_lr: Actor 學習率
        :param critic_lr: Critic 學習率
        :param gamma: 折扣因子
        :param tau: 軟更新參數
        :param buffer_size: 經驗回放緩衝區大小
        :param batch_size: 批次大小
        :param model_type: 模型類型
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層數量
        :param use_attention: 是否使用自注意力機制（僅 Final Agent）
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型 ('simple' 或 'sequence')
        """
        super().__init__(state_dim, action_dim)
        
        self.num_actions = 3  # {-1, 0, 1}
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        
        print(f"\n[DDPGStrategy] 初始化參數:")
        print(f"  - 狀態維度: {state_dim}")
        print(f"  - 動作維度: {action_dim} (股票數量)")
        print(f"  - 動作空間: {{-1 (賣), 0 (持有), 1 (買)}}")
        print(f"  - 模型類型: {model_type}")
        print(f"  - 使用注意力機制: {use_attention}")
        if use_attention:
            print(f"  - 注意力類型: {attention_type}")
            print(f"  - 注意力頭數: {num_heads}")
        print(f"  - 隱藏層維度: {hidden_dim}\n")
        
        # 移除可能的多餘參數
        kwargs.pop('max_timesteps', None)
        kwargs.pop('k', None)
        configs_dict = kwargs.pop('configs', None) or configs  # ← 提取 configs
        
        # ========== 創建 Actor Network ==========
        if use_attention:
            # 使用注意力機制 Actor（Final Agent）
            # ← 使用 ContinuousAttentionActor（為連續控制設計）
            self.actor = ContinuousAttentionActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
        else:
            # 使用原始 MLP Actor（Sub-Agent）
            self.actor = ModelFactory.create_actor(
                model_type=model_type,
                input_dim=state_dim,
                output_dim=action_dim,
                actor_type='continuous',
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                configs=configs_dict  # ← 傳遞完整配置
            )
        
        # Critic: 輸入狀態+動作
        self.critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            configs=configs_dict  # ← 傳遞完整配置
        )
        
        # ========== 創建目標 Actor Network ==========
        if use_attention:
            self.target_actor = ContinuousAttentionActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
        else:
            self.target_actor = ModelFactory.create_actor(
                model_type=model_type,
                input_dim=state_dim,
                output_dim=action_dim,
                actor_type='continuous',
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                configs=configs_dict  # ← 傳遞完整配置
            )
        
        # 目標 Critic
        self.target_critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            configs=configs_dict  # ← 傳遞完整配置
        )
        
        # 優化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 移到設備
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        
        # 初始化目標網絡
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
    
    def _hard_update(self, target, source):
        """硬更新目標網絡"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, target, source):
        """軟更新目標網絡"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def _continuous_to_discrete(self, continuous_action):
        """
        將連續動作轉換為離散動作
        
        :param continuous_action: 連續動作值 [-1, 1]，可能是 torch.Tensor 或 numpy array
        :return: 離散動作 [0, 1, 2]，numpy array
        """
        # ★ 處理 torch.Tensor
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.detach().cpu().numpy()
        
        # 確保是 numpy array
        continuous_action = np.array(continuous_action)
        
        # Clip 到 [-1, 1]
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        
        # 映射到離散動作
        discrete_action = np.zeros_like(continuous_action, dtype=np.int32)
        discrete_action[continuous_action < -0.33] = 0  # 賣
        discrete_action[(continuous_action >= -0.33) & (continuous_action < 0.33)] = 1  # 持有
        discrete_action[continuous_action >= 0.33] = 2  # 買
        
        return discrete_action
    
    def _discrete_to_continuous(self, discrete_action: np.ndarray) -> np.ndarray:
        """
        將離散動作 {-1, 0, 1} 轉換為連續值（用於 Critic）
        
        轉換：-1 → -1.0, 0 → 0.0, 1 → 1.0
        """
        return discrete_action.astype(np.float32)
    
    def select_action(self, state, noise_scale=0.1):
        """
        選擇動作（含探索噪聲）
        
        :param state: 當前狀態
        :param noise_scale: 噪聲標準差
        :return: 離散動作陣列 {-1, 0, 1}
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor 輸出連續值 [-1, 1]
            continuous_action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        
        # 添加探索噪聲
        noise = np.random.normal(0, noise_scale, size=continuous_action.shape)
        continuous_action = continuous_action + noise
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        
        # 轉換為離散動作 {-1, 0, 1}
        discrete_action = self._continuous_to_discrete(continuous_action)
        return discrete_action
    
    def select_action_deterministic(self, state):
        """
        無噪聲的動作選擇（用於評估）
        
        :param state: 當前狀態
        :return: 離散動作陣列 {-1, 0, 1}
        """
        return self.select_action(state, noise_scale=0.0)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存儲經驗
        
        :param action: 離散動作 {-1, 0, 1}
        """
        # 存儲時將離散動作轉為連續值，便於 Critic 處理
        continuous_action = self._discrete_to_continuous(np.array(action))
        self.replay_buffer.add(state, continuous_action, reward, next_state, done)
    
    def update_model(self):
        """更新 Actor 和 Critic 網絡"""
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # 從 replay buffer 採樣
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 轉換為 tensor
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.FloatTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ★ 檢查是否為時序輸入
        is_temporal = (state_tensor.dim() == 3)
        
        # ========== 更新 Critic ==========
        with torch.no_grad():
            # Target Actor 產生下一步的動作
            next_continuous_action = self.target_actor(next_state_tensor)
            
            # ★ 保持為 tensor 進行拼接，稍後再轉換
            # 不需要轉換為離散動作，直接使用連續值
            
            # ★ 處理時序維度
            if is_temporal:
                # state: (batch, window_size, features)
                # action: (batch, action_dim)
                batch_size, window_size, _ = next_state_tensor.size()
                next_action_expanded = next_continuous_action.unsqueeze(1).repeat(1, window_size, 1)
                critic_input = torch.cat([next_state_tensor, next_action_expanded], dim=2)
            else:
                # 平面模式
                critic_input = torch.cat([next_state_tensor, next_continuous_action], dim=1)
            
            # Target Critic 估計 Q 值
            target_q = self.target_critic(critic_input)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * target_q
        
        # Current Critic 估計 Q 值
        if is_temporal:
            batch_size, window_size, _ = state_tensor.size()
            action_expanded = action_tensor.unsqueeze(1).repeat(1, window_size, 1)
            current_critic_input = torch.cat([state_tensor, action_expanded], dim=2)
        else:
            current_critic_input = torch.cat([state_tensor, action_tensor], dim=1)
        
        current_q = self.critic(current_critic_input)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== 更新 Actor ==========
        # Actor 產生動作
        predicted_continuous_action = self.actor(state_tensor)
        
        # 計算 Q 值（使用連續動作）
        if is_temporal:
            batch_size, window_size, _ = state_tensor.size()
            predicted_action_expanded = predicted_continuous_action.unsqueeze(1).repeat(1, window_size, 1)
            actor_critic_input = torch.cat([state_tensor, predicted_action_expanded], dim=2)
        else:
            actor_critic_input = torch.cat([state_tensor, predicted_continuous_action], dim=1)
        
        actor_loss = -self.critic(actor_critic_input).mean()
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== 軟更新 Target 網絡 ==========
        self._soft_update(self.critic, self.target_critic)
        self._soft_update(self.actor, self.target_actor)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """加載模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
    
    def get_algorithm_name(self) -> str:
        return "DDPG"