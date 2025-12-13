import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from trader.algos.base_algo import AlgorithmStrategy
from trader.model_factory import ModelFactory
from trader.models.self_attention import (
    SimpleAttentionActor,
    SequenceAttentionActor
)

class DDQNStrategy(AlgorithmStrategy):
    """
    Double DQN 策略 - 離散動作空間 {-1, 0, 1}
    
    每支股票獨立輸出 3 個 Q 值
    支持自注意力機制（僅 Final Agent）
    """
    
    def __init__(self, state_dim: int, action_dim: int, model_type: str = 'mlp',
                 actor_lr: float = 1e-4, gamma: float = 0.99, 
                 hidden_dim: int = 256, buffer_size: int = 100000,
                 batch_size: int = 64, tau: float = 0.005,
                 use_attention: bool = False, num_heads: int = 4,
                 attention_type: str = 'simple', configs: dict = None, **kwargs):
        """
        初始化 DDQN
        
        :param state_dim: 狀態維度
        :param action_dim: 動作維度（股票數量）
        :param model_type: 模型類型
        :param actor_lr: 學習率
        :param gamma: 折扣因子
        :param hidden_dim: 隱藏層維度
        :param buffer_size: 經驗回放緩衝區大小
        :param batch_size: 批次大小
        :param tau: 軟更新參數
        :param use_attention: 是否使用自注意力機制（僅 Final Agent）
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型 ('simple' 或 'sequence')
        """
        super().__init__(state_dim, action_dim)
        
        self.num_actions = 3  # {-1, 0, 1}
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        
        print(f"\n[DDQNStrategy] 初始化參數:")
        print(f"  - 狀態維度: {state_dim}")
        print(f"  - 動作維度: {action_dim} (股票數量)")
        print(f"  - 動作空間: {{-1 (賣), 0 (持有), 1 (買)}}")
        print(f"  - 模型類型: {model_type}")
        print(f"  - 使用注意力機制: {use_attention}")
        if use_attention:
            print(f"  - 注意力類型: {attention_type}")
            print(f"  - 注意力頭數: {num_heads}")
        print(f"  - 隱藏層維度: {hidden_dim}\n")
        
        # 移除 max_timesteps（如果有傳入的話）
        kwargs.pop('max_timesteps', None)
        kwargs.pop('k', None)
        configs_dict = kwargs.pop('configs', None) or configs  # ← 提取 configs
        
        # ========== 創建 Q-Network ==========
        if use_attention:
            # 使用注意力機制 Q-Network（Final Agent）
            if attention_type == 'sequence':
                self.q_network = self._create_attention_q_network(
                    SequenceAttentionActor,
                    state_dim, action_dim, hidden_dim, num_heads
                )
                self.target_q_network = self._create_attention_q_network(
                    SequenceAttentionActor,
                    state_dim, action_dim, hidden_dim, num_heads
                )
            else:  # simple
                self.q_network = self._create_attention_q_network(
                    SimpleAttentionActor,
                    state_dim, action_dim, hidden_dim, num_heads
                )
                self.target_q_network = self._create_attention_q_network(
                    SimpleAttentionActor,
                    state_dim, action_dim, hidden_dim, num_heads
                )
        else:
            # 使用原始 MLP Q-Network（Sub-Agent）
            self.q_network = ModelFactory.create_critic(
                model_type=model_type,
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                configs=configs_dict  # ← 傳遞完整配置
            )
            self.target_q_network = ModelFactory.create_critic(
                model_type=model_type,
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                configs=configs_dict  # ← 傳遞完整配置
            )
        
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        
        # 初始化目標網絡
        self._hard_update()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=actor_lr)
        
        # 經驗回放
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def _create_attention_q_network(self, attention_actor_class, 
                                state_dim, action_dim, hidden_dim, num_heads):
        """
        建立注意力 Q-Network 的包裝器
        
        將注意力 Actor 輸出的 logits 轉換為 Q 值
        """
        class AttentionQNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # ← 移除 num_actions 參數
                self.attention_actor = attention_actor_class(
                    state_dim=state_dim,
                    action_dim=action_dim * 3,  # ← 直接設置輸出維度為 action_dim * 3
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
                self.action_dim = action_dim
            
            def forward(self, x):
                """
                前向傳播
                
                :param x: 輸入狀態 (batch_size, state_dim)
                :return: Q 值 (batch_size, action_dim, 3)
                """
                # 獲取 logits (batch_size, action_dim * 3)
                logits = self.attention_actor(x)
                # 重塑為 (batch_size, action_dim, 3)
                q_values = logits.view(-1, self.action_dim, 3)
                return q_values
        
        return AttentionQNetwork()
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        ε-greedy 動作選擇
        
        :param state: 當前狀態
        :param noise_scale: ε 值（探索率）
        :return: 動作陣列 [-1, 0, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # ε-greedy 探索
        if np.random.random() < noise_scale:
            # 隨機選擇動作
            actions = np.random.randint(0, self.num_actions, size=self.action_dim)
        else:
            # 根據 Q 值選擇最優動作
            with torch.no_grad():
                q_values = self.q_network(state_tensor)  # (1, action_dim, 3)
                actions = torch.argmax(q_values, dim=-1).squeeze(0).cpu().numpy()
        
        # 轉換為 [-1, 0, 1]
        return actions - 1
    
    def select_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """
        無探索的貪心動作選擇
        
        :param state: 當前狀態
        :return: 動作陣列 [-1, 0, 1]
        """
        return self.select_action(state, noise_scale=0.0)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存儲經驗
        
        :param action: 動作 [-1, 0, 1]
        """
        # 將 [-1, 0, 1] 轉換回 [0, 1, 2]
        action_idx = action + 1
        self.replay_buffer.append((state, action_idx, reward, next_state, done))
    
    def update_model(self) -> dict:
        """更新模型"""
        if len(self.replay_buffer) < self.batch_size:
            return {'q_loss': 0.0}
        
        # 隨機採樣
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)  # (batch, action_dim)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # ============ Double DQN 更新 ============
        with torch.no_grad():
            # 使用 Q-Network 選擇最優動作
            next_q_values = self.q_network(next_states)  # (batch, action_dim, 3)
            next_actions = torch.argmax(next_q_values, dim=-1)  # (batch, action_dim)
            
            # 使用 Target Q-Network 評估最優動作
            target_q_values = self.target_q_network(next_states)  # (batch, action_dim, 3)
            
            # 選取 next_actions 對應的 Q 值
            next_q_selected = target_q_values.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            # (batch, action_dim)
            
            # 計算目標 Q 值（每支股票的平均）
            next_q_mean = next_q_selected.mean(dim=-1, keepdim=True)  # (batch, 1)
            td_target = rewards + self.gamma * next_q_mean * (1 - dones)
        
        # 計算當前 Q 值（評估）
        current_q_values = self.q_network(states)  # (batch, action_dim, 3)
        
        # 選取實際動作對應的 Q 值
        current_q_selected = current_q_values.gather(2, (actions).unsqueeze(-1)).squeeze(-1)
        # (batch, action_dim)
        
        # 計算當前 Q 值的平均
        current_q_mean = current_q_selected.mean(dim=-1, keepdim=True)  # (batch, 1)
        
        # 計算 TD 誤差
        q_loss = F.mse_loss(current_q_mean, td_target)
        
        # 更新網絡
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        # 軟更新目標網絡
        self._soft_update()
        
        return {'q_loss': q_loss.item()}
    
    def _hard_update(self):
        """硬更新目標網絡"""
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(param.data)
    
    def _soft_update(self):
        """軟更新目標網絡"""
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"[DDQNStrategy] 模型已保存到 {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[DDQNStrategy] 模型已從 {path} 加載")
    
    def get_algorithm_name(self) -> str:
        return "DDQN"