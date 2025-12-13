import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trader.utils.replay_buffer import ReplayBuffer
from trader.algos.base_algo import AlgorithmStrategy
from trader.model_factory import ModelFactory
from trader.models.self_attention import (
    SimpleAttentionActor,
    SequenceAttentionActor
)

class A2CStrategy(AlgorithmStrategy):
    """
    A2C 策略 - 3 個固定動作 {-1, 0, 1}
    
    支持：
    1. 無注意力機制（Sub-Agent）
    2. 自注意力機制（Final Agent）
    """
    
    def __init__(self, state_dim: int, action_dim: int, model_type: str = 'mlp',
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, hidden_dim: int = 256, 
                 use_attention: bool = False, num_heads: int = 4,
                 attention_type: str = 'simple', configs: dict = None, 
                 **kwargs):
        """
        初始化 A2C
        
        :param state_dim: 狀態維度
        :param action_dim: 動作維度（股票數量）
        :param model_type: 模型類型 ('mlp', 'bilstm', 'lstm', etc.)
        :param actor_lr: Actor 學習率
        :param critic_lr: Critic 學習率
        :param gamma: 折扣因子
        :param hidden_dim: 隱藏層維度
        :param use_attention: 是否使用自注意力機制
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型 ('simple' 或 'sequence')
        :param configs: 配置字典
        """
        super().__init__(state_dim, action_dim)
        
        self.num_actions = 3  # 固定 3 個動作: {-1 (賣), 0 (持有), 1 (買)}
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        # ← 计算正确的 Actor 输出维度
        actor_output_dim = action_dim * self.num_actions
        
        print(f"\n{'='*70}")
        print(f"[A2CStrategy] 初始化")
        print(f"{'='*70}")
        print(f"  - 狀態維度: {state_dim}")
        print(f"  - 股票數量: {action_dim}")
        print(f"  - 動作空間: 3 個固定動作 {{-1 (賣), 0 (持有), 1 (買)}}")
        print(f"  - 模型類型: {model_type.upper()}")
        print(f"  - Actor 輸出維度: {actor_output_dim} ({action_dim} × {self.num_actions})")
        print(f"  - 使用注意力機制: {'✓ 是' if use_attention else '✗ 否'}")
        if use_attention:
            print(f"    - 注意力類型: {attention_type}")
            print(f"    - 注意力頭數: {num_heads}")
        print(f"  - 隱藏層維度: {hidden_dim}")
        print(f"  - Actor 學習率: {actor_lr}")
        print(f"  - Critic 學習率: {critic_lr}\n")
        
        # 清理 kwargs
        kwargs.pop('max_timesteps', None)
        kwargs.pop('k', None)
        configs_dict = kwargs.pop('configs', None) or configs
        n_layers = kwargs.pop('n_layers', 2)
        dropout = kwargs.pop('dropout', 0.2)
        
        # ========== 創建 Actor Network ==========
        if use_attention:
            # 使用自注意力機制 Actor（Final Agent）
            print(f"[A2CStrategy] 使用自注意力機制 Actor\n")
            
            if attention_type == 'sequence':
                self.actor = SequenceAttentionActor(
                    state_dim=state_dim,
                    action_dim=actor_output_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
            else:  # simple
                self.actor = SimpleAttentionActor(
                    state_dim=state_dim,
                    action_dim=actor_output_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
        else:
            # 使用標準 Actor（Sub-Agent）
            print(f"[A2CStrategy] 使用標準 {model_type.upper()} Actor\n")
            
            self.actor = ModelFactory.create_actor(
                model_type=model_type,
                input_dim=state_dim,
                output_dim=actor_output_dim,
                actor_type='discrete',
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                configs=configs_dict
            )
        
        self.actor.to(self.device)
        
        # ========== 創建 Critic Network ==========
        print(f"[A2CStrategy] 創建 Critic Network\n")
        
        self.critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            configs=configs_dict
        ).to(self.device)
        
        # ========== 優化器 ==========
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 經驗緩存
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        print(f"{'='*70}\n")
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        選擇動作（帶探索噪音）
        
        :param state: 當前狀態
        :param noise_scale: ε-貪心探索率 (0.0 ~ 1.0)
        :return: 動作陣列 [-1, 0, 1] for each stock
        """
        # ← 確保狀態是正確的形狀
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif state.ndim == 2:
                # (seq_len, features) -> (1, seq_len, features) for BiLSTM
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif state.ndim == 3:
                # (batch, seq_len, features) -> use as-is
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                raise ValueError(f"Invalid state shape: {state.shape}")
        else:
            state_tensor = state.to(self.device) if isinstance(state, torch.Tensor) else state
        
        with torch.no_grad():
            logits = self.actor(state_tensor)  # (batch, action_dim * 3)
            
            # ← 調試檢查：確保 logits 有效
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[WARNING] Actor 輸出包含 NaN 或 Inf！")
                print(f"  - logits 統計: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
                # 用隨機值替換
                logits = torch.randn_like(logits)
            
            # 驗證輸出維度
            if logits.shape[-1] != self.action_dim * self.num_actions:
                raise RuntimeError(f"Actor output dimension mismatch: "
                                 f"expected {self.action_dim * self.num_actions}, "
                                 f"got {logits.shape[-1]}")
            
            # 提取第一個樣本的 logits
            if logits.shape[0] > 1:
                logits = logits[0:1]  # 只取第一個
            
            # 重塑為 (1, num_stocks, 3)
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, self.action_dim, self.num_actions)
            
            # ← 數值穩定的 Softmax（防止溢出）
            # 對每個股票分別進行 softmax
            probs = torch.softmax(logits, dim=-1)  # (1, num_stocks, 3)
            
            # 再次檢查 NaN
            if torch.isnan(probs).any():
                print(f"[WARNING] Softmax 產生 NaN！")
                print(f"  - logits min/max: {logits.min()}/{logits.max()}")
                print(f"  - probs: {probs}")
                # 使用均勻分佈作為後備
                probs = torch.ones_like(probs) / self.num_actions
            
            # ε-貪心探索
            if np.random.random() < noise_scale:
                # 隨機探索：每支股票隨機選擇 {-1, 0, 1}
                actions = np.random.randint(0, self.num_actions, size=self.action_dim)
            else:
                # 根據機率採樣
                actions_list = []
                for stock_idx in range(self.action_dim):
                    # ← 安全地創建分佈
                    prob_vec = probs[0, stock_idx]  # (3,)
                    
                    # 確保機率有效
                    if torch.isnan(prob_vec).any() or prob_vec.sum() <= 0:
                        # 使用均勻分佈
                        prob_vec = torch.ones(self.num_actions, device=self.device) / self.num_actions
                    else:
                        # 正規化機率（以防萬一）
                        prob_vec = prob_vec / prob_vec.sum()
                    
                    try:
                        dist = torch.distributions.Categorical(prob_vec)
                        action = dist.sample().item()
                    except Exception as e:
                        print(f"[WARNING] 無法創建分佈: {e}")
                        print(f"  - prob_vec: {prob_vec}")
                        # 選擇最大機率的動作
                        action = torch.argmax(prob_vec).item()
                    
                    actions_list.append(action)
                actions = np.array(actions_list)
        
        # 轉換索引為實際動作值：[0,1,2] → [-1,0,1]
        return actions - 1
    
    def select_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """
        確定性動作選擇（用於評估，無探索）
        
        :param state: 當前狀態
        :return: 動作陣列 [-1, 0, 1] for each stock
        """
        # ← 確保狀態是正確的形狀
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif state.ndim == 2:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif state.ndim == 3:
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                raise ValueError(f"Invalid state shape: {state.shape}")
        else:
            state_tensor = state
        
        with torch.no_grad():
            logits = self.actor(state_tensor)
            
            # 提取第一個樣本
            if logits.shape[0] > 1:
                logits = logits[0:1]
            
            logits = logits.view(1, self.action_dim, self.num_actions)
            # 選擇機率最高的動作
            actions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # 轉換索引為實際動作值
        return actions - 1
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存儲單步經驗
        
        :param action: 實際動作 [-1, 0, 1]
        """
        self.states.append(state)
        # 轉換動作值回索引：[-1,0,1] → [0,1,2]
        self.actions.append(action + 1)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update_model(self) -> dict:
        """
        更新 Actor 和 Critic 網絡
        
        :return: 損失字典
        """
        if len(self.states) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # ========== 轉換為張量 ==========
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)  # (batch, num_stocks)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device)
        
        batch_size = states.shape[0]
        
        # ========== 計算 Critic 損失 ==========
        values = self.critic(states)  # (batch_size, 1)
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        # TD 目標 = R + γ * V(s')
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        
        # 優勢函數 A = TD 目標 - V(s)
        advantages = td_targets - values
        
        # ← 修復：優勢函數正規化（處理單個樣本情況）
        if batch_size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 如果只有一個樣本，不進行正規化，防止 std=0
        
        # Critic 損失 = MSE(V(s), TD 目標)
        critic_loss = nn.MSELoss()(values, td_targets.detach())
        
        # ========== 計算 Actor 損失 ==========
        logits = self.actor(states)  # (batch_size, action_dim * 3)
        logits = logits.view(batch_size, self.action_dim, self.num_actions)
        
        # ← 數值穩定的 log_softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 檢查 NaN
        if torch.isnan(log_probs).any():
            print(f"[WARNING] log_softmax 產生 NaN，跳過此更新")
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # 提取實際動作的 log 機率
        # actions: (batch_size, num_stocks)
        selected_log_probs = log_probs.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, num_stocks)
        
        # 對所有股票的 log prob 求和
        selected_log_probs = selected_log_probs.sum(dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Actor 損失 = -Σ(log π(a|s) * A(s,a))
        actor_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # ========== 反向傳播和優化 ==========
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ========== 清空經驗 ==========
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'model_type': self.model_type,
                'use_attention': self.use_attention,
                'hidden_dim': self.hidden_dim,
                'num_actions': self.num_actions,
            }
        }, path)
        print(f"[A2CStrategy] 模型已保存到 {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"[A2CStrategy] 模型已從 {path} 加載")
    
    def get_algorithm_name(self) -> str:
        return "A2C"