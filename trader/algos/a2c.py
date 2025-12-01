import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from trader.utils.replay_buffer import ReplayBuffer
from trader.algos.base_algo import AlgorithmStrategy
from trader.factory import ModelFactory

class A2CStrategy(AlgorithmStrategy):
    """A2C 演算法 - 支持離散動作空間"""
    
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, k=5, model_type='mlp', buffer_size=10000, 
                 batch_size=64, hidden_dim=128, n_layers=2):
        """
        初始化 A2C
        
        :param k: 動作範圍 [-k, k]
        :param action_dim: 股票數量
        """
        super().__init__(state_dim, action_dim)
        
        self.k = k
        self.action_range = 2 * k + 1  # 離散動作數（例如 k=5 時為 11）
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n[A2CStrategy] 初始化參數:")
        print(f"  - 狀態維度: {state_dim}")
        print(f"  - 動作維度: {action_dim} (股票數)")
        print(f"  - 動作範圍: [-{k}, {k}]")
        print(f"  - 每支股票的動作數: {self.action_range}")
        print(f"  - Actor 輸出維度: {action_dim * self.action_range}")
        print(f"  - 模型類型: {model_type}\n")
        
        # ← 使用 ModelFactory 創建 Actor（離散型，無激活）
        # 輸出維度 = action_dim * action_range
        # 例如：5 支股票 * 11 個動作 = 55 個輸出（每個對應一個動作的 logit）
        self.actor = ModelFactory.create_actor(
            model_type=model_type,
            input_dim=state_dim,
            output_dim=action_dim * self.action_range,  # ← 重要！
            actor_type='discrete',  # ← 指定為離散型
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # ← 使用 ModelFactory 創建 Critic（只輸入狀態）
        self.critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # 優化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 移到設備
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
    
    def select_action(self, state, noise_scale=0.1):
        """選擇動作（ε-貪心）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor 輸出 (1, action_dim * action_range)
            action_logits = self.actor(state_tensor).squeeze(0).detach().cpu().numpy()
            # shape: (55,) 如果 action_dim=5, action_range=11
        
        # 重新整形為 (action_dim, action_range)
        action_logits = action_logits.reshape(self.action_dim, self.action_range)
        # shape: (5, 11)
        
        actions = np.zeros(self.action_dim, dtype=int)
        
        for i in range(self.action_dim):
            if np.random.random() < noise_scale:
                # 探索：隨機選擇
                actions[i] = np.random.randint(0, self.action_range)
            else:
                # 開採：選擇最大概率的動作
                actions[i] = np.argmax(action_logits[i])
        
        return actions
    
    def select_action_deterministic(self, state):
        """無噪聲的動作選擇"""
        return self.select_action(state, noise_scale=0.0)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update_model(self) -> dict:
        """更新模型"""
        if len(self.replay_buffer) < self.batch_size:
            return {'critic_loss': 0.0, 'actor_loss': 0.0}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # ← 獲取實際批次大小（可能小於 self.batch_size）
        actual_batch_size = states.shape[0]
        
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)  # (batch_size, action_dim)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ============ Critic 更新 ============
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)  # (actual_batch_size, 1)
            td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        
        value = self.critic(state_tensor)  # (actual_batch_size, 1)
        critic_loss = F.mse_loss(value, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============ Actor 更新 ============
        advantage = (td_target - value.detach()).squeeze()  # (actual_batch_size,)
        
        # Actor 輸出 (actual_batch_size, action_dim * action_range)
        action_logits = self.actor(state_tensor)
        # 重新整形為 (actual_batch_size, action_dim, action_range)
        action_logits = action_logits.reshape(
            actual_batch_size, self.action_dim, self.action_range
        )
        
        actor_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(actual_batch_size):  # ← 使用實際大小
            for j in range(self.action_dim):
                stock_logits = action_logits[i, j]  # (action_range,)
                prob = F.softmax(stock_logits, dim=0)
                action_idx = action_tensor[i, j]
                log_prob = torch.log(prob[action_idx] + 1e-8)
                actor_loss = actor_loss - log_prob * advantage[i]
        
        actor_loss = actor_loss / (actual_batch_size * self.action_dim)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'k': self.k,
        }, path)
    
    def load_model(self, path: str):
        """加載模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.k = checkpoint.get('k', self.k)
    
    def get_algorithm_name(self) -> str:
        return "A2C"