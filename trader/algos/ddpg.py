import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from trader.utils.replay_buffer import ReplayBuffer
from trader.algos.base_algo import AlgorithmStrategy
from trader.factory import ModelFactory

class DDPGStrategy(AlgorithmStrategy):
    """DDPG 演算法 - 支持離散動作空間"""
    
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64,
                 model_type='mlp', k=5, hidden_dim=128, n_layers=2):
        """
        初始化 DDPG
        
        :param k: 動作範圍 [-k, k]
        """
        super().__init__(state_dim, action_dim)
        
        self.k = k
        self.action_range = 2 * k + 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n[DDPGStrategy] 初始化參數:")
        print(f"  - 狀態維度: {state_dim}")
        print(f"  - 動作維度: {action_dim}")
        print(f"  - 動作範圍: [-{k}, {k}]")
        print(f"  - 模型類型: {model_type}\n")
        
        # ← 使用 ModelFactory 創建 Actor（連續型，Tanh 激活）
        self.actor = ModelFactory.create_actor(
            model_type=model_type,
            input_dim=state_dim,
            output_dim=action_dim,  # ← 每支股票一個連續值
            actor_type='continuous',  # ← 指定為連續型
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # ← 使用 ModelFactory 創建 Critic（狀態+動作）
        self.critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim + action_dim,  # ← 輸入狀態和動作
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # 目標網絡
        self.target_actor = ModelFactory.create_actor(
            model_type=model_type,
            input_dim=state_dim,
            output_dim=action_dim,
            actor_type='continuous',
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        self.target_critic = ModelFactory.create_critic(
            model_type=model_type,
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers
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
    
    def _continuous_to_discrete(self, continuous_action: np.ndarray) -> np.ndarray:
        """
        將連續動作 [-1, 1] 轉換為離散索引 [0, 2k]
        
        轉換公式：
        連續值 [-1, 1] → 離散索引 [0, 2k]
        discrete_idx = round((continuous + 1) / 2 * 2k)
        """
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        discrete_action = ((continuous_action + 1.0) / 2.0) * (2 * self.k)
        discrete_action = np.round(discrete_action).astype(int)
        discrete_action = np.clip(discrete_action, 0, 2 * self.k)
        return discrete_action
    
    def select_action(self, state, noise_scale=0.1):
        """選擇動作（含探索噪聲）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor 輸出連續值 [-1, 1]
            continuous_action = self.actor(state_tensor).squeeze(0).detach().cpu().numpy()
        
        # 添加探索噪聲
        noise = np.random.normal(0, noise_scale, size=continuous_action.shape)
        continuous_action = continuous_action + noise
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        
        # 轉換為離散動作
        discrete_action = self._continuous_to_discrete(continuous_action)
        return discrete_action
    
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
        
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.FloatTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ============ Critic 更新 ============
        with torch.no_grad():
            next_continuous_action = self.target_actor(next_state_tensor)
            next_discrete_action = torch.FloatTensor(
                np.array([self._continuous_to_discrete(a.cpu().numpy())
                         for a in next_continuous_action])
            ).to(self.device)
            
            target_q = self.target_critic(
                torch.cat([next_state_tensor, next_discrete_action], dim=1)
            )
            td_target = reward_tensor + self.gamma * target_q * (1 - done_tensor)
        
        current_q = self.critic(torch.cat([state_tensor, action_tensor], dim=1))
        critic_loss = F.mse_loss(current_q, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============ Actor 更新 ============
        policy_continuous_action = self.actor(state_tensor)
        policy_discrete_action = torch.FloatTensor(
            np.array([self._continuous_to_discrete(a.detach().cpu().numpy())
                     for a in policy_continuous_action])
        ).to(self.device)
        
        actor_loss = -self.critic(
            torch.cat([state_tensor, policy_discrete_action], dim=1)
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 軟更新目標網絡
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'k': self.k,
        }, path)
    
    def load_model(self, path: str):
        """加載模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.k = checkpoint.get('k', self.k)
    
    def get_algorithm_name(self) -> str:
        return "DDPG"