from agents.mlp import Actor, Critic
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from trader.algos.base_algo import AlgorithmStrategy

class A2CStrategy(AlgorithmStrategy):
    """A2C演算法策略實現"""
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        super().__init__(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
    
    def select_action(self, state):
        """
        A2C 動作選擇策略

        :param state: 當前環境狀態
        :return action: 選擇的動作
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()
    
    def update_model(self, state, action, reward, next_state, done) -> dict:
        """
        A2C 模型更新策略
        
        :param state: 當前狀態
        :param action: 採取的動作
        :param reward: 獲得的獎勵
        :param next_state: 下一狀態
        :param done: 是否結束

        :return: dict - 包含 {'critic_loss': float, 'actor_loss': float} 的損失字典

        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        # Critic update
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        td_target = reward + self.gamma * next_value * (1 - int(done))
        advantage = td_target - value
        
        critic_loss = F.mse_loss(value, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        actor_loss = -dist.log_prob(torch.tensor(action).to(self.device)) * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def save_model(self, path):
        """A2C 保存策略"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """A2C 載入策略"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    def get_algorithm_name(self):
        return "A2C"

