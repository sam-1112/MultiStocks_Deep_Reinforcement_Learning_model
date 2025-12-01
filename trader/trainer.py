import numpy as np
from trader.factory import AlgorithmFactory
from trader.envs.factory import EnvironmentFactory
from trader.algos.base_algo import AlgorithmStrategy
from trader.envs.trading_env import TradingEnv
from trader.utils.seed import SeedManager, EnvironmentSeeder

class Trainer:
    def __init__(self, env: TradingEnv = None, algorithm: str = 'ddpg',
                 max_episodes: int = 100, max_timesteps: int = 50000,
                 update_frequency: int = 10, model_type: str = 'mlp',
                 seed: int = 42, **agent_kwargs):
        """
        初始化訓練器
        
        :param env: 環境
        :param algorithm: 演算法名稱
        :param max_episodes: 最大回合數
        :param max_timesteps: 最大步數
        :param update_frequency: 更新頻率
        :param model_type: 模型類型
        :param seed: 隨機種子
        :param agent_kwargs: 演算法參數
        """
        # ← 設置全局隨機種子
        SeedManager.set_seed(seed)
        
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.total_timesteps = 0
        self.update_frequency = update_frequency
        self.seed = seed

        # ← 添加探索率衰減參數
        self.initial_noise_scale = 0.3   # 訓練初期的探索率
        self.final_noise_scale = 0.01     # 訓練後期的探索率
        
        # 創建環境
        if env is None:
            env_config = {
                'num_stocks': agent_kwargs.pop('num_stocks', 30),
                'stock_symbols': agent_kwargs.pop('stock_symbols', []),
                'start_date': agent_kwargs.pop('start_date', '2010-01-01'),
                'end_date': agent_kwargs.pop('end_date', '2023-03-01'),
                'initial_balance': agent_kwargs.pop('initial_balance', 100000),
                'max_steps': agent_kwargs.pop('max_steps', 252),
                'k': agent_kwargs.pop('k', 5),
                'transaction_cost': agent_kwargs.pop('transaction_cost', 0.001),
                'seed': seed  # ← 傳遞種子
            }
            self.train_env = EnvironmentFactory.create_trading_env(env_config)
            self.test_env = EnvironmentFactory.create_trading_env(env_config)
        else:
            self.train_env = env
            self.test_env = self._clone_env(env)
        
        # 創建代理
        self.algo: AlgorithmStrategy = AlgorithmFactory.create(
            algorithm,
            state_dim=self.train_env.state_dim,
            action_dim=self.train_env.action_dim,
            model_type=model_type,
            k=self.train_env.k,
            **agent_kwargs
        )
        
        # 環境種子管理器
        self.seeder = EnvironmentSeeder(seed)
        
        print(f"[Trainer] 初始化完成")
        print(f"  - 演算法: {algorithm}")
        print(f"  - 隨機種子: {seed}")
        print(f"  - 最大回合數: {max_episodes}\n")
    
    def _clone_env(self, env: TradingEnv) -> TradingEnv:
        """克隆環境"""
        config = {
            'num_stocks': env.num_stocks,
            'initial_balance': env.initial_balance,
            'max_steps': env.max_steps,
            'stock_data': env.stock_data.copy(),
            'technical_indicators': env.technical_indicators.copy(),
            'fundamental_data': env.fundamental_data.copy(),
            'k': env.k,
            'transaction_cost': env.transaction_cost,
            'seed': self.seed
        }
        return TradingEnv(config)
    
    def _get_noise_scale(self, episode: int) -> float:
        """
        計算當前回合的探索率（ε-decay 衰減）
        
        :param episode: 當前回合號
        :return: 當前探索率
        """
        progress = episode / self.max_episodes
        # 線性衰減
        noise_scale = (self.initial_noise_scale - self.final_noise_scale) * (1 - progress) + self.final_noise_scale
        return max(noise_scale, self.final_noise_scale)

    def train(self):
        """訓練循環"""
        print(f"[Trainer] 開始訓練...\n")
        
        episode = 0
        episode_rewards = []
        
        while episode < self.max_episodes and self.total_timesteps < self.max_timesteps:
            episode += 1

            current_noise_scale = self._get_noise_scale(episode)
            
            # ← 為每個 reset 設置種子
            reset_seed = self.seeder.get_reset_seed()
            observation, info = self.train_env.reset(seed=reset_seed)
            
            done = False
            truncated = False
            total_reward = 0
            step = 0
            
            print(f"[Trainer] Episode {episode}/{self.max_episodes} "
                  f"(Seed: {reset_seed}, Timesteps: {self.total_timesteps}/{self.max_timesteps})")
            
            while not done and not truncated:
                if self.total_timesteps >= self.max_timesteps:
                    break
                
                # 選擇動作
                action = self.algo.select_action(observation, noise_scale=current_noise_scale)
                
                # 執行動作
                next_observation, reward, done, truncated, info = self.train_env.step(action)
                
                # 存儲經驗
                if hasattr(self.algo, 'store_experience'):
                    self.algo.store_experience(observation, action, reward, next_observation, done)
                
                # 定期更新模型
                if step % self.update_frequency == 0:
                    losses = self.algo.update_model()
                else:
                    losses = {'critic_loss': 0.0, 'actor_loss': 0.0}
                
                observation = next_observation
                total_reward += reward
                step += 1
                self.total_timesteps += 1
                
                if step % 50 == 0:
                    print(f"  Step {step}: Total={self.total_timesteps}, "
                          f"Reward={total_reward:.4f}, "
                          f"Critic loss={losses.get('critic_loss', 0.0):.4f}")
            
            episode_rewards.append(total_reward)
            
            print(f"[Trainer] Episode {episode} 完成:")
            print(f"  - Total reward: {total_reward:.4f}")
            print(f"  - Episode steps: {step}")
            print(f"  - Total timesteps: {self.total_timesteps}/{self.max_timesteps}\n")
            
            if self.total_timesteps >= self.max_timesteps:
                break
        
        print(f"\n{'='*70}")
        print(f"[Trainer] 訓練完成！")
        print(f"[Trainer] Total episodes: {episode}")
        print(f"[Trainer] Average reward: {np.mean(episode_rewards):.4f}\n")
        
        return {
            'episodes': episode,
            'total_timesteps': self.total_timesteps,
            'episode_rewards': episode_rewards
        }
    
    def evaluate(self, num_episodes: int = 10, deterministic_seed: bool = True):
        """
        評估模型性能
        
        :param num_episodes: 評估回合數
        :param deterministic_seed: 是否使用確定性種子
        """
        print(f"\n{'='*70}")
        print(f"[Trainer] 評估模型...")
        print(f"  - Test episodes: {num_episodes}")
        print(f"  - Deterministic seed: {deterministic_seed}\n")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            if deterministic_seed:
                # ← 使用固定種子基於 base_seed 和 episode
                # 確保完全可重現
                eval_seed = (self.seed + episode) % (2**32)
            else:
                eval_seed = self.seeder.get_reset_seed()
            
            observation, info = self.test_env.reset(seed=eval_seed)
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                # ← 評估時：使用 deterministic action selection
                action = self.algo.select_action_deterministic(observation)
                next_observation, reward, done, truncated, info = self.test_env.step(action)
                
                episode_reward += reward
                observation = next_observation
            
            total_rewards.append(episode_reward)
            
            portfolio_value = info.get('portfolio_value', 0)
            print(f"[Trainer] Test Episode {episode+1}/{num_episodes} "
                f"(Seed: {eval_seed}):")
            print(f"  - Reward: {episode_reward:.4f}")
            print(f"  - Portfolio value: ${portfolio_value:,.2f}\n")
        
        avg_reward = np.mean(total_rewards)
        print(f"[Trainer] 平均獎勵: {avg_reward:.4f}\n")
        
        return total_rewards
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.algo.save_model(save_path)
        print(f"[Trainer] 模型已保存到 {save_path}")
    
    def load_model(self, path: str):
        """載入模型"""
        self.algo.load_model(path)
        print(f"[Trainer] 模型已從 {path} 加載")