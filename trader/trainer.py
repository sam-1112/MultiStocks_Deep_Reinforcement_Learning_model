import os
from typing import Dict, Optional
import numpy as np
from trader.factory import AlgorithmFactory
from trader.envs.factory import EnvironmentFactory
from trader.algos.base_algo import AlgorithmStrategy
from trader.envs.trading_env import TradingEnv
from trader.parallel_trainer import SubAgentEnsemble
from trader.utils.seed import SeedManager, EnvironmentSeeder
from tqdm import tqdm
from trader.utils.logging import TrainingLogger

class Trainer:
    def __init__(self, agent_name: str = 'default_agent', env: TradingEnv = None, algorithm: str = 'ddpg',
                 max_episodes: int = 100,
                 update_frequency: int = 10, model_type: str = 'mlp',
                 seed: int = 42, agent_mode: str = 'multi-agent',
                 use_attention: bool = False, num_heads: int = 4,
                 attention_type: str = 'simple', **agent_kwargs):
        """
        初始化訓練器
        
        :param agent_name: 代理名稱
        :param env: 環境
        :param algorithm: 演算法名稱
        :param max_episodes: 最大回合數
        :param update_frequency: 更新頻率
        :param model_type: 模型類型
        :param seed: 隨機種子
        :param agent_mode: 代理模式 ('single-agent' 或 'multi-agent')
        :param use_attention: 是否使用自注意力機制（僅 Final Agent）
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型
        :param agent_kwargs: 演算法參數
        """
        # ← 設置全局隨機種子
        SeedManager.set_seed(seed)
        self.agent_name = agent_name
        self.max_episodes = max_episodes
        self.total_timesteps = 0
        self.update_frequency = update_frequency
        self.seed = seed
        self.agent_mode = agent_mode
        self.num_of_agents = agent_kwargs.pop('num_of_subagents', 1) if agent_mode == 'multi-agent' else 1
        
        # ← 注意力機制參數
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attention_type = attention_type

        # ← 添加探索率衰減參數
        self.initial_noise_scale = 0.3   # 訓練初期的探索率
        self.final_noise_scale = 0.01     # 訓練後期的探索率
        
        # 移除 max_timesteps（如果有傳入的話）
        agent_kwargs.pop('max_timesteps', None)
        
        # 從 agent_kwargs 提取環境相關參數（避免傳給演算法）
        env_related_keys = ['num_stocks', 'stock_symbols', 'start_date', 'end_date', 
                           'initial_balance', 'max_steps', 'transaction_cost']
        env_params = {k: agent_kwargs.pop(k) for k in env_related_keys if k in agent_kwargs}
        
        # 從 agent_kwargs 提取 k（如果有的話）
        k_value = agent_kwargs.pop('k', 1)
        
        # 創建環境
        if env is None:
            env_config = {
                'num_stocks': env_params.get('num_stocks', 30),
                'stock_symbols': env_params.get('stock_symbols', []),
                'start_date': env_params.get('train_date_start', '2010-01-01'),
                'end_date': env_params.get('train_date_end', '2023-03-01'),
                'initial_balance': env_params.get('initial_balance', 100000),
                'max_steps': env_params.get('max_steps', 252),
                'k': k_value,
                'transaction_cost': env_params.get('transaction_cost', 0.001),
                'seed': seed
            }
            self.train_env = EnvironmentFactory.create_trading_env(env_config)
            self.test_env = EnvironmentFactory.create_trading_env(env_config)
        else:
            self.train_env = env
            self.test_env = self._clone_env(env)
        
        # 創建代理（傳遞注意力參數）
        self.algo: AlgorithmStrategy = AlgorithmFactory.create(
            algorithm,
            state_dim=self.train_env.state_dim,
            action_dim=self.train_env.action_dim,
            model_type=model_type,
            k=self.train_env.k,  # 使用環境的 k 值
            use_attention=use_attention,
            num_heads=num_heads,
            attention_type=attention_type,
            **agent_kwargs
        )
        
        # 環境種子管理器
        self.seeder = EnvironmentSeeder(seed)
        
        print(f"[Trainer] 初始化完成")
        print(f"  - 代理名稱: {self.agent_name}")
        print(f"  - 演算法: {self.algo.get_algorithm_name()}")
        print(f"  - 隨機種子: {self.seed}")
        print(f"  - 使用注意力機制: {self.use_attention}")
        if self.use_attention:
            print(f"  - 注意力類型: {self.attention_type}")
            print(f"  - 注意力頭數: {self.num_heads}")
        print(f"  - 最大回合數: {self.max_episodes}\n")

        # 初始化日誌記錄器
        self.logger = TrainingLogger(
            agent_name=self.agent_name,
            algorithm=self.algo.get_algorithm_name(),
            agent_mode=self.agent_mode,
            log_dir='./logs',
            save_frequency=10
        )
    
    def _clone_env(self, env: TradingEnv) -> TradingEnv:
        """
        克隆環境
        
        如果是 FinalAgentEnv，保持其類型和 SubAgentEnsemble 引用
        如果是普通 TradingEnv，進行深度克隆
        """
        # ★★★ 檢查是否為 FinalAgentEnv ★★★
        if hasattr(env, 'base_env') and hasattr(env, 'ensemble'):
            # 這是 FinalAgentEnv
            from trader.envs.final_agent_env import FinalAgentEnv
            
            # 克隆基礎環境
            base_config = {
                'num_stocks': env.base_env.num_stocks,
                'initial_balance': env.base_env.initial_balance,
                'max_steps': env.base_env.max_steps,
                'stock_data': env.base_env.stock_data.copy(),
                'technical_indicators': env.base_env.technical_indicators.copy(),
                'fundamental_data': env.base_env.fundamental_data.copy(),
                'k': env.base_env.k,
                'transaction_cost': env.base_env.transaction_cost,
                'seed': self.seed,
                # ★ 保持相同的模型配置
                'model_type': env.base_env.model_type,
                'window_size': env.base_env.window_size,

            }
            base_env_clone = TradingEnv(base_config)
            
            # 使用相同的 ensemble 創建新的 FinalAgentEnv
            return FinalAgentEnv(base_env_clone, env.ensemble)
        else:
            # 普通 TradingEnv
            config = {
                'num_stocks': env.num_stocks,
                'initial_balance': env.initial_balance,
                'max_steps': env.max_steps,
                'stock_data': env.stock_data.copy(),
                'technical_indicators': env.technical_indicators.copy(),
                'fundamental_data': env.fundamental_data.copy(),
                'k': env.k,
                'transaction_cost': env.transaction_cost,
                'seed': self.seed,
                # ★ 保持相同的模型配置
                'model_type': env.model_type,
                'window_size': env.window_size,
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

    def single_agent_train(self):
        """單代理訓練循環"""
        print(f"[Trainer] 開始訓練...\n")
        
        episode = 0
        episode_rewards = []

        # 進度條：只追蹤 Episodes
        episode_pbar = tqdm(
            total=self.max_episodes,
            desc=f"{self.agent_name} - Episodes",
            unit="ep",
            position=0,
            leave=True,
            colour='green',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        while episode < self.max_episodes:
            self.logger.start_episode(episode)
            episode += 1

            current_noise_scale = self._get_noise_scale(episode)
            
            # ← 為每個 reset 設置種子
            reset_seed = self.seeder.get_reset_seed()
            observation, info = self.train_env.reset(seed=reset_seed)
            
            # ← 記錄初始 portfolio 值
            initial_portfolio = info.get('portfolio_value', self.train_env.initial_balance)
            
            done = False
            truncated = False
            total_reward = 0
            step = 0
            episode_critic_loss = 0.0
            episode_actor_loss = 0.0
            update_count = 0
            
            while not done and not truncated:
                # 選擇動作
                action = self.algo.select_action(observation, noise_scale=current_noise_scale)
                
                # 執行動作
                next_observation, reward, done, truncated, info = self.train_env.step(action)
                
                # 記錄每步資訊
                self.logger.log_step(step, action, reward, info['portfolio_value'], info['balance'])

                # 存儲經驗
                if hasattr(self.algo, 'store_experience'):
                    self.algo.store_experience(observation, action, reward, next_observation, done)
                
                # 定期更新模型
                if step % self.update_frequency == 0:
                    losses = self.algo.update_model()
                    episode_critic_loss += losses.get('critic_loss', 0.0)
                    episode_actor_loss += losses.get('actor_loss', 0.0)
                    update_count += 1
                
                observation = next_observation
                total_reward += reward
                step += 1
                self.total_timesteps += 1
            
            # ← 計算平均 loss（移到 while 迴圈外）
            avg_critic_loss = episode_critic_loss / max(update_count, 1)
            avg_actor_loss = episode_actor_loss / max(update_count, 1)
            
            # ← Episode 結束後記錄（移到 while 迴圈外）
            self.logger.end_episode(
                episode_reward=total_reward,
                episode_length=step,
                actor_loss=avg_actor_loss,
                critic_loss=avg_critic_loss,
                noise_scale=current_noise_scale,
                initial_portfolio=initial_portfolio,
                final_portfolio=info['portfolio_value']
            )
            
            episode_rewards.append(total_reward)
            
            # 更新 episode 進度條
            episode_pbar.update(1)
            episode_pbar.set_postfix({
                'reward': f'{total_reward:.2f}',
                'avg_reward': f'{np.mean(episode_rewards[-10:]):.2f}',
                'c_loss': f'{avg_critic_loss:.4f}',
                'a_loss': f'{avg_actor_loss:.4f}',
                'steps': step
            })

        # 訓練結束
        self.logger.finalize()

        # 關閉進度條
        episode_pbar.close()
        
        print(f"\n{'='*70}")
        print(f"[Trainer] ✅ Agent '{self.agent_name}' 訓練完成！")
        print(f"  - Total episodes: {episode}")
        print(f"  - Total timesteps: {self.total_timesteps}")
        print(f"  - Average reward: {np.mean(episode_rewards):.4f}")
        print(f"  - Best reward: {np.max(episode_rewards):.4f}")
        print(f"  - Last 10 avg reward: {np.mean(episode_rewards[-10:]):.4f}\n")
        
        return {
            'episodes': episode,
            'total_timesteps': self.total_timesteps,
            'episode_rewards': episode_rewards
        }

    def multi_agent_train(self):
        """多代理訓練"""
        for agent_id in range(self.num_of_agents):
            print(f"\n{'='*30} 訓練代理 {agent_id+1}/{self.num_of_agents} {'='*30}\n")
            self.single_agent_train()

    def train(self):
        """訓練循環"""
        if self.agent_mode == 'single-agent':
            return self.single_agent_train()
        elif self.agent_mode == 'multi-agent':
            self.multi_agent_train()
        else:
            raise ValueError(f"未知的代理模式: {self.agent_mode}")
    
    def evaluate(self, deterministic_seed: bool = True):
        """
        評估模型性能 - 走完整個測試期間的所有 timesteps
        
        :param deterministic_seed: 是否使用確定性種子
        """
        print(f"\n{'='*70}")
        print(f"[Trainer] 🧪 開始回測 Agent: {self.agent_name}")
        print(f"  - Deterministic seed: {deterministic_seed}\n")
        
        if deterministic_seed:
            eval_seed = self.seed
        else:
            eval_seed = self.seeder.get_reset_seed()
        
        observation, info = self.test_env.reset(seed=eval_seed)
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        # 記錄每日數據
        daily_rewards = []
        daily_portfolio_values = []
        daily_actions = []
        
        # ★★★ 計算總交易日數 ★★★
        # 從環境的股票數據獲取交易日數量
        if hasattr(self.test_env, 'base_env'):
            # FinalAgentEnv 的情況
            total_trading_days = self.test_env.base_env.stock_data.shape[0]
        else:
            # 普通 TradingEnv 的情況
            total_trading_days = self.test_env.stock_data.shape[0]
        
        print(f"  - Total trading days: {total_trading_days}\n")
        
        # ★★★ 使用計算出的交易日數作為進度條上限 ★★★
        eval_pbar = tqdm(
            total=total_trading_days,
            desc=f"🧪 {self.agent_name} - Backtesting",
            unit="day",
            colour='yellow',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        initial_portfolio = info.get('portfolio_value', self.test_env.initial_balance)
        
        while not done and not truncated:
            # 使用 deterministic action selection（無噪音）
            action = self.algo.select_action_deterministic(observation)
            next_observation, reward, done, truncated, info = self.test_env.step(action)
            
            total_reward += reward
            step += 1
            
            # 記錄每日數據
            daily_rewards.append(reward)
            daily_portfolio_values.append(info.get('portfolio_value', 0))
            daily_actions.append(action.copy())
            
            observation = next_observation
            
            # 更新進度條
            eval_pbar.update(1)
            eval_pbar.set_postfix({
                'portfolio': f"${info.get('portfolio_value', 0):,.0f}",
                'reward': f'{reward:.4f}',
                'total': f'{total_reward:.2f}'
            })
        
        eval_pbar.close()
        
        # 計算績效指標
        final_portfolio = info.get('portfolio_value', 0)
        # ← 保護：避免除以零
        total_return = 0.0 if initial_portfolio == 0 else (final_portfolio - initial_portfolio) / initial_portfolio * 100
        
        # 計算每日報酬率（添加保護機制）
        daily_returns = []
        if len(daily_portfolio_values) > 1:
            for i in range(1, len(daily_portfolio_values)):
                prev_value = daily_portfolio_values[i - 1]
                curr_value = daily_portfolio_values[i]
                # ← 避免除以零和無效值
                if prev_value > 0:
                    daily_ret = (curr_value - prev_value) / prev_value
                    if np.isfinite(daily_ret):
                        daily_returns.append(daily_ret)
        
        daily_returns = np.array(daily_returns) if daily_returns else np.array([])
        
        # 計算 Sharpe Ratio (假設無風險利率為 0，年化)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 計算最大回撤 (Maximum Drawdown)（添加保護機制）
        max_drawdown = 0.0
        if len(daily_portfolio_values) > 0:
            peak = np.maximum.accumulate(daily_portfolio_values)
            # ← 避免除以零
            drawdown = np.zeros_like(peak, dtype=float)
            for i, p in enumerate(peak):
                if p > 0:
                    drawdown[i] = (p - daily_portfolio_values[i]) / p
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0.0
        
        print(f"\n{'='*70}")
        print(f"[Trainer] 📊 回測結果 - {self.agent_name}")
        print(f"{'='*70}")
        print(f"  📅 回測期間: {step} 個交易日 (佔總交易日數 {step}/{total_trading_days})")
        print(f"  💰 初始資金: ${initial_portfolio:,.2f}")
        print(f"  💵 最終資金: ${final_portfolio:,.2f}")
        print(f"  📈 總報酬率: {total_return:.2f}%")
        print(f"  📉 最大回撤: {max_drawdown:.2f}%")
        print(f"  📊 Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  🎯 累積獎勵: {total_reward:.4f}")
        print(f"{'='*70}\n")
        
        return {
            'agent_name': self.agent_name,
            'total_steps': step,
            'total_trading_days': total_trading_days,
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_reward': total_reward,
            'daily_rewards': daily_rewards,
            'daily_portfolio_values': daily_portfolio_values,
            'daily_actions': daily_actions
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.algo.save_model(save_path)
        print(f"[Trainer] 模型已保存到 {save_path}")
    
    def load_model(self, path: str):
        """載入模型"""
        self.algo.load_model(path)
        print(f"[Trainer] 模型已從 {path} 加載")


class HierarchicalTrainer:
    """
    分層訓練器
    
    1. 平行訓練 Sub-Agents（不使用注意力機制）
    2. 載入 Sub-Agent 模型
    3. 訓練 Final Agent（使用注意力機制）
    """
    
    def __init__(self, config: Dict, seed: int = 42):
        """
        初始化分層訓練器
        
        Args:
            config: 完整配置字典
            seed: 隨機種子
        """
        from trader.parallel_trainer import ParallelSubAgentTrainer, SubAgentEnsemble
        from trader.envs.final_agent_env import FinalAgentEnv
        
        self.config = config
        self.seed = seed
        
        # Sub-Agent 訓練器
        self.parallel_trainer = ParallelSubAgentTrainer(config, seed)
        
        # Sub-Agent 集成器（訓練後初始化）
        self.ensemble: Optional[SubAgentEnsemble] = None
        
        # Final Agent 環境和訓練器（訓練 Sub-Agent 後初始化）
        self.final_env: Optional[FinalAgentEnv] = None
        self.final_trainer: Optional[Trainer] = None
        
        self.sub_agent_results = {}
    
    def train(self, num_workers: int = None):
        """
        完整訓練流程
        
        1. 平行訓練 Sub-Agents（無注意力機制）
        2. 載入 Sub-Agent 模型並建立集成器
        3. 創建 Final Agent 環境
        4. 訓練 Final Agent（使用注意力機制）
        """
        from trader.parallel_trainer import ParallelSubAgentTrainer, SubAgentEnsemble
        from trader.envs.final_agent_env import FinalAgentEnv
        from trader.envs.factory import EnvironmentFactory
        
        print(f"\n{'='*70}")
        print(f"[HierarchicalTrainer] 🚀 開始分層訓練")
        print(f"{'='*70}\n")
        
        # ========== 階段 1: 平行訓練 Sub-Agents ==========
        print(f"\n{'='*70}")
        print(f"[階段 1/3] 平行訓練 Sub-Agents（無注意力機制）")
        print(f"{'='*70}\n")
        
        self.sub_agent_results = self.parallel_trainer.train_sub_agents_parallel(num_workers)
        
        # ========== 階段 2: 載入 Sub-Agent 模型 ==========
        print(f"\n{'='*70}")
        print(f"[階段 2/3] 載入 Sub-Agent 模型並建立集成器")
        print(f"{'='*70}\n")
        
        self.ensemble = self._create_ensemble()
        
        # ========== 階段 3: 訓練 Final Agent ==========
        print(f"\n{'='*70}")
        print(f"[階段 3/3] 訓練 Final Agent（使用注意力機制）")
        print(f"{'='*70}\n")
        
        self._train_final_agent()
        
        print(f"\n{'='*70}")
        print(f"[HierarchicalTrainer] ✅ 分層訓練完成！")
        print(f"{'='*70}\n")
    
    def _create_ensemble(self) -> SubAgentEnsemble:
        """建立 Sub-Agent 集成器"""
        from trader.parallel_trainer import SubAgentEnsemble
        from trader.envs.factory import EnvironmentFactory
        
        data_cfg = self.config['data']
        env_cfg = self.config['env']
        hyper_cfg = self.config['hyperparameters']
        
        # 創建一個臨時環境來獲取 state_dim 和 action_dim
        temp_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(data_cfg['ticker_list']),
            'stock_symbols': data_cfg['ticker_list'],
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': data_cfg['date_start'],
            'end_date': data_cfg['date_end'],
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': self.seed,
        })
        
        # 準備模型載入資訊
        model_paths = {}
        sub_agents_cfg = self.config['agent_mode'].get('sub_agents', [])
        
        for i, sub_agent in enumerate(sub_agents_cfg):
            agent_name = sub_agent.get('name', f'Sub-Agent-{i}')
            
            if agent_name in self.sub_agent_results:
                result = self.sub_agent_results[agent_name]
                if result['status'] == 'success':
                    model_paths[agent_name] = {
                        'path': result['model_path'],
                        'algorithm': sub_agent.get('algorithm', 'a2c'),
                        'model_type': sub_agent.get('model_type', 'mlp'),
                        'state_dim': temp_env.state_dim,
                        'action_dim': temp_env.action_dim,
                        'hidden_dim': int(hyper_cfg['hidden_dim']),
                    }
        
        return SubAgentEnsemble(model_paths)
    
    def _train_final_agent(self):
        """訓練 Final Agent（使用注意力機制）"""
        from trader.envs.final_agent_env import FinalAgentEnv
        from trader.envs.factory import EnvironmentFactory
        
        data_cfg = self.config['data']
        env_cfg = self.config['env']
        train_cfg = self.config['training']
        hyper_cfg = self.config['hyperparameters']
        final_agent_cfg = self.config['agent_mode'].get('final_agent', {})
        
        # ★ 提取模型配置
        model_type = final_agent_cfg.get('model_type', 'mlp')
    

        # ★★★ 提取訓練和測試日期 ★★★
        train_start = data_cfg.get('train_date_start', data_cfg.get('date_start', '2010-01-01'))
        train_end = data_cfg.get('train_date_end', data_cfg.get('date_end', '2021-10-01'))
        test_start = data_cfg.get('test_date_start', data_cfg.get('date_start', '2021-10-01'))
        test_end = data_cfg.get('test_date_end', data_cfg.get('date_end', '2023-03-01'))
        
        print(f"\n[HierarchicalTrainer] 📅 日期範圍:")
        print(f"  - 訓練期間: {train_start} 至 {train_end}")
        print(f"  - 測試期間: {test_start} 至 {test_end}\n")
        
        # ★★★ 創建訓練環境（使用訓練日期） ★★★
        train_base_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(data_cfg['ticker_list']),
            'stock_symbols': data_cfg['ticker_list'],
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': train_start,
            'end_date': train_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': self.seed + 100,
            'agent_type': final_agent_cfg.get('agent_type', 'final'),
            # ★ 模型配置
            'model_type': model_type,
            'window_size': env_cfg.get('window_size', 10),
        })
        
        # 包裝為 Final Agent 訓練環境
        train_final_env = FinalAgentEnv(train_base_env, self.ensemble)
        
        # ★★★ 創建測試環境（使用測試日期） ★★★
        test_base_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(data_cfg['ticker_list']),
            'stock_symbols': data_cfg['ticker_list'],
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': test_start,
            'end_date': test_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': self.seed + 101,
            'agent_type': final_agent_cfg.get('agent_type', 'final'),
            # ★ 模型配置
            'model_type': model_type,
            'window_size': env_cfg.get('window_size', 10),
        })
        
        # 包裝為 Final Agent 測試環境
        test_final_env = FinalAgentEnv(test_base_env, self.ensemble)
        
        # ★★★ 關鍵修正：使用 FinalAgentEnv 的 state_dim ★★★
        # FinalAgentEnv.state_dim = base_state_dim + q_values_dim
        print(f"[HierarchicalTrainer] Final Agent 環境設置:")
        print(f"  - Base state dim: {train_base_env.state_dim}")
        print(f"  - Q-values dim: {self.ensemble.get_q_values_dim()}")
        print(f"  - Total state dim: {train_final_env.state_dim}")
        print(f"  - Action dim: {train_final_env.action_dim}")
        
        # 從配置提取 Final Agent 的注意力參數
        use_attention = final_agent_cfg.get('use_attention', False)
        num_heads = final_agent_cfg.get('num_heads', 4)
        attention_type = final_agent_cfg.get('attention_type', 'simple')
        
        # 創建 Final Agent 訓練器（使用正確的 state_dim 和注意力參數）
        self.final_trainer = Trainer(
            agent_name=final_agent_cfg.get('name', 'Final_Agent'),
            env=train_final_env,  # ★★★ 使用訓練環境 ★★★
            algorithm=final_agent_cfg.get('algorithm', 'ddpg'),
            max_episodes=train_cfg['max_episodes'],
            update_frequency=train_cfg['update_frequency'],
            model_type=final_agent_cfg.get('model_type', 'mlp'),
            seed=self.seed + 100,
            agent_mode='single-agent',
            use_attention=use_attention,
            num_heads=num_heads,
            attention_type=attention_type,
            actor_lr=float(hyper_cfg['actor_lr']),
            critic_lr=float(hyper_cfg['critic_lr']),
            gamma=float(hyper_cfg['gamma']),
            hidden_dim=int(hyper_cfg['hidden_dim']),
            batch_size=int(hyper_cfg['batch_size']),
        )
        
        # ★★★ 手動設置測試環境 ★★★
        self.final_trainer.test_env = test_final_env
        
        # 訓練
        self.final_trainer.train()
        
        # 儲存模型
        os.makedirs('./models', exist_ok=True)
        final_model_path = f"./models/{final_agent_cfg.get('name', 'Final_Agent')}_agent.pth"
        self.final_trainer.save_model(final_model_path)

    def initialize_for_evaluation(self):
        """
        初始化評估環境
        
        在評估模式下，從已保存的模型加載 Sub-Agents 和 Final Agent
        """
        from trader.parallel_trainer import SubAgentEnsemble
        from trader.envs.final_agent_env import FinalAgentEnv
        from trader.envs.factory import EnvironmentFactory
        
        data_cfg = self.config['data']
        env_cfg = self.config['env']
        train_cfg = self.config['training']
        hyper_cfg = self.config['hyperparameters']
        final_agent_cfg = self.config['agent_mode'].get('final_agent', {})
        
        # ★★★ 提取測試日期 ★★★
        test_start = data_cfg.get('test_date_start', data_cfg.get('date_start', '2021-10-01'))
        test_end = data_cfg.get('test_date_end', data_cfg.get('date_end', '2023-03-01'))
        
        print(f"\n[HierarchicalTrainer] 初始化評估環境...")
        print(f"  📅 評估期間: {test_start} 至 {test_end}\n")
        
        # ========== 第一步：建立 Sub-Agent 集成器 ==========
        if self.ensemble is None:
            print(f"\n  [階段 1/3] 載入 Sub-Agent 模型...")
            sub_agent_model_dir = './models/sub_agents'
            
            if not os.path.exists(sub_agent_model_dir):
                raise FileNotFoundError(f"Sub-Agent 模型目錄不存在: {sub_agent_model_dir}")
            
            model_paths = {}
            sub_agents_cfg = self.config['agent_mode'].get('sub_agents', [])
            
            # ★★★ 使用測試日期創建臨時環境 ★★★
            temp_env = EnvironmentFactory.create_trading_env({
                'num_stocks': len(data_cfg['ticker_list']),
                'stock_symbols': data_cfg['ticker_list'],
                'initial_balance': env_cfg['initial_balance'],
                'max_steps': env_cfg['max_steps'],
                'start_date': test_start,
                'end_date': test_end,
                'transaction_cost': env_cfg['transaction_cost'],
                'seed': self.seed,
            })
            
            print(f"    基礎環境維度: state_dim={temp_env.state_dim}, action_dim={temp_env.action_dim}")
            
            for i, sub_agent in enumerate(sub_agents_cfg):
                agent_name = sub_agent.get('name', f'Sub-Agent-{i}')
                model_file = os.path.join(sub_agent_model_dir, f"{agent_name}_agent.pth")
                
                if os.path.exists(model_file):
                    model_paths[agent_name] = {
                        'path': model_file,
                        'algorithm': sub_agent.get('algorithm', 'a2c'),
                        'model_type': sub_agent.get('model_type', 'mlp'),
                        'state_dim': temp_env.state_dim,
                        'action_dim': temp_env.action_dim,
                        'hidden_dim': int(hyper_cfg.get('hidden_dim', 256)),
                    }
                    print(f"    ✓ {agent_name}: {model_file}")
                else:
                    print(f"    ✗ {agent_name}: 未找到 {model_file}")
            
            if not model_paths:
                raise ValueError(f"未能加載任何 Sub-Agent 模型，請確認 {sub_agent_model_dir} 目錄中是否有模型文件")
            
            print(f"\n    建立 Sub-Agent 集成器...")
            self.ensemble = SubAgentEnsemble(model_paths)
            print(f"    ✓ Sub-Agent 集成器建立完成 ({len(model_paths)} 個代理)")
        
        # ========== 第二步：驗證和建立 Final Agent 環境 ==========
        if self.final_env is None:
            print(f"\n  [階段 2/3] 創建 Final Agent 環境...")
            
            # ★★★ 使用測試日期創建評估環境 ★★★
            base_env = EnvironmentFactory.create_trading_env({
                'num_stocks': len(data_cfg['ticker_list']),
                'stock_symbols': data_cfg['ticker_list'],
                'initial_balance': env_cfg['initial_balance'],
                'max_steps': env_cfg['max_steps'],
                'start_date': test_start,
                'end_date': test_end,
                'transaction_cost': env_cfg['transaction_cost'],
                'seed': self.seed + 100,
                'agent_type': final_agent_cfg.get('agent_type', 'final'),
            })
            
            # 包裝為 Final Agent 環境
            self.final_env = FinalAgentEnv(base_env, self.ensemble)
            
            print(f"    ✓ Final Agent 環境已建立")
            print(f"      - Base state dim: {self.final_env.base_state_dim}")
            print(f"      - Q-values dim: {self.final_env.q_values_dim}")
            print(f"      - Total state dim: {self.final_env.state_dim}")
            
            # ★★★ 驗證狀態維度 ★★★
            try:
                test_obs, _ = self.final_env.reset(seed=self.seed)
                actual_state_dim = len(test_obs)
                print(f"    ✓ 測試 reset 成功，實際 state dim: {actual_state_dim}")
                
                if actual_state_dim != self.final_env.state_dim:
                    print(f"    ⚠️ 警告：狀態維度不一致 (預期: {self.final_env.state_dim}, 實際: {actual_state_dim})")
            except Exception as e:
                print(f"    ⚠️ 警告：測試 reset 失敗: {e}")
        
        # ========== 第三步：建立和加載 Final Agent ==========
        if self.final_trainer is None:
            print(f"\n  [階段 3/3] 創建並加載 Final Agent 訓練器...")
            
            use_attention = final_agent_cfg.get('use_attention', False)
            num_heads = final_agent_cfg.get('num_heads', 4)
            attention_type = final_agent_cfg.get('attention_type', 'simple')
            
            # ★★★ 使用評估環境的實際 state_dim ★★★
            actual_state_dim = self.final_env.state_dim
            print(f"    使用評估環境的 state_dim: {actual_state_dim}")
            
            # 創建訓練器
            self.final_trainer = Trainer(
                agent_name=final_agent_cfg.get('name', 'Final_Agent'),
                env=self.final_env,
                algorithm=final_agent_cfg.get('algorithm', 'ddpg'),
                max_episodes=train_cfg['max_episodes'],
                update_frequency=train_cfg['update_frequency'],
                model_type=final_agent_cfg.get('model_type', 'mlp'),
                seed=self.seed + 100,
                agent_mode='single-agent',
                use_attention=use_attention,
                num_heads=num_heads,
                attention_type=attention_type,
                actor_lr=float(hyper_cfg['actor_lr']),
                critic_lr=float(hyper_cfg['critic_lr']),
                gamma=float(hyper_cfg['gamma']),
                hidden_dim=int(hyper_cfg.get('hidden_dim', 256)),
                batch_size=int(hyper_cfg.get('batch_size', 64)),
            )
            
            # ★★★ 驗證載入的模型與當前環境維度是否匹配 ★★★
            final_agent_name = final_agent_cfg.get('name', 'Final_Agent')
            model_path = f"./models/{final_agent_name}_agent.pth"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Final Agent 模型未找到: {model_path}")
            
            # 在載入前檢查模型的狀態維度
            print(f"    檢查模型兼容性...")
            try:
                # 嘗試載入並檢查第一個層的輸入維度
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 檢查模型中的第一個線性層
                if hasattr(self.final_trainer.algo, 'actor') and self.final_trainer.algo.actor is not None:
                    actor_model = self.final_trainer.algo.actor
                    # 獲取第一個層的權重
                    for module in actor_model.modules():
                        if isinstance(module, torch.nn.Linear):
                            model_input_dim = module.weight.shape[1]
                            print(f"      模型期望的輸入維度: {model_input_dim}")
                            print(f"      環境提供的狀態維度: {actual_state_dim}")
                            if model_input_dim != actual_state_dim:
                                print(f"      ⚠️ 維度不匹配！這可能導致評估失敗")
                            break
            except Exception as e:
                print(f"    ⚠️ 無法驗證模型維度: {e}")
            
            # 載入模型
            self.final_trainer.load_model(model_path)
            print(f"    ✓ Final Agent 模型已載入: {model_path}")
        
        print(f"\n✅ 評估環境初始化完成\n")

    def evaluate(self, deterministic_seed: bool = True):
        """
        評估 Final Agent
        
        如果 final_trainer 還未初始化，自動初始化評估環境（包含所有 Sub-Agents）
        """
        # 確保評估環境已初始化（包括 Sub-Agents 和 Final Agent）
        if self.final_trainer is None:
            try:
                self.initialize_for_evaluation()
            except Exception as e:
                print(f"\n❌ 評估環境初始化失敗: {e}")
                raise
        
        if self.final_trainer is None:
            raise ValueError("無法初始化評估環境，請確認所有模型文件是否存在")
        
        return self.final_trainer.evaluate(deterministic_seed=deterministic_seed)
    
    def save_all_models(self, base_path: str = './models'):
        """儲存所有模型"""
        os.makedirs(base_path, exist_ok=True)
        
        # Sub-Agent 模型已在訓練時儲存
        print(f"[HierarchicalTrainer] Sub-Agent 模型位置:")
        for name, result in self.sub_agent_results.items():
            if result['status'] == 'success':
                print(f"  - {name}: {result['model_path']}")
        
        # 儲存 Final Agent
        if self.final_trainer is not None:
            final_path = os.path.join(base_path, 'Final_Agent.pth')
            self.final_trainer.save_model(final_path)
            print(f"  - Final Agent: {final_path}")