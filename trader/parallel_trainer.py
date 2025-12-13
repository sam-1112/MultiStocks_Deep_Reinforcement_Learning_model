"""
平行化 Sub-Agent 訓練器
"""

import os
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from trader.factory import AlgorithmFactory
from trader.envs.factory import EnvironmentFactory
from trader.envs.trading_env import TradingEnv
from trader.algos.base_algo import AlgorithmStrategy
from trader.utils.seed import SeedManager, EnvironmentSeeder
from trader.utils.logging import TrainingLogger


def train_single_sub_agent(
    agent_config: Dict,
    result_queue: Queue,
    progress_queue: Queue = None
):
    """
    訓練單個 Sub-Agent（在獨立進程中執行）
    
    Args:
        agent_config: Sub-Agent 配置
        result_queue: 用於回傳訓練結果的 Queue
        progress_queue: 用於回報進度的 Queue
    """
    try:
        agent_name = agent_config['name']
        agent_id = agent_config['agent_id']
        seed = agent_config['seed']
        device = agent_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 設置種子
        SeedManager.set_seed(seed)
        
        # 創建環境
        env = EnvironmentFactory.create_trading_env(agent_config['env_config'])
        
        # 創建演算法（指定設備）
        algo_kwargs = agent_config.get('algo_kwargs', {})
        algo_kwargs['device'] = device
        
        algo = AlgorithmFactory.create(
            agent_config['algorithm'],
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            model_type=agent_config['model_type'],
            **algo_kwargs
        )
        
        # 環境種子管理器
        seeder = EnvironmentSeeder(seed)
        
        # 訓練參數
        max_episodes = agent_config['max_episodes']
        update_frequency = agent_config['update_frequency']
        initial_noise_scale = agent_config.get('initial_noise_scale', 0.3)
        final_noise_scale = agent_config.get('final_noise_scale', 0.01)
        
        episode_rewards = []
        
        for episode in range(max_episodes):
            # 計算探索率
            progress = episode / max_episodes
            noise_scale = (initial_noise_scale - final_noise_scale) * (1 - progress) + final_noise_scale
            
            # Reset 環境
            reset_seed = seeder.get_reset_seed()
            observation, info = env.reset(seed=reset_seed)
            
            done = False
            truncated = False
            total_reward = 0
            step = 0
            
            while not done and not truncated:
                # 選擇動作
                action = algo.select_action(observation, noise_scale=noise_scale)
                
                # 執行動作
                next_observation, reward, done, truncated, info = env.step(action)
                
                # 存儲經驗
                if hasattr(algo, 'store_experience'):
                    algo.store_experience(observation, action, reward, next_observation, done)
                
                # 定期更新模型
                if step % update_frequency == 0:
                    algo.update_model()
                
                observation = next_observation
                total_reward += reward
                step += 1
            
            episode_rewards.append(total_reward)
            
            # 回報進度
            if progress_queue is not None:
                progress_queue.put({
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'episode': episode + 1,
                    'reward': total_reward,
                    'avg_reward': np.mean(episode_rewards[-10:])
                })
        
        # 儲存模型
        model_save_path = agent_config['model_save_path']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        algo.save_model(model_save_path)
        
        # 回傳結果
        result_queue.put({
            'agent_id': agent_id,
            'agent_name': agent_name,
            'status': 'success',
            'model_path': model_save_path,
            'episode_rewards': episode_rewards,
            'final_avg_reward': np.mean(episode_rewards[-10:])
        })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'agent_id': agent_config.get('agent_id', -1),
            'agent_name': agent_config.get('name', 'Unknown'),
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class ParallelSubAgentTrainer:
    """
    平行化 Sub-Agent 訓練器
    
    使用 multiprocessing 同時訓練多個 Sub-Agent
    """
    
    def __init__(self, config: Dict, seed: int = 42):
        """
        初始化平行訓練器
        
        Args:
            config: 完整配置字典
            seed: 基礎隨機種子
        """
        self.config = config
        self.seed = seed
        self.sub_agents_config = config['agent_mode'].get('sub_agents', [])
        self.trained_models: Dict[str, str] = {}  # agent_name -> model_path
        
    def train_sub_agents_parallel(self, num_workers: int = None) -> Dict[str, Dict]:
        """
        平行訓練所有 Sub-Agents
        
        Args:
            num_workers: 最大並行數（預設為 Sub-Agent 數量）
        
        Returns:
            訓練結果字典
        """
        # 使用 spawn context 以支援 CUDA
        ctx = mp.get_context('spawn')
        
        if num_workers is None:
            num_workers = len(self.sub_agents_config)
        
        # 限制最大 worker 數
        num_workers = min(num_workers, mp.cpu_count(), len(self.sub_agents_config))
        
        print(f"\n{'='*70}")
        print(f"[ParallelTrainer] 🚀 開始平行訓練 {len(self.sub_agents_config)} 個 Sub-Agents")
        print(f"  - 並行數: {num_workers}")
        print(f"  - 基礎種子: {self.seed}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU 數量: {torch.cuda.device_count()}")
        print(f"{'='*70}\n")
        
        # 準備每個 Sub-Agent 的配置
        agent_configs = self._prepare_agent_configs()
        
        # 使用 spawn context 創建 Queue
        result_queue = ctx.Queue()
        progress_queue = ctx.Queue()
        
        # 創建並啟動進程
        processes = []
        for config in agent_configs:
            p = ctx.Process(
                target=train_single_sub_agent,
                args=(config, result_queue, progress_queue)
            )
            processes.append(p)
            p.start()
        
        # 監控進度
        self._monitor_progress(
            processes, 
            progress_queue, 
            len(agent_configs),
            agent_configs[0]['max_episodes']
        )
        
        # 等待所有進程完成
        for p in processes:
            p.join()
        
        # 收集結果
        results = {}
        while not result_queue.empty():
            result = result_queue.get()
            agent_name = result['agent_name']
            results[agent_name] = result
            
            if result['status'] == 'success':
                self.trained_models[agent_name] = result['model_path']
                print(f"  ✓ {agent_name}: 訓練完成 (avg reward: {result['final_avg_reward']:.2f})")
            else:
                print(f"  ✗ {agent_name}: 訓練失敗 - {result.get('error', 'Unknown error')}")
                if 'traceback' in result:
                    print(f"    Traceback: {result['traceback'][:500]}")
        
        print(f"\n{'='*70}")
        print(f"[ParallelTrainer] ✅ 所有 Sub-Agents 訓練完成！")
        print(f"{'='*70}\n")
        
        return results
    
    def _prepare_agent_configs(self) -> List[Dict]:
        """準備每個 Sub-Agent 的配置"""
        data_cfg = self.config['data']
        env_cfg = self.config['env']
        train_cfg = self.config['training']
        hyper_cfg = self.config['hyperparameters']
        
        stock_symbols = data_cfg['ticker_list']
        
        # 決定每個 Agent 使用的設備
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        configs = []
        for i, sub_agent in enumerate(self.sub_agents_config):
            agent_seed = self.seed + i
            
            # 分配 GPU（如果有多個 GPU，輪流分配）
            if num_gpus > 0:
                device = f'cuda:{i % num_gpus}'
            else:
                device = 'cpu'
            
            config = {
                'agent_id': i,
                'name': sub_agent.get('name', f'Sub-Agent-{i}'),
                'algorithm': sub_agent.get('algorithm', 'a2c'),
                'model_type': sub_agent.get('model_type', 'mlp'),
                'agent_type': sub_agent.get('agent_type', 'direction'),
                'seed': agent_seed,
                'device': device,
                'max_episodes': train_cfg['max_episodes'],
                'update_frequency': train_cfg['update_frequency'],
                'initial_noise_scale': 0.3,
                'final_noise_scale': 0.01,
                'model_save_path': f"./models/{sub_agent.get('name', f'Sub-Agent-{i}')}_agent.pth",
                'env_config': {
                    'num_stocks': len(stock_symbols),
                    'stock_symbols': stock_symbols,
                    'initial_balance': env_cfg['initial_balance'],
                    'max_steps': env_cfg['max_steps'],
                    'start_date': data_cfg['date_start'],
                    'end_date': data_cfg['date_end'],
                    'transaction_cost': env_cfg['transaction_cost'],
                    'seed': agent_seed,
                    'agent_type': sub_agent.get('agent_type', 'direction'),
                },
                'algo_kwargs': {
                    'actor_lr': float(hyper_cfg['actor_lr']),
                    'critic_lr': float(hyper_cfg['critic_lr']),
                    'gamma': float(hyper_cfg['gamma']),
                    'hidden_dim': int(hyper_cfg['hidden_dim']),
                    'batch_size': int(hyper_cfg['batch_size']),
                    'device': device,
                }
            }
            configs.append(config)
        
        return configs
    
    def _monitor_progress(self, processes: List[Process], progress_queue: Queue,
                         num_agents: int, max_episodes: int):
        """監控訓練進度"""
        from collections import defaultdict
        
        progress = defaultdict(lambda: {'episode': 0, 'reward': 0, 'avg_reward': 0})
        
        # 創建進度條
        pbar = tqdm(
            total=num_agents * max_episodes,
            desc="Training Sub-Agents",
            unit="ep",
            colour='cyan'
        )
        
        completed_episodes = 0
        
        while any(p.is_alive() for p in processes):
            try:
                # 非阻塞讀取
                while not progress_queue.empty():
                    update = progress_queue.get_nowait()
                    agent_name = update['agent_name']
                    progress[agent_name] = update
                    
                    pbar.update(1)
                    completed_episodes += 1
                    
                    # 更新進度條描述
                    status_str = " | ".join([
                        f"{name[:10]}: E{p['episode']}" 
                        for name, p in progress.items()
                    ])
                    pbar.set_postfix_str(status_str[:60])
                    
            except Exception:
                pass
            
            import time
            time.sleep(0.1)
        
        # 清空剩餘的進度更新
        while not progress_queue.empty():
            try:
                progress_queue.get_nowait()
                pbar.update(1)
            except Exception:
                break
        
        pbar.close()


class SubAgentEnsemble:
    """
    Sub-Agent 集成器
    
    載入訓練好的 Sub-Agent 模型，並產生 Q-values 作為 Final Agent 的輸入
    
    功能：
    - 載入多個 Sub-Agent 模型
    - 執行狀態適配（從完整特徵縮小到各 agent 需要的特徵）
    - 生成 Q-values 供 Final Agent 使用
    """
    
    def __init__(self, model_paths: Dict[str, Dict], base_env: TradingEnv = None, device: str = None):
        """
        初始化集成器
        
        Args:
            model_paths: {agent_name: {
                'path': str,                # 模型檔案路徑
                'algorithm': str,           # 演算法名稱
                'state_dim': int,           # Sub-Agent 期望的狀態維度
                'action_dim': int,          # 動作維度
                'agent_type': str,          # Agent 類型（direction/fundamental/risk_regime）
                'model_type': str,          # 模型類型
                'hidden_dim': int           # 隱藏層維度
            }}
            base_env: 基礎 TradingEnv（用於狀態適配和特徵配置）
            device: 計算設備（None 表示自動選擇）
        """
        # 自動選擇設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sub_agents: Dict[str, AlgorithmStrategy] = {}
        self.agent_info = model_paths
        self.base_env = base_env
        
        # 存儲每個 agent 的 feature_config
        self.agent_feature_configs: Dict[str, dict] = {}
        
        # 存儲每個 agent 的狀態維度（用於驗證）
        self.agent_state_dims: Dict[str, int] = {}
        
        # 計算基礎特徵維度（用於狀態適配）
        if base_env is not None:
            self.base_state_features = self._calculate_base_features_dim()
        else:
            self.base_state_features = None
        
        print(f"[SubAgentEnsemble] 使用設備: {self.device}")
        self._load_models(model_paths)
    
    def _calculate_base_features_dim(self) -> Dict[str, int]:
        """
        計算基礎狀態中各類特徵的維度
        
        Returns:
            {
                'stock': int,       # OHLCV 特徵數
                'technical': int,   # 技術指標數
                'fundamental': int, # 基本面數
                'portfolio': int    # 投資組合狀態數（balance + holdings + portfolio_value）
            }
        """
        features_dim = {}
        
        # Stock features (OHLCV)
        if len(self.base_env.stock_data.shape) == 3:
            # (timesteps, num_stocks, features)
            features_dim['stock'] = self.base_env.stock_data.shape[1] * self.base_env.stock_data.shape[2]
        else:
            features_dim['stock'] = self.base_env.stock_data.shape[1]
        
        # Technical indicators
        if len(self.base_env.technical_indicators.shape) == 3:
            features_dim['technical'] = (self.base_env.technical_indicators.shape[1] * 
                                        self.base_env.technical_indicators.shape[2])
        else:
            features_dim['technical'] = self.base_env.technical_indicators.shape[1]
        
        # Fundamental data
        if len(self.base_env.fundamental_data.shape) == 3:
            features_dim['fundamental'] = (self.base_env.fundamental_data.shape[1] * 
                                          self.base_env.fundamental_data.shape[2])
        else:
            features_dim['fundamental'] = self.base_env.fundamental_data.shape[1]
        
        # Portfolio state (balance + holdings + portfolio_value)
        features_dim['portfolio'] = 1 + self.base_env.num_stocks + 1
        
        return features_dim
    
    def _get_agent_feature_config(self, agent_type: str) -> dict:
        """
        根據 agent_type 取得該 agent 應該使用的特徵配置
        
        Args:
            agent_type: 'direction', 'fundamental', 'risk_regime', 'final'
        
        Returns:
            feature_config: {
                'use_stock': bool,
                'use_technical': bool,
                'use_fundamental': bool,
                'use_portfolio': bool
            }
        """
        if agent_type == 'direction':
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': False,
                'use_portfolio': True
            }
        elif agent_type == 'fundamental':
            return {
                'use_stock': True,
                'use_technical': False,
                'use_fundamental': True,
                'use_portfolio': True
            }
        elif agent_type == 'risk_regime':
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': False,
                'use_portfolio': True
            }
        else:  # 'final' or others
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': True,
                'use_portfolio': True
            }
    
    def _get_agent_state(self, full_state: np.ndarray, agent_name: str) -> np.ndarray:
        """
        從完整狀態適配到特定 Sub-Agent 需要的狀態
        
        Args:
            full_state: 完整狀態向量（包含所有特徵）
            agent_name: Agent 名稱
        
        Returns:
            適配後的狀態向量
        """
        if self.base_state_features is None or agent_name not in self.agent_feature_configs:
            # 無法適配，直接返回
            return full_state
        
        feature_config = self.agent_feature_configs[agent_name]
        features_dim = self.base_state_features
        
        # 計算每個特徵的位置
        features_list = []
        offset = 0
        
        # Stock features
        if feature_config['use_stock']:
            stock_dim = features_dim['stock']
            features_list.append(full_state[offset:offset + stock_dim])
            offset += stock_dim
        else:
            offset += features_dim['stock']
        
        # Technical indicators
        if feature_config['use_technical']:
            tech_dim = features_dim['technical']
            features_list.append(full_state[offset:offset + tech_dim])
            offset += tech_dim
        else:
            offset += features_dim['technical']
        
        # Fundamental data
        if feature_config['use_fundamental']:
            fund_dim = features_dim['fundamental']
            features_list.append(full_state[offset:offset + fund_dim])
            offset += fund_dim
        else:
            offset += features_dim['fundamental']
        
        # Portfolio state
        if feature_config['use_portfolio']:
            port_dim = features_dim['portfolio']
            features_list.append(full_state[offset:offset + port_dim])
            # offset += port_dim (不需要再用)
        
        # 拼接適配後的狀態
        if features_list:
            adapted_state = np.concatenate(features_list)
        else:
            # 如果沒有任何特徵被選中，返回空狀態
            adapted_state = np.array([], dtype=np.float32)
        
        return adapted_state.astype(np.float32)
    
    def _load_models(self, model_paths: Dict[str, Dict]):
        """載入所有 Sub-Agent 模型"""
        print(f"\n[SubAgentEnsemble] 載入 {len(model_paths)} 個 Sub-Agent 模型...")
        
        for agent_name, info in model_paths.items():
            try:
                # 獲取 agent_type 並存儲 feature_config
                agent_type = info.get('agent_type', 'direction')
                self.agent_feature_configs[agent_name] = self._get_agent_feature_config(agent_type)
                self.agent_state_dims[agent_name] = info['state_dim']
                
                # 創建演算法時指定設備
                algo = AlgorithmFactory.create(
                    info['algorithm'],
                    state_dim=info['state_dim'],
                    action_dim=info['action_dim'],
                    model_type=info.get('model_type', 'mlp'),
                    hidden_dim=info.get('hidden_dim', 256),
                    device=str(self.device)  # 傳遞設備字串
                )
                
                # 載入模型權重
                algo.load_model(info['path'])
                
                # 確保模型在正確的設備上並設為評估模式
                self._move_algo_to_device(algo)
                
                self.sub_agents[agent_name] = algo
                print(f"  ✓ {agent_name} ({agent_type}): 載入成功 (設備: {self.device}, state_dim: {info['state_dim']})")
            except Exception as e:
                import traceback
                print(f"  ✗ {agent_name}: 載入失敗 - {e}")
                traceback.print_exc()
    
    def _move_algo_to_device(self, algo: AlgorithmStrategy):
        """將演算法的所有模型移動到指定設備並設為評估模式"""
        # 移動 Actor
        if hasattr(algo, 'actor') and algo.actor is not None:
            algo.actor = algo.actor.to(self.device)
            algo.actor.eval()
        
        # 移動 Critic
        if hasattr(algo, 'critic') and algo.critic is not None:
            algo.critic = algo.critic.to(self.device)
            algo.critic.eval()
        
        # 移動 Target Actor
        if hasattr(algo, 'target_actor') and algo.target_actor is not None:
            algo.target_actor = algo.target_actor.to(self.device)
            algo.target_actor.eval()
        
        # 移動 Target Critic
        if hasattr(algo, 'target_critic') and algo.target_critic is not None:
            algo.target_critic = algo.target_critic.to(self.device)
            algo.target_critic.eval()
        
        # 移動 Q Network (for DQN/DDQN)
        if hasattr(algo, 'q_network') and algo.q_network is not None:
            algo.q_network = algo.q_network.to(self.device)
            algo.q_network.eval()
        
        if hasattr(algo, 'target_q_network') and algo.target_q_network is not None:
            algo.target_q_network = algo.target_q_network.to(self.device)
            algo.target_q_network.eval()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        獲取所有 Sub-Agent 的 Q-values（執行狀態適配）
        
        Args:
            state: 完整狀態向量（包含所有特徵）
        
        Returns:
            q_values: (num_sub_agents * output_dim,) 展平的 Q-values
        """
        all_q_values = []
        
        for agent_name, algo in self.sub_agents.items():
            # ★ 關鍵：執行狀態適配
            adapted_state = self._get_agent_state(state, agent_name)
            
            # 驗證適配後的狀態維度
            expected_dim = self.agent_state_dims.get(agent_name, -1)
            if len(adapted_state) != expected_dim and expected_dim > 0:
                print(f"  ⚠ {agent_name}: 狀態維度不匹配 (期望: {expected_dim}, 實際: {len(adapted_state)})")
            
            # 獲取 Q-values
            q_values = self._get_agent_q_values(algo, adapted_state)
            all_q_values.append(q_values)
        
        # 展平所有 Q-values
        if all_q_values:
            return np.concatenate(all_q_values)
        else:
            return np.array([])
    
    def _get_agent_q_values(self, algo: AlgorithmStrategy, state: np.ndarray) -> np.ndarray:
        """
        獲取單個 Agent 的 Q-values
        
        不同演算法有不同的獲取方式
        """
        # 確保 state 是正確的格式
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        
        with torch.no_grad():
            if hasattr(algo, 'q_network') and algo.q_network is not None:
                # DDQN: 直接從 Q-network 獲取
                q_values = algo.q_network(state_tensor).cpu().numpy().flatten()
            elif hasattr(algo, 'actor') and algo.actor is not None:
                # A2C/DDPG: 從 actor 獲取 logits/actions
                logits = algo.actor(state_tensor).cpu().numpy().flatten()
                
                if hasattr(algo, 'critic') and algo.critic is not None:
                    # 嘗試獲取 critic value
                    try:
                        # 對於 A2C，critic 只接受 state
                        value = algo.critic(state_tensor).cpu().numpy().flatten()
                        q_values = np.concatenate([logits, value])
                    except Exception:
                        # 對於 DDPG，critic 需要 state 和 action
                        q_values = logits
                else:
                    q_values = logits
            elif hasattr(algo, 'critic') and algo.critic is not None:
                q_values = algo.critic(state_tensor).cpu().numpy().flatten()
            else:
                # 預設：回傳零向量
                action_dim = getattr(algo, 'action_dim', 10)
                q_values = np.zeros(action_dim)
        
        return q_values
    
    def get_ensemble_actions(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        獲取所有 Sub-Agent 的動作建議（執行狀態適配）
        
        Args:
            state: 完整狀態向量
        
        Returns:
            actions: {agent_name: action_array}
        """
        actions = {}
        for agent_name, algo in self.sub_agents.items():
            try:
                # ★ 執行狀態適配
                adapted_state = self._get_agent_state(state, agent_name)
                
                # 使用確定性動作選擇
                if hasattr(algo, 'select_action_deterministic'):
                    action = algo.select_action_deterministic(adapted_state)
                else:
                    action = algo.select_action(adapted_state, noise_scale=0.0)
                actions[agent_name] = action
            except Exception as e:
                print(f"  ⚠ {agent_name} 選擇動作失敗: {e}")
                actions[agent_name] = np.zeros(algo.action_dim)
        return actions
    
    def get_q_values_dim(self) -> int:
        """
        獲取 Q-values 的總維度
        
        透過實際計算來獲取準確的維度
        """
        # 嘗試用一個假的 state 來計算實際維度
        if not self.sub_agents or self.base_env is None:
            return self._estimate_q_values_dim()
        
        # 獲取基礎環境的完整狀態維度
        full_state_dim = self.base_env.state_dim
        
        # 創建一個假的完整狀態
        dummy_state = np.zeros(full_state_dim)
        
        try:
            # 實際計算 Q-values 維度
            q_values = self.get_q_values(dummy_state)
            actual_dim = len(q_values)
            print(f"[SubAgentEnsemble] 實際 Q-values 維度: {actual_dim}")
            return actual_dim
        except Exception as e:
            print(f"[SubAgentEnsemble] 無法計算實際維度，使用估算: {e}")
            # 回退到估算方法
            return self._estimate_q_values_dim()
    
    def _estimate_q_values_dim(self) -> int:
        """估算 Q-values 維度（備用方法）"""
        total_dim = 0
        for agent_name, algo in self.sub_agents.items():
            action_dim = getattr(algo, 'action_dim', 10)
            
            if hasattr(algo, 'q_network') and algo.q_network is not None:
                # DDQN: Q-network 輸出維度
                total_dim += action_dim * 3  # 3 = buy, hold, sell
            elif hasattr(algo, 'actor') and algo.actor is not None:
                # Actor-Critic 方法
                actor_output_dim = action_dim * 3  # A2C 輸出是 action_dim * 3
                
                if hasattr(algo, 'critic') and algo.critic is not None:
                    critic_output_dim = 1
                    total_dim += actor_output_dim + critic_output_dim
                else:
                    total_dim += actor_output_dim
            else:
                total_dim += action_dim
        
        return total_dim