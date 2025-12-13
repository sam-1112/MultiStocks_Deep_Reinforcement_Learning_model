import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class TrainingLogger:
    """
    訓練日誌記錄器
    
    記錄訓練過程中的各種指標，並繪製圖表
    """
    
    def __init__(self, 
                 agent_name: str,
                 algorithm: str,
                 agent_mode: str = 'single-agent',
                 log_dir: str = './logs',
                 save_frequency: int = 10):
        """
        初始化日誌記錄器
        
        :param agent_name: Agent 名稱
        :param algorithm: 演算法名稱
        :param agent_mode: Agent 模式 (single-agent / multi-agent)
        :param log_dir: 日誌目錄
        :param save_frequency: 儲存頻率（每 N 個 episode）
        """
        self.agent_name = agent_name
        self.algorithm = algorithm.upper()
        self.agent_mode = agent_mode
        self.save_frequency = save_frequency
        
        # 創建帶時間戳的日誌目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{algorithm}_{agent_mode}_{timestamp}"
        self.log_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'plots'), exist_ok=True)
        
        # ==================== 訓練指標 ====================
        self.training_metrics = {
            # 每 Episode 指標
            'episode': [],
            'episode_reward': [],
            'episode_length': [],
            'cumulative_reward': [],
            
            # 損失函數
            'actor_loss': [],
            'critic_loss': [],
            
            # 探索率
            'noise_scale': [],
            
            # 時間戳
            'timesteps': [],
            'wall_time': [],
        }
        
        # ==================== 交易績效指標 ====================
        self.trading_metrics = {
            'episode': [],
            'initial_portfolio': [],
            'final_portfolio': [],
            'total_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'volatility': [],
            'win_rate': [],
            'num_trades': [],
        }
        
        # ==================== 動作分佈統計 ====================
        self.action_metrics = {
            'episode': [],
            'buy_count': [],
            'hold_count': [],
            'sell_count': [],
            'buy_ratio': [],
            'hold_ratio': [],
            'sell_ratio': [],
        }
        
        # ==================== 每步詳細記錄（可選） ====================
        self.step_metrics = {
            'episode': [],
            'step': [],
            'reward': [],
            'portfolio_value': [],
            'balance': [],
            'action': [],  # 存儲動作陣列
        }
        
        # 內部追蹤變數
        self._episode_start_time = None
        self._current_episode = 0
        self._total_timesteps = 0
        self._cumulative_reward = 0.0
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_portfolio_values = []
        
        print(f"[TrainingLogger] 初始化完成")
        print(f"  - 日誌目錄: {self.log_dir}")
        print(f"  - Agent: {agent_name}")
        print(f"  - 演算法: {self.algorithm}\n")
    
    # ==================== Episode 級別記錄 ====================
    
    def start_episode(self, episode: int):
        """開始新的 Episode"""
        self._episode_start_time = datetime.now()
        self._current_episode = episode
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_portfolio_values = []
    
    def log_step(self, 
                 step: int,
                 action: np.ndarray,
                 reward: float,
                 portfolio_value: float,
                 balance: float,
                 info: Dict = None):
        """
        記錄單步資訊
        
        :param step: 當前步數
        :param action: 動作陣列 {-1, 0, 1}
        :param reward: 獎勵
        :param portfolio_value: 資產組合價值
        :param balance: 現金餘額
        :param info: 額外資訊
        """
        self._episode_actions.append(action.copy())
        self._episode_rewards.append(reward)
        self._episode_portfolio_values.append(portfolio_value)
        self._total_timesteps += 1
    
    def end_episode(self,
                episode_reward: float,
                episode_length: int,
                actor_loss: float,
                critic_loss: float,
                noise_scale: float,
                initial_portfolio: float,
                final_portfolio: float,
                info: Dict = None):
        """
        結束 Episode 並記錄彙總指標
        """
        self._cumulative_reward += episode_reward
        wall_time = (datetime.now() - self._episode_start_time).total_seconds()
        
        # 訓練指標
        self.training_metrics['episode'].append(self._current_episode)
        self.training_metrics['episode_reward'].append(episode_reward)
        self.training_metrics['episode_length'].append(episode_length)
        self.training_metrics['cumulative_reward'].append(self._cumulative_reward)
        self.training_metrics['actor_loss'].append(actor_loss)
        self.training_metrics['critic_loss'].append(critic_loss)
        self.training_metrics['noise_scale'].append(noise_scale)
        self.training_metrics['timesteps'].append(self._total_timesteps)
        self.training_metrics['wall_time'].append(wall_time)
        
        # ==================== 計算交易績效 ====================
        
        # 1. 總報酬率（百分比）
        if initial_portfolio > 0:
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
        else:
            total_return = 0.0
        
        # 2. Sharpe Ratio（年化）
        sharpe_ratio = 0.0
        volatility = 0.0
        
        if len(self._episode_portfolio_values) > 1:
            # 計算日收益率
            portfolio_values = np.array(self._episode_portfolio_values, dtype=np.float64)
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]  # 不做 * 100
            
            daily_mean = np.mean(daily_returns)
            daily_std = np.std(daily_returns)
            
            # ★ 修復：Sharpe Ratio 正確計算
            # Sharpe = (avg_return - risk_free_rate) / std_return * sqrt(252)
            risk_free_daily = 0.02 / 252  # 年化 2% 轉換為日率
            
            if daily_std > 1e-8:  # 避免除以 0
                sharpe_ratio = (daily_mean - risk_free_daily) / daily_std * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # ★ 修復：年化波動率（正確轉換）
            volatility = daily_std * np.sqrt(252) * 100  # 最後才 * 100
        
        # 3. 最大回撤（相對最高點的下跌）
        max_drawdown = 0.0
        if len(self._episode_portfolio_values) > 0:
            portfolio_values = np.array(self._episode_portfolio_values, dtype=np.float64)
            peak = np.maximum.accumulate(portfolio_values)
            
            # ★ 修復：避免除以 0（peak 可能為 0）
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = (peak - portfolio_values) / peak
                drawdown = np.where(np.isfinite(drawdown), drawdown, 0)  # NaN 轉為 0
            
            max_drawdown = np.max(drawdown) * 100
        
        # 4. 勝率（獎勵 > 0 的步數比例）
        win_rate = 0.0
        if len(self._episode_rewards) > 0:
            positive_rewards = np.sum(np.array(self._episode_rewards) > 0)
            win_rate = positive_rewards / len(self._episode_rewards) * 100
        
        # 5. 交易次數（多股票時要考慮維度）
        num_trades = 0
        if len(self._episode_actions) > 0:
            all_actions = np.array(self._episode_actions)
            # 計算所有非持有動作的總數
            num_trades = np.sum(all_actions != 0)
        
        # ==================== 記錄交易績效 ====================
        
        self.trading_metrics['episode'].append(self._current_episode)
        self.trading_metrics['initial_portfolio'].append(initial_portfolio)
        self.trading_metrics['final_portfolio'].append(final_portfolio)
        self.trading_metrics['total_return'].append(total_return)
        self.trading_metrics['sharpe_ratio'].append(sharpe_ratio)
        self.trading_metrics['max_drawdown'].append(max_drawdown)
        self.trading_metrics['volatility'].append(volatility)
        self.trading_metrics['win_rate'].append(win_rate)
        self.trading_metrics['num_trades'].append(num_trades)
        
        # ==================== 動作分佈統計 ====================
        
        if len(self._episode_actions) > 0:
            all_actions = np.array(self._episode_actions).flatten()
            total_actions = len(all_actions)
            
            # ★ 檢測並轉換動作值
            if np.all((all_actions >= 0) & (all_actions <= 2)):
                all_actions = all_actions - 1  # 轉換 [0,1,2] → [-1,0,1]
            
            # 計算各動作計數
            buy_count = np.sum(all_actions == 1)
            hold_count = np.sum(all_actions == 0)
            sell_count = np.sum(all_actions == -1)
            counted_total = int(buy_count + hold_count + sell_count)
            
            self.action_metrics['episode'].append(self._current_episode)
            self.action_metrics['buy_count'].append(int(buy_count))
            self.action_metrics['hold_count'].append(int(hold_count))
            self.action_metrics['sell_count'].append(int(sell_count))
            
            # 計算比例
            if counted_total > 0:
                buy_ratio = buy_count / counted_total * 100
                hold_ratio = hold_count / counted_total * 100
                sell_ratio = sell_count / counted_total * 100
                
                self.action_metrics['buy_ratio'].append(buy_ratio)
                self.action_metrics['hold_ratio'].append(hold_ratio)
                self.action_metrics['sell_ratio'].append(sell_ratio)
                
                # 調試檢查
                if counted_total != total_actions:
                    print(f"[TrainingLogger] ⚠️ Episode {self._current_episode}: "
                        f"動作計數不符 (計數: {counted_total}, 實際: {total_actions})")
        
        # 定期儲存
        if self._current_episode % self.save_frequency == 0:
            self.save_metrics()
    
    # ==================== 儲存與載入 ====================
    
    def save_metrics(self):
        """儲存所有指標到 CSV 和 JSON"""
        # 儲存訓練指標
        training_df = pd.DataFrame(self.training_metrics)
        training_df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)
        
        # 儲存交易績效
        trading_df = pd.DataFrame(self.trading_metrics)
        trading_df.to_csv(os.path.join(self.log_dir, 'trading_metrics.csv'), index=False)
        
        # 儲存動作分佈
        action_df = pd.DataFrame(self.action_metrics)
        action_df.to_csv(os.path.join(self.log_dir, 'action_metrics.csv'), index=False)
        
        # 儲存配置資訊
        config = {
            'agent_name': self.agent_name,
            'algorithm': self.algorithm,
            'agent_mode': self.agent_mode,
            'total_episodes': self._current_episode,
            'total_timesteps': self._total_timesteps,
            'final_cumulative_reward': self._cumulative_reward,
        }
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # ==================== 繪圖功能 ====================
    
    def plot_training_curves(self):
        """繪製訓練曲線"""
        if len(self.training_metrics['episode']) == 0:
            print("[TrainingLogger] 沒有足夠的數據繪製圖表")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.algorithm} Training Curves - {self.agent_name}', fontsize=14)
        
        episodes = self.training_metrics['episode']
        
        # 1. Episode Reward
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.training_metrics['episode_reward'], alpha=0.6, label='Episode Reward')
        # 移動平均
        if len(episodes) >= 10:
            ma = pd.Series(self.training_metrics['episode_reward']).rolling(10).mean()
            ax1.plot(episodes, ma, linewidth=2, label='MA(10)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Reward
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.training_metrics['cumulative_reward'], color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Reward')
        ax2.grid(True, alpha=0.3)
        
        # 3. Actor & Critic Loss
        ax3 = axes[0, 2]
        ax3.plot(episodes, self.training_metrics['actor_loss'], label='Actor Loss', alpha=0.7)
        ax3.plot(episodes, self.training_metrics['critic_loss'], label='Critic Loss', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Actor & Critic Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Total Return
        ax4 = axes[1, 0]
        returns = self.trading_metrics['total_return']
        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax4.bar(self.trading_metrics['episode'], returns, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Return (%)')
        ax4.set_title('Total Return per Episode')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio
        ax5 = axes[1, 1]
        ax5.plot(self.trading_metrics['episode'], self.trading_metrics['sharpe_ratio'], color='purple')
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.set_title('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        # 6. Exploration Rate
        ax6 = axes[1, 2]
        ax6.plot(episodes, self.training_metrics['noise_scale'], color='orange')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Noise Scale / Epsilon')
        ax6.set_title('Exploration Rate Decay')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'plots', 'training_curves.png'), dpi=150)
        plt.close()
        print(f"[TrainingLogger] 訓練曲線已儲存")
    
    def plot_action_distribution(self):
        """繪製動作分佈圖"""
        if len(self.action_metrics['episode']) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{self.algorithm} Action Distribution - {self.agent_name}', fontsize=14)
        
        episodes = self.action_metrics['episode']
        
        # 1. 堆疊面積圖
        ax1 = axes[0]
        ax1.stackplot(episodes,
                      self.action_metrics['sell_ratio'],
                      self.action_metrics['hold_ratio'],
                      self.action_metrics['buy_ratio'],
                      labels=['Sell (-1)', 'Hold (0)', 'Buy (+1)'],
                      colors=['red', 'gray', 'green'],
                      alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Action Distribution Over Episodes')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # 2. 最終分佈圓餅圖
        ax2 = axes[1]
        final_buy = np.mean(self.action_metrics['buy_ratio'][-10:]) if len(self.action_metrics['buy_ratio']) >= 10 else self.action_metrics['buy_ratio'][-1]
        final_hold = np.mean(self.action_metrics['hold_ratio'][-10:]) if len(self.action_metrics['hold_ratio']) >= 10 else self.action_metrics['hold_ratio'][-1]
        final_sell = np.mean(self.action_metrics['sell_ratio'][-10:]) if len(self.action_metrics['sell_ratio']) >= 10 else self.action_metrics['sell_ratio'][-1]
        
        sizes = [final_sell, final_hold, final_buy]
        labels = [f'Sell\n{final_sell:.1f}%', f'Hold\n{final_hold:.1f}%', f'Buy\n{final_buy:.1f}%']
        colors = ['#ff6b6b', '#868e96', '#51cf66']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax2.set_title('Average Action Distribution (Last 10 Episodes)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'plots', 'action_distribution.png'), dpi=150)
        plt.close()
        print(f"[TrainingLogger] 動作分佈圖已儲存")
    
    def plot_portfolio_performance(self):
        """繪製投資組合績效圖"""
        if len(self.trading_metrics['episode']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.algorithm} Portfolio Performance - {self.agent_name}', fontsize=14)
        
        episodes = self.trading_metrics['episode']
        
        # 1. Portfolio Value
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.trading_metrics['final_portfolio'], label='Final Portfolio', color='blue')
        ax1.axhline(y=self.trading_metrics['initial_portfolio'][0], color='red', linestyle='--', label='Initial')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Max Drawdown
        ax2 = axes[0, 1]
        ax2.fill_between(episodes, 0, self.trading_metrics['max_drawdown'], color='red', alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Max Drawdown (%)')
        ax2.set_title('Maximum Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate
        ax3 = axes[1, 0]
        ax3.plot(episodes, self.trading_metrics['win_rate'], color='green')
        ax3.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # 4. Number of Trades
        ax4 = axes[1, 1]
        ax4.bar(episodes, self.trading_metrics['num_trades'], color='purple', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of Trades')
        ax4.set_title('Trading Activity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'plots', 'portfolio_performance.png'), dpi=150)
        plt.close()
        print(f"[TrainingLogger] 投資組合績效圖已儲存")
    
    def plot_all(self):
        """繪製所有圖表"""
        self.plot_training_curves()
        self.plot_action_distribution()
        self.plot_portfolio_performance()
        print(f"[TrainingLogger] 所有圖表已儲存至 {self.log_dir}/plots/")
    
    # ==================== 摘要報告 ====================
    
    def print_summary(self):
        """打印訓練摘要"""
        print(f"\n{'='*70}")
        print(f"📊 訓練摘要 - {self.algorithm} ({self.agent_name})")
        print(f"{'='*70}")
        
        if len(self.training_metrics['episode']) == 0:
            print("  沒有訓練數據")
            return
        
        # 訓練統計
        print(f"\n📈 訓練統計:")
        print(f"  - 總 Episodes: {self._current_episode}")
        print(f"  - 總 Timesteps: {self._total_timesteps}")
        print(f"  - 累積獎勵: {self._cumulative_reward:.4f}")
        print(f"  - 平均 Episode 獎勵: {np.mean(self.training_metrics['episode_reward']):.4f}")
        print(f"  - 最高 Episode 獎勵: {np.max(self.training_metrics['episode_reward']):.4f}")
        print(f"  - 最低 Episode 獎勵: {np.min(self.training_metrics['episode_reward']):.4f}")
        
        # 交易績效
        print(f"\n💰 交易績效 (最後 10 Episodes 平均):")
        last_n = min(10, len(self.trading_metrics['total_return']))
        print(f"  - 平均報酬率: {np.mean(self.trading_metrics['total_return'][-last_n:]):.2f}%")
        print(f"  - 平均 Sharpe Ratio: {np.mean(self.trading_metrics['sharpe_ratio'][-last_n:]):.4f}")
        print(f"  - 平均最大回撤: {np.mean(self.trading_metrics['max_drawdown'][-last_n:]):.2f}%")
        print(f"  - 平均勝率: {np.mean(self.trading_metrics['win_rate'][-last_n:]):.2f}%")
        
        # 動作分佈
        if len(self.action_metrics['buy_ratio']) > 0:
            print(f"\n🎯 動作分佈 (最後 10 Episodes 平均):")
            last_n = min(10, len(self.action_metrics['buy_ratio']))
            print(f"  - Buy:  {np.mean(self.action_metrics['buy_ratio'][-last_n:]):.1f}%")
            print(f"  - Hold: {np.mean(self.action_metrics['hold_ratio'][-last_n:]):.1f}%")
            print(f"  - Sell: {np.mean(self.action_metrics['sell_ratio'][-last_n:]):.1f}%")
        
        print(f"\n📁 日誌目錄: {self.log_dir}")
        print(f"{'='*70}\n")
    
    def finalize(self):
        """完成訓練，儲存所有數據和圖表"""
        self.save_metrics()
        self.plot_all()
        self.print_summary()