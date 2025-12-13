import numpy as np
import pandas as pd
import torch
from typing import Dict
from datetime import datetime
import json
import os


class BacktestEngine:
    """
    回測引擎 - 評估強化學習交易策略的表現
    
    功能：
    1. 加載已訓練的模型
    2. 在歷史數據上運行策略
    3. 計算性能指標 (收益率、Sharpe Ratio、最大回撤等)
    4. 生成回測報告和可視化
    """
    
    def __init__(self, initial_balance: float = 100000, transaction_cost: float = 0.001):
        """
        初始化回測引擎
        
        :param initial_balance: 初始資金
        :param transaction_cost: 交易成本（百分比）
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 回測結果
        self.results = {
            'daily_values': [],
            'portfolio_values': [],
            'returns': [],
            'actions': [],
            'timestamps': [],
            'trades': []
        }
    
    def load_model(self, path: str) -> object:
        """
        加載已訓練的模型
        
        :param path: 模型檔案路徑
        :return: 加載的模型實例
        """
        print(f"[BacktestEngine] 加載模型: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型檔案不存在: {path}")
        
        try:
            # 載入檢查點
            checkpoint = torch.load(path, map_location=self.device)
            
            # ← 檢查模型類型（根據檢查點的鍵來判斷）
            if 'q_network' in checkpoint:
                # DDQN 模型
                algo_type = 'ddqn'
                print(f"  ├─ 檢測到演算法: DDQN")
            elif 'actor' in checkpoint and 'critic' in checkpoint:
                # DDPG 或 A2C 模型
                if 'target_actor' in checkpoint:
                    algo_type = 'ddpg'
                    print(f"  ├─ 檢測到演算法: DDPG")
                else:
                    algo_type = 'a2c'
                    print(f"  ├─ 檢測到演算法: A2C")
            else:
                raise ValueError(f"未知的模型格式: {list(checkpoint.keys())}")
            
            # ← 根據檢查點內容推斷模型配置
            # 檢查是否使用注意力
            has_attention = False
            if algo_type == 'ddqn':
                has_attention = any('attention' in k for k in checkpoint.get('q_network', {}).keys())
            else:
                has_attention = any('attention' in k for k in checkpoint.get('actor', {}).keys())
            
            print(f"  ├─ 使用注意力: {has_attention}")
            print(f"  └─ ✓ 模型檢測完成\n")
            
            # ← 返回檢查點和模型類型，讓調用者決定如何處理
            return {
                'checkpoint': checkpoint,
                'algo_type': algo_type,
                'has_attention': has_attention,
            }
        
        except Exception as e:
            print(f"  ❌ 錯誤: {str(e)}")
            raise
    
    def run_backtest(self, env, model, num_episodes: int = 1,
                     deterministic: bool = True) -> dict:
        """
        執行回測
        
        :param env: 交易環境
        :param model: 強化學習模型實例
        :param num_episodes: 回測回合數
        :param deterministic: 是否使用確定性動作選擇
        :return: 回測結果字典
        """
        print(f"[BacktestEngine] 開始回測...")
        print(f"  - 初始資金: ${self.initial_balance:,.2f}")
        print(f"  - 交易成本: {self.transaction_cost * 100:.2f}%")
        print(f"  - 環境: {num_episodes} 回合\n")
        
        results = {
            'episode_returns': [],
            'episode_final_values': [],
            'total_trades': [],
            'winning_trades': [],
            'daily_returns': [],
            'actions_history': [],
        }
        
        for episode in range(num_episodes):
            print(f"[BacktestEngine] 回合 {episode + 1}/{num_episodes}")
            
            # 重置環境
            state = env.reset()
            done = False
            episode_return = 0
            episode_trades = 0
            winning_trades = 0
            
            episode_values = [self.initial_balance]
            episode_actions = []
            
            step = 0
            while not done:
                # 選擇動作
                if deterministic:
                    action = model.select_action_deterministic(state)
                else:
                    action = model.select_action(state, noise_scale=0.0)
                
                # 執行動作
                next_state, reward, done, info = env.step(action)
                
                # 記錄
                episode_return += reward
                episode_values.append(info.get('portfolio_value', self.initial_balance))
                episode_actions.append(action)
                
                # 統計交易
                if 'num_trades' in info:
                    trades = info['num_trades']
                    if trades > episode_trades:
                        episode_trades = trades
                        if reward > 0:
                            winning_trades += 1
                
                state = next_state
                step += 1
                
                if step >= env.max_steps:
                    break
            
            # 記錄結果
            final_value = episode_values[-1]
            total_return = (final_value - self.initial_balance) / self.initial_balance
            
            results['episode_returns'].append(total_return)
            results['episode_final_values'].append(final_value)
            results['total_trades'].append(episode_trades)
            results['winning_trades'].append(winning_trades)
            results['daily_returns'].append(episode_values)
            results['actions_history'].append(episode_actions)
            
            print(f"  ├─ 最終資金: ${final_value:,.2f}")
            print(f"  ├─ 回報率: {total_return * 100:.2f}%")
            print(f"  ├─ 交易次數: {episode_trades}")
            print(f"  └─ 勝率: {winning_trades}/{episode_trades if episode_trades > 0 else 1}\n")
        
        return results
    
    def calculate_metrics(self, results: Dict, risk_free_rate: float = 0.02) -> Dict:
        """
        計算回測性能指標
        
        為什麼多個 Episodes 能提供更好的指標：
        - Sharpe Ratio: 基於多 Episodes 的標準差計算，更可靠
        - Max Drawdown: 考慮整個回測期間的最大損失
        - Win Rate: 基於多 Episodes 的勝率統計
        - 穩定性: 能評估收益的一致性
        """
        print(f"[BacktestEngine] 計算性能指標...\n")
        
        returns = np.array(results['episode_returns'])
        final_values = np.array(results['episode_final_values'])
        sharpes = np.array(results['episode_sharpe'])
        max_dds = np.array(results['episode_max_drawdown'])
        
        # 基本指標
        total_return = np.mean(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        # Sharpe Ratio（多 Episodes 平均）
        sharpe_ratio = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        
        # 最大回撤（多 Episodes 平均）
        max_drawdown = np.mean(max_dds)
        
        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 勝率
        num_winning = np.sum(returns > 0)
        win_rate = num_winning / len(returns)
        
        # 平均交易次數
        avg_trades = np.mean(results['total_trades'])
        avg_winning_trades = np.mean(results['winning_trades'])
        
        # ← 新增：穩定性指標
        # 回報的係數變異數（越小越穩定）
        cv_return = std_return / abs(total_return) if total_return != 0 else float('inf')
        
        metrics = {
            'total_return': float(total_return),
            'return_std': float(std_return),
            'min_return': float(min_return),
            'max_return': float(max_return),
            'return_cv': float(cv_return),  # ← 穩定性
            'annual_return': float(total_return),
            'volatility': float(std_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sharpe_std': float(sharpe_std),  # ← Sharpe 的穩定性
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'avg_trades': float(avg_trades),
            'avg_winning_trades': float(avg_winning_trades),
            'final_value': float(np.mean(final_values)),
            'num_episodes': len(returns),
        }
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """改進的回測報告"""
        print(f"\n{'='*80}")
        print(f"{'回測報告':<40} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        print(f"📊 回測配置")
        print(f"{'─'*80}")
        print(f"  回合數: {metrics['num_episodes']} (為什麼?: 計算可靠的統計指標)")
        
        print(f"\n📈 收益指標")
        print(f"{'─'*80}")
        print(f"  平均回報率:     {metrics['total_return']*100:>10.2f}%")
        print(f"  標準差:        {metrics['return_std']*100:>10.2f}% (波動程度)")
        print(f"  最小回報:       {metrics['min_return']*100:>10.2f}%")
        print(f"  最大回報:       {metrics['max_return']*100:>10.2f}%")
        print(f"  變異係數:       {metrics['return_cv']:>10.2f}x (↓ 越小越穩定)")
        
        print(f"\n🎯 風險調整指標")
        print(f"{'─'*80}")
        print(f"  Sharpe 比率:    {metrics['sharpe_ratio']:>10.2f} (平均)")
        print(f"  Sharpe 穩定性:  {metrics['sharpe_std']:>10.2f} std (↓ 越小越穩定)")
        print(f"  最大回撤:       {metrics['max_drawdown']*100:>10.2f}%")
        print(f"  Calmar 比率:    {metrics['calmar_ratio']:>10.2f}")
        
        print(f"\n🎯 交易指標")
        print(f"{'─'*80}")
        print(f"  勝率:           {metrics['win_rate']*100:>10.2f}%")
        print(f"  平均交易次數:   {metrics['avg_trades']:>10.1f}")
        print(f"  平均勝交易:     {metrics['avg_winning_trades']:>10.1f}")
        
        print(f"\n{'='*80}\n")
    
    def save_report(self, metrics: Dict, results: Dict, output_path: str = './backtest_report.json'):
        """
        保存回測報告
        
        :param metrics: 性能指標
        :param results: 回測結果
        :param output_path: 輸出檔案路徑
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'num_episodes': len(results['episode_returns']),
                'total_trades': int(np.sum(results['total_trades'])),
                'winning_trades': int(np.sum(results['winning_trades'])),
            }
        }
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[BacktestEngine] 報告已保存到: {output_path}\n")
    
    def generate_csv_report(self, results: Dict, output_path: str = './backtest_results.csv'):
        """
        生成 CSV 回測結果
        
        :param results: 回測結果
        :param output_path: 輸出檔案路徑
        """
        df_data = {
            'Episode': np.arange(1, len(results['episode_returns']) + 1),
            'Final Value': results['episode_final_values'],
            'Return %': np.array(results['episode_returns']) * 100,
            'Num Trades': results['total_trades'],
            'Winning Trades': results['winning_trades'],
        }
        
        df = pd.DataFrame(df_data)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"[BacktestEngine] CSV 報告已保存到: {output_path}\n")


class BacktestComparator:
    """
    回測比較器 - 比較多個模型的性能
    """
    
    def __init__(self):
        self.backtest_results = {}
    
    def add_result(self, model_name: str, metrics: Dict):
        """
        添加回測結果
        
        :param model_name: 模型名稱
        :param metrics: 性能指標
        """
        self.backtest_results[model_name] = metrics
    
    def compare(self) -> pd.DataFrame:
        """
        比較多個模型
        
        :return: 比較結果 DataFrame
        """
        df = pd.DataFrame(self.backtest_results).T
        
        # 按 Sharpe Ratio 排序
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    def print_comparison(self):
        """
        打印比較結果
        """
        df = self.compare()
        
        print(f"\n{'='*100}")
        print(f"{'模型性能比較':<50}")
        print(f"{'='*100}\n")
        
        print(df.to_string())
        
        print(f"\n{'='*100}\n")
    
    def save_comparison(self, output_path: str = './model_comparison.csv'):
        """
        保存比較結果
        
        :param output_path: 輸出檔案路徑
        """
        df = self.compare()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path)
        
        print(f"[BacktestComparator] 比較結果已保存到: {output_path}\n")