from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from trader.utils.seed import SeedManager, EnvironmentSeeder

class BaseTradingEnv(gym.Env):
    """
    多股票交易環境 - 支持種子
    
    ========== 狀態空間結構 ==========
    [stock_features | technical_indicators | fundamental_data | balance_ratio | holdings]
    
    ========== 獎勵函數公式 ==========
    Reward = Change_in_Portfolio + Sharpe_ratio + 0.9 * daily_returns - fee_penalty
    
    其中：
    - Change_in_Portfolio = ΔPortfolio / old_portfolio_value
    - Sharpe_ratio = (E[R_a] - R_b) / σ_a
      - E[R_a] = mean(daily_returns)
      - R_b = 日化無風險利率（美國國債利率）
      - σ_a = std(daily_returns)
    - daily_returns = (new_portfolio - old_portfolio) / old_portfolio
    - fee_penalty = transaction_cost / portfolio_value * 0.5
    """
    
    @staticmethod
    def load_bond_yields(csv_path: str = None) -> pd.DataFrame:
        """
        從 CSV 檔案讀取美國國債利率
        
        :param csv_path: CSV 檔案路徑，若為 None 則自動搜尋專案目錄
        :return: 國債利率 DataFrame（索引為日期）
        """
        if csv_path is None:
            # 自動搜尋 data/bonds 目錄
            bond_dir = Path(__file__).parent.parent.parent / "data" / "bonds"
            csv_files = list(bond_dir.glob("bond_yields_*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"找不到國債利率 CSV 檔案，搜尋路徑：{bond_dir}")
            csv_path = csv_files[0]
            print(f"[BaseTradingEnv] 自動發現國債利率檔案：{csv_path}")
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        print(f"[BaseTradingEnv] 已載入國債利率數據，共 {len(df)} 筆記錄")
        print(f"  - 時間範圍：{df.index[0].date()} 至 {df.index[-1].date()}")
        print(f"  - 可用期限：{list(df.columns)}")
        print(f"  - 10Y 利率範圍：{df['10Y'].min():.4f}% 至 {df['10Y'].max():.4f}%\n")
        
        return df
    
    def __init__(self, config: dict):
        """
        :param config: 配置字典
            必需：
            - num_stocks: 股票數量
            - initial_balance: 初始資金
            - max_steps: 最大步數
            - stock_data: 股票數據 (num_steps, num_stocks, 5)
            
            可選：
            - k: 動作範圍 [-k, k]，默認 5
            - transaction_cost: 交易成本，默認 0.001
            - technical_indicators: 技術指標 (num_steps, num_stocks, n_indicators)
            - fundamental_data: 基本面數據 (num_steps, num_stocks, n_fundamentals)
            - stock_dates: 股票數據日期 (num_steps,) pandas DatetimeIndex
            - bond_yields_csv: 國債利率 CSV 檔案路徑
            - bond_yield_column: 使用的國債期限，默認 '10Y'
            - seed: 隨機種子
        """
        self.num_stocks = config['num_stocks']
        self.initial_balance = config['initial_balance']
        self.max_steps = config['max_steps']
        self.k = config.get('k', 5)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.bond_yield_column = config.get('bond_yield_column', '10Y')
        
        # 載入國債利率
        bond_csv_path = config.get('bond_yields_csv', None)
        self.bond_yields = self.load_bond_yields(bond_csv_path)
        
        # 數據
        self.stock_data = config['stock_data']
        assert self.stock_data.ndim == 3, f"stock_data 必須是 3D，got {self.stock_data.ndim}D"
        assert self.stock_data.shape[1] == self.num_stocks, "股票數量不匹配"
        
        self.technical_indicators = config.get('technical_indicators',
                                              np.zeros((self.stock_data.shape[0], self.num_stocks, 1)))
        self.fundamental_data = config.get('fundamental_data',
                                          np.zeros((self.stock_data.shape[0], self.num_stocks, 1)))
        
        # ← 股票數據日期（用於對齐國債利率）
        self.stock_dates = config.get('stock_dates', None)
        
        # 正確計算 state_dim
        n_stock_features = self.stock_data.shape[2]
        n_tech_indicators = self.technical_indicators.shape[2]
        n_fund_features = self.fundamental_data.shape[2]
        
        self.state_dim = (self.num_stocks * n_stock_features +
                         self.num_stocks * n_tech_indicators +
                         self.num_stocks * n_fund_features +
                         1 +  # balance_ratio
                         self.num_stocks)  # holdings for each stock
        
        print(f"[BaseTradingEnv] State dimension calculation:")
        print(f"  - Stock features: {self.num_stocks} × {n_stock_features} = {self.num_stocks * n_stock_features}")
        print(f"  - Technical indicators: {self.num_stocks} × {n_tech_indicators} = {self.num_stocks * n_tech_indicators}")
        print(f"  - Fundamental data: {self.num_stocks} × {n_fund_features} = {self.num_stocks * n_fund_features}")
        print(f"  - Balance ratio: 1")
        print(f"  - Holdings: {self.num_stocks}")
        print(f"  - Total state_dim: {self.state_dim}\n")

        # 動作空間
        self.action_space = spaces.MultiDiscrete(
            [2 * self.k + 1] * self.num_stocks
        )
        self.action_dim = self.num_stocks
        self.action_mode = 'discrete'

        self.action_mask = None
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 種子管理器
        base_seed = config.get('seed', 42)
        self.seeder = EnvironmentSeeder(base_seed)
        self.rng = np.random.Generator(np.random.MT19937(base_seed))
        
        # 初始化狀態
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks, dtype=np.float32)  # ← 改為 float32
        self.portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        
        # ← 用於計算 Sharpe ratio 的變量
        self.daily_returns = []
        self.episode_portfolio_values = [self.initial_balance]

        print(f"[BaseTradingEnv] Validating data shapes...")
        print(f"  - Stock data shape: {self.stock_data.shape}")
        print(f"  - Technical indicators shape: {self.technical_indicators.shape}")
        print(f"  - Fundamental data shape: {self.fundamental_data.shape}")
        
        assert self.stock_data.ndim == 3, f"stock_data 必須是 3D 數組"
        assert self.stock_data.shape[1] == self.num_stocks, f"股票數量不匹配"
        assert self.stock_data.shape[0] > 0, f"時間步數不能為 0"
        
        print(f"[BaseTradingEnv] Data validation passed!\n")
    
    def _get_current_bond_yield(self) -> float:
        """
        獲取當前的國債利率
        
        :return: 年化國債利率（百分比）
        """
        if self.stock_dates is not None:
            # ← 如果有日期信息，使用日期對齐
            if self.current_step < len(self.stock_dates):
                current_date = self.stock_dates[self.current_step]
                
                # 嘗試精確匹配日期
                if current_date in self.bond_yields.index:
                    return self.bond_yields.loc[current_date, self.bond_yield_column]
                
                # 如果沒有精確匹配，使用最近的日期
                idx = self.bond_yields.index.searchsorted(current_date)
                if idx > 0:
                    idx -= 1
                return self.bond_yields.iloc[idx][self.bond_yield_column]
        
        # ← 如果沒有日期信息，使用步數比例
        if len(self.bond_yields) > 0:
            # 按比例映射 current_step 到 bond_yields
            idx = int(self.current_step * len(self.bond_yields) / self.stock_data.shape[0])
            idx = min(idx, len(self.bond_yields) - 1)
            return self.bond_yields.iloc[idx][self.bond_yield_column]
        
        # 默認值
        return 2.0
    
    def _get_current_bond_yield_daily(self) -> float:
        """
        獲取日化國債利率
        
        :return: 日化無風險利率（小數）
        """
        annual_rate = self._get_current_bond_yield() / 100.0
        daily_rate = annual_rate / 252.0
        return daily_rate
    
    def _get_current_prices(self) -> np.ndarray:
        """
        獲取當前股票價格（收盤價，第 0 列）
        
        :return: (num_stocks,) 價格數組
        """
        if self.current_step >= self.stock_data.shape[0]:
            raise IndexError(
                f"current_step {self.current_step} 超出數據範圍 "
                f"[0, {self.stock_data.shape[0]-1}]"
            )
        if self.current_step < 0:
            raise IndexError(f"current_step {self.current_step} 不能為負")
        
        prices = self.stock_data[self.current_step, :, 0].astype(np.float32)
        
        if np.any(np.isnan(prices)):
            print(f"[Warning] 步驟 {self.current_step} 的股票價格中存在 NaN")
            if self.current_step > 0:
                fallback_prices = self.stock_data[self.current_step - 1, :, 0]
                prices = np.where(np.isnan(prices), fallback_prices, prices).astype(np.float32)
            else:
                if self.stock_data.shape[0] > 1:
                    fallback_prices = self.stock_data[1, :, 0]
                    prices = np.where(np.isnan(prices), fallback_prices, prices).astype(np.float32)
                else:
                    prices = np.nan_to_num(prices, nan=1.0)
        
        if np.any(prices <= 0):
            print(f"[Warning] 步驟 {self.current_step} 有非正價格")
            prices = np.abs(prices) + 1e-6
            prices = np.where(prices < 1e-6, 1.0, prices)
        
        return prices.astype(np.float32)
    
    def _build_state(self) -> np.ndarray:
        """
        構建狀態向量
        
        結構：[stock_features | tech_indicators | fund_data | balance_ratio | holdings]
        
        :return: (state_dim,) 狀態向量
        """
        current_prices = self._get_current_prices()
        portfolio_value = self.balance + np.sum(self.holdings * current_prices)
        
        balance_ratio = self.balance / max(self.initial_balance, 1e-6)
        balance_ratio = np.clip(balance_ratio, -10, 10)
        
        state = np.concatenate([
            self.stock_data[self.current_step].flatten(),
            self.technical_indicators[self.current_step].flatten(),
            self.fundamental_data[self.current_step].flatten(),
            [balance_ratio],
            self.holdings / self.k
        ], dtype=np.float32)
        
        if state.shape[0] != self.state_dim:
            raise ValueError(f"狀態維度不匹配：期望 {self.state_dim}，收到 {state.shape[0]}")
        
        if np.any(np.isnan(state)):
            state = np.nan_to_num(state, nan=0.0)
        
        if np.any(np.isinf(state)):
            state = np.clip(state, -1e6, 1e6)
        
        return state.astype(np.float32)
    
    def step(self, action):
        """
        執行一步
        
        獎勵函數公式：
        Reward = Change_in_Portfolio + Sharpe_ratio + 0.9 * daily_returns - fee_penalty
        
        其中：
        - Change_in_Portfolio = ΔPortfolio / old_portfolio_value
        - Sharpe_ratio = (E[R_a] - R_b) / σ_a
          - E[R_a] = mean(daily_returns)
          - R_b = 日化無風險利率
          - σ_a = std(daily_returns)
        - daily_returns = (new_portfolio - old_portfolio) / old_portfolio
        - fee_penalty = transaction_cost / portfolio_value * 0.5
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = np.round(action).astype(int)
        
        # 將索引 [0, 2k] 轉換為交易數量 [-k, k]
        trade_quantities = action - self.k
        trade_quantities = np.clip(trade_quantities, -self.k, self.k)
        
        # 執行交易
        # ← 檢查是否執行了被 mask 的動作（即使被 mask 了，policy 仍可能執行）
        current_prices = self._get_current_prices()
        illegal_action_penalty = 0.0
        
        for i in range(self.num_stocks):
            quantity = trade_quantities[i]
            if quantity > 0:
                cost = quantity * current_prices[i] * (1 + self.transaction_cost)
                
                if cost > self.balance:
                    illegal_action_penalty += 0.05  # 輕度懲罰
                    trade_quantities[i] = 0
        
        transaction_cost_total = self._execute_trades(trade_quantities)
        
        # 計算投資組合變化
        old_portfolio_value = self.portfolio_value
        
        self.portfolio_value = self.balance + np.sum(self.holdings * current_prices)
        
        # ========== 獎勵函數計算 ==========
        
        # 1️⃣ 投資組合變化
        portfolio_change = self.portfolio_value - old_portfolio_value
        portfolio_change_ratio = portfolio_change / max(old_portfolio_value, 1e-6)
        
        # 2️⃣ 當日回報率
        daily_return = (self.portfolio_value - self.portfolio_history[-1]) / max(self.portfolio_history[-1], 1e-6)
        self.daily_returns.append(daily_return)
        
        # 3️⃣ Sharpe Ratio 計算
        # Sharpe_ratio = (E[R_a] - R_b) / σ_a
        if len(self.daily_returns) > 1:
            mean_return = np.mean(self.daily_returns)  # E[R_a]
            std_return = np.std(self.daily_returns)     # σ_a
            risk_free_rate_daily = self._get_current_bond_yield_daily()  # R_b
            
            excess_return = mean_return - risk_free_rate_daily  # E[R_a - R_b]
            
            if std_return > 1e-8:
                sharpe_ratio = excess_return / std_return
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        sharpe_ratio = np.clip(sharpe_ratio, -1.0, 1.0)
        
        # 4️⃣ 手續費懲罰
        if self.portfolio_value > 0:
            fee_penalty = (transaction_cost_total / self.portfolio_value) * 0.5
        else:
            fee_penalty = 0.0
        
        # ========== 最終獎勵公式 ==========
        reward = (portfolio_change_ratio +
                 sharpe_ratio +
                 0.9 * daily_return -
                 fee_penalty - illegal_action_penalty)
        
        reward = np.clip(reward, -1.0, 1.0)
        
        # 更新狀態
        self.current_step += 1
        self.portfolio_history.append(self.portfolio_value)
        self.episode_portfolio_values.append(self.portfolio_value)
        
        observation = self._build_state()
        self.action_mask = self._compute_action_mask()
        
        done = (self.portfolio_value <= self.initial_balance * 0.5 or
                self.current_step >= self.max_steps)
        truncated = False
        
        # ← 完整的 info 返回
        current_bond_yield = self._get_current_bond_yield()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_change': portfolio_change,
            'portfolio_change_ratio': portfolio_change_ratio,
            'daily_return': daily_return,
            'mean_return': np.mean(self.daily_returns) if len(self.daily_returns) > 0 else 0.0,
            'std_return': np.std(self.daily_returns) if len(self.daily_returns) > 0 else 0.0,
            'sharpe_ratio': sharpe_ratio,
            'balance': self.balance,
            'holdings': self.holdings.astype(np.float32).copy(),
            'current_prices': current_prices,
            'transaction_cost': transaction_cost_total,
            'bond_yield_annual': current_bond_yield,
            'bond_yield_daily': self._get_current_bond_yield_daily(),
            'reward': reward,
            'reward_breakdown': {
                'portfolio_change_ratio': portfolio_change_ratio,
                'sharpe_ratio': sharpe_ratio,
                'daily_return_weighted': 0.9 * daily_return,
                'fee_penalty': -fee_penalty,
                'illegal_action_penalty': -illegal_action_penalty
            },
            'done': done
        }
        
        return observation, reward, done, truncated, info
    
    def _execute_trades(self, trade_quantities: np.ndarray) -> float:
        """
        執行交易（資金不足時按比例縮小）
        
        :param trade_quantities: 交易數量 [-k, k]
        :return: 總手續費
        """
        current_prices = self._get_current_prices()
        
        trade_quantities = np.round(trade_quantities).astype(int)
        trade_quantities = np.clip(trade_quantities, -self.k, self.k)
        
        # 第一階段：計算所需資金
        cash_needed = 0.0
        buy_indices = []
        
        for i in range(self.num_stocks):
            if trade_quantities[i] > 0:
                buy_indices.append(i)
                cost = trade_quantities[i] * current_prices[i] * (1 + self.transaction_cost)
                cash_needed += cost
        
        # ← 資金不足時按比例縮小所有買入訂單
        if cash_needed > self.balance and cash_needed > 0:
            scale_factor = self.balance / cash_needed
            for i in buy_indices:
                original_qty = trade_quantities[i]
                trade_quantities[i] = int(np.floor(original_qty * scale_factor))
        
        # 第二階段：執行所有交易
        total_fee = 0.0
        
        for i in range(self.num_stocks):
            quantity = trade_quantities[i]
            if quantity == 0:
                continue
            
            price = current_prices[i]
            
            if quantity > 0:
                # 買入
                cost = quantity * price * (1 + self.transaction_cost)
                fee = quantity * price * self.transaction_cost
                
                if cost <= self.balance:
                    self.balance -= cost
                    self.holdings[i] += quantity
                    total_fee += fee
                else:
                    print(f"[Warning] 步驟 {self.current_step}：股票 {i} 買入失敗（資金不足）")
            
            else:
                # 賣出
                quantity_to_sell = min(abs(quantity), int(self.holdings[i]))
                
                if quantity_to_sell > 0:
                    revenue = quantity_to_sell * price * (1 - self.transaction_cost)
                    fee = quantity_to_sell * price * self.transaction_cost
                    
                    self.balance += revenue
                    self.holdings[i] -= quantity_to_sell
                    total_fee += fee
        
        # ← 驗證不變性
        assert self.balance >= 0, f"負餘額：{self.balance}"
        assert np.all(self.holdings >= 0), f"負持倉：{self.holdings}"
        
        return total_fee
    
    def reset(self, seed: Optional[int] = None, options=None):
        """重置環境"""
        if seed is not None:
            super().reset(seed=seed)
            actual_seed = seed
        else:
            actual_seed = self.seeder.get_reset_seed()
            super().reset(seed=actual_seed)
        
        self.rng = np.random.Generator(np.random.MT19937(actual_seed))
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        
        # ← 重置 Sharpe ratio 相關變量
        self.daily_returns = []
        self.episode_portfolio_values = [self.initial_balance]
        
        observation = self._build_state()
        
        # ← 計算初始 action mask
        self.action_mask = self._compute_action_mask()
        
        info = {
            'seed': actual_seed,
            'episode': self.seeder.get_episode_count(),
            'action_mask': self.action_mask  # ← 返回初始 mask
        }
        
        return observation, info

    def _compute_action_mask(self) -> np.ndarray:
        """
        計算有效的動作 mask
        
        :return: (num_stocks * (2*k+1),) 布林陣列，True 表示有效動作
        """
        current_prices = self._get_current_prices()
        mask = np.ones(self.action_space.nvec.sum(), dtype=bool)
        
        for i in range(self.num_stocks):
            price = current_prices[i]
            
            for action_idx in range(2 * self.k + 1):
                quantity = action_idx - self.k
                
                if quantity > 0:
                    cost = quantity * price * (1 + self.transaction_cost)
                    
                    if cost > self.balance:
                        stock_action_start = i * (2 * self.k + 1)
                        mask[stock_action_start + action_idx] = False
        
        return mask