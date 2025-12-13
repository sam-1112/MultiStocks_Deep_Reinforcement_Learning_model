"""
多股票交易環境實現
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from trader.envs.rewards import (
    DirectionAgentReward,
    FundamentalScoreAgentReward,
    RiskRegimeAgentReward,
    FinalAgentReward,
    get_reward_function
)


class TradingEnv(gym.Env):
    """
    多股票交易環境
    
    State: [股票交易數據, 技術指標, 基本面數據, 剩餘資金, 持股張數, 組合價值]
    動作空間: 每支股票的動作為 {-1, 0, 1}
      - -1: 賣出固定數量
      -  0: 持有
      -  1: 買入固定數量
    Reward: 根據 agent_type 使用不同的獎勵函數
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.num_stocks = config.get('num_stocks', 30)
        self.initial_balance = config.get('initial_balance', 1000000.0)
        self.max_steps = config.get('max_steps', 252)
        self.trade_unit = config.get('trade_unit', 100)  # 固定買賣單位
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.seed_value = config.get('seed', None)
        
        # ← 資金分配策略
        self.allocation_strategy = config.get('allocation_strategy', 'priority')  
        # 可選: 'priority', 'equal', 'proportional', 'random'
        
        # ← 每支股票最大資金比例（用於 'equal' 策略）
        self.max_position_ratio = config.get('max_position_ratio', 0.1)  # 每支股票最多佔 10% 資金
        
        # ========== 新增：Reward 函數配置 ==========
        self.agent_type = config.get('agent_type', 'final')  # 預設使用 final agent
        # ★ 新增：根據 agent_type 決定使用哪些特徵
        self.feature_config = self._get_feature_config()
        # reward 函數可能需要的額外參數
        reward_kwargs = {
            'risk_free_rate': config.get('risk_free_rate', 0.02),
            'lookback_period': config.get('lookback_period', 20),
            'reward_weights': config.get('reward_weights', None),
        }
        
        # 初始化 reward 函數
        self.reward_function = get_reward_function(self.agent_type, **reward_kwargs)
        
        # 歷史數據（用於某些 reward 計算）
        self.price_history = []
        self.return_history = []
        # ========================================
        # ★ 新增：模型類型和時序配置
        self.model_type = config.get('model_type', 'mlp')  # mlp, lstm, timesnet
        self.window_size = config.get('window_size', 10)
        
        # 根據模型類型決定是否使用時序
        self.use_temporal = self._should_use_temporal()
        
        # 歷史觀察緩存
        self.observation_history = []

        # 股票數據
        self.stock_data = config.get('stock_data', np.random.randn(1000, self.num_stocks))
        self.technical_indicators = config.get('technical_indicators', np.random.randn(1000, self.num_stocks, 5))
        self.fundamental_data = config.get('fundamental_data', np.random.randn(1000, self.num_stocks, 11))
        
        # 動作空間: 每支股票 {-1, 0, 1}
        self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)
        
        # 狀態空間
        self._calculate_state_dim()
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(self.state_dim,),
        #     dtype=np.float32
        # )
        
        # 內部狀態
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        
        # 交易統計
        self.total_trades = 0
        self.successful_buys = 0
        self.failed_buys = 0  # 因資金不足失敗的買入
        
        # 為了兼容舊代碼
        self.k = 1
        self.action_dim = self.num_stocks

        

    def _get_feature_config(self) -> dict:
        """
        根據 agent_type 決定使用哪些特徵
        
        Returns:
            feature_config: {
                'use_stock': bool,      # OHLCV
                'use_technical': bool,  # 技術指標
                'use_fundamental': bool, # 基本面
                'use_portfolio': bool   # 投資組合狀態（餘額、持倉、總值）
            }
        """
        if self.agent_type == 'direction':
            # Direction Agent: OHLCV + Technical
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': True,
                'use_portfolio': True  # 仍需要知道投資組合狀態
            }
        elif self.agent_type == 'fundamental':
            # Fundamental Agent: OHLCV + Fundamental
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': True,
                'use_portfolio': True
            }
        elif self.agent_type == 'risk_regime':
            # Risk_Regime Agent: OHLCV + Technical
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': True,
                'use_portfolio': True
            }
        else:  # 'final' or others
            # Final Agent: 使用所有特徵
            return {
                'use_stock': True,
                'use_technical': True,
                'use_fundamental': True,
                'use_portfolio': True
            }
    def _should_use_temporal(self) -> bool:
        """
        根據模型類型決定是否使用時序輸入
        
        - MLP: 平面輸入 (feature_dim,)
        - LSTM: 可選時序 (window_size, feature_dim) 或平面
        - TimesNet: 必須時序 (window_size, feature_dim)
        """
        if self.model_type == 'timesnet':
            return True
        elif self.model_type == 'lstm':
            # LSTM 可選，從配置讀取
            return self.window_size > 1
        else:  # mlp
            return False
    
    def _get_current_prices(self) -> np.ndarray:
        """
        獲取當前收盤價
        
        Returns:
            prices: 形狀為 (num_stocks,) 的收盤價數組
        """
        if self.current_step < len(self.stock_data):
            price_data = self.stock_data[self.current_step]
        else:
            price_data = self.stock_data[-1]
        
        # stock_data 形狀可能是 (num_stocks,) 或 (num_stocks, 5)
        if price_data.ndim == 1:
            return price_data.astype(np.float32)
        else:
            # 假設第一列是收盤價 (Close)
            # 根據 dataloader 的順序: [Close, High, Low, Open, Volume]
            return price_data[:, 0].astype(np.float32)

    def _get_portfolio_value(self) -> float:
        """
        計算當前投資組合總價值
        
        Returns:
            portfolio_value: 現金餘額 + 所有持股市值
        """
        current_prices = self._get_current_prices()
        
        # 計算持股市值
        holdings_value = np.sum(self.holdings * current_prices)
        
        # 總價值 = 現金 + 持股市值
        return float(self.balance + holdings_value)

    def _calculate_state_dim(self):
        """計算狀態維度（根據 feature_config）"""
        total_features = 0
        feature_breakdown = {}
        
        # 股票數據：OHLCV (num_stocks, 5)
        if self.feature_config['use_stock']:
            if len(self.stock_data.shape) == 3:
                stock_features = self.stock_data.shape[1] * self.stock_data.shape[2]
            elif len(self.stock_data.shape) == 2:
                stock_features = self.stock_data.shape[1]
            else:
                stock_features = self.num_stocks
            total_features += stock_features
            feature_breakdown['stock'] = stock_features
        else:
            feature_breakdown['stock'] = 0
        
        # 技術指標
        if self.feature_config['use_technical']:
            if len(self.technical_indicators.shape) == 3:
                tech_features = self.technical_indicators.shape[1] * self.technical_indicators.shape[2]
            elif len(self.technical_indicators.shape) == 2:
                tech_features = self.technical_indicators.shape[1]
            else:
                tech_features = 0
            total_features += tech_features
            feature_breakdown['technical'] = tech_features
        else:
            feature_breakdown['technical'] = 0
        
        # 基本面數據
        if self.feature_config['use_fundamental']:
            if len(self.fundamental_data.shape) == 3:
                fund_features = self.fundamental_data.shape[1] * self.fundamental_data.shape[2]
            elif len(self.fundamental_data.shape) == 2:
                fund_features = self.fundamental_data.shape[1]
            else:
                fund_features = 0
            total_features += fund_features
            feature_breakdown['fundamental'] = fund_features
        else:
            feature_breakdown['fundamental'] = 0
        
        # 投資組合狀態
        if self.feature_config['use_portfolio']:
            portfolio_features = 1 + self.num_stocks + 1  # balance + holdings + portfolio_value
            total_features += portfolio_features
            feature_breakdown['portfolio'] = portfolio_features
        else:
            feature_breakdown['portfolio'] = 0
        
        # 設置狀態維度
        self.base_feature_dim = total_features
        self.state_dim = self.base_feature_dim
        
        # 設置觀察空間
        if self.use_temporal:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, self.base_feature_dim),
                dtype=np.float32
            )
            print(f"[TradingEnv] 時序模式 ({self.model_type.upper()}) - Agent: {self.agent_type.upper()}")
            print(f"  - Window size: {self.window_size}")
            print(f"  - Feature dim: {self.base_feature_dim}")
            print(f"  - Observation shape: {(self.window_size, self.base_feature_dim)}")
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            )
            print(f"[TradingEnv] 平面模式 ({self.model_type.upper()}) - Agent: {self.agent_type.upper()}")
            print(f"  - State dim: {self.state_dim}")
        
        print(f"  - Feature breakdown:")
        print(f"    • Stock (OHLCV): {feature_breakdown['stock']}")
        print(f"    • Technical: {feature_breakdown['technical']}")
        print(f"    • Fundamental: {feature_breakdown['fundamental']}")
        print(f"    • Portfolio: {feature_breakdown['portfolio']}")
        
    def _get_observation(self) -> np.ndarray:
        """
        獲取當前觀察狀態（根據 feature_config 選擇性拼接）
        
        Returns:
            - 平面模式: (feature_dim,)
            - 時序模式: (window_size, feature_dim)
        """
        # 獲取當前時間步的數據
        stock_obs = self.stock_data[self.current_step]
        tech_obs = self.technical_indicators[self.current_step]
        fund_obs = self.fundamental_data[self.current_step]
        
        # 準備要拼接的特徵列表
        features = []
        
        # 根據 feature_config 選擇性添加特徵
        if self.feature_config['use_stock']:
            stock_flat = stock_obs.flatten().astype(np.float32)
            features.append(stock_flat)
        
        if self.feature_config['use_technical']:
            tech_flat = tech_obs.flatten().astype(np.float32)
            features.append(tech_flat)
        
        if self.feature_config['use_fundamental']:
            fund_flat = fund_obs.flatten().astype(np.float32)
            features.append(fund_flat)
        
        if self.feature_config['use_portfolio']:
            balance = np.array([self.balance], dtype=np.float32)
            holdings = self.holdings.flatten().astype(np.float32)
            portfolio_value = np.array([self._get_portfolio_value()], dtype=np.float32)
            features.extend([balance, holdings, portfolio_value])
        
        # 拼接所有特徵
        current_obs = np.concatenate(features).astype(np.float32)
        
        # 根據模型類型返回不同格式
        if self.use_temporal:
            # 時序模式：維護滑動窗口
            self.observation_history.append(current_obs)
            
            if len(self.observation_history) > self.window_size:
                self.observation_history.pop(0)
            
            # 冷啟動填充
            while len(self.observation_history) < self.window_size:
                self.observation_history.insert(0, np.zeros_like(current_obs))
            
            # 堆疊為 (window_size, feature_dim)
            temporal_obs = np.stack(self.observation_history, axis=0)
            return temporal_obs
        else:
            # 平面模式
            return current_obs
    
    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """將動作從 [0, 1, 2] 轉換為 [-1, 0, 1]"""
        return action - 1
    
    def _execute_trades_priority(self, action: np.ndarray, prices: np.ndarray) -> dict:
        """
        優先級策略：先執行所有賣出，再按價格排序執行買入
        
        優點：賣出釋放的資金可用於買入，低價股優先買入
        """
        total_transaction_cost = 0.0
        trades_executed = {'buy': 0, 'sell': 0, 'hold': 0, 'failed_buy': 0}
        
        # ========== 第一階段：執行所有賣出 ==========
        sell_indices = np.where(action == -1)[0]
        for i in sell_indices:
            price = prices[i]
            # 跳過無效價格
            if price <= 0 or np.isnan(price) or np.isinf(price):
                continue
                
            sell_quantity = min(self.trade_unit, int(self.holdings[i]))
            
            if sell_quantity > 0:
                revenue = sell_quantity * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.holdings[i] -= sell_quantity
                total_transaction_cost += sell_quantity * price * self.transaction_cost
                trades_executed['sell'] += 1
                self.total_trades += 1
        
        # ========== 第二階段：按價格排序執行買入 ==========
        buy_indices = np.where(action == 1)[0]
        
        if len(buy_indices) > 0:
            # 過濾掉無效價格的股票
            valid_buy_indices = []
            for i in buy_indices:
                price = prices[i]
                if price > 0 and not np.isnan(price) and not np.isinf(price):
                    valid_buy_indices.append(i)
            
            if len(valid_buy_indices) > 0:
                valid_buy_indices = np.array(valid_buy_indices)
                # 按價格從低到高排序（優先買入便宜的股票）
                buy_prices = prices[valid_buy_indices]
                sorted_order = np.argsort(buy_prices)
                sorted_buy_indices = valid_buy_indices[sorted_order]
                
                for i in sorted_buy_indices:
                    price = prices[i]
                    cost_per_unit = price * (1 + self.transaction_cost)
                    
                    # 確保 cost_per_unit 有效
                    if cost_per_unit <= 0:
                        trades_executed['failed_buy'] += 1
                        self.failed_buys += 1
                        continue
                    
                    # 計算可買數量
                    max_affordable = int(self.balance / cost_per_unit)
                    buy_quantity = min(self.trade_unit, max_affordable)
                    
                    if buy_quantity > 0:
                        cost = buy_quantity * cost_per_unit
                        self.balance -= cost
                        self.holdings[i] += buy_quantity
                        total_transaction_cost += buy_quantity * price * self.transaction_cost
                        trades_executed['buy'] += 1
                        self.total_trades += 1
                        self.successful_buys += 1
                    else:
                        trades_executed['failed_buy'] += 1
                        self.failed_buys += 1
        
        # 統計持有
        trades_executed['hold'] = np.sum(action == 0)
        
        return {
            'transaction_cost': total_transaction_cost,
            'trades': trades_executed
        }
    
    def _execute_trades_equal(self, action: np.ndarray, prices: np.ndarray) -> dict:
        """
        均等分配策略：預先為每支要買入的股票分配相等的資金
        
        優點：所有股票公平對待，避免先到先得
        """
        total_transaction_cost = 0.0
        trades_executed = {'buy': 0, 'sell': 0, 'hold': 0, 'failed_buy': 0}
        
        # ========== 第一階段：執行所有賣出 ==========
        sell_indices = np.where(action == -1)[0]
        for i in sell_indices:
            price = prices[i]
            # 跳過無效價格
            if price <= 0 or np.isnan(price) or np.isinf(price):
                continue
                
            sell_quantity = min(self.trade_unit, int(self.holdings[i]))
            
            if sell_quantity > 0:
                revenue = sell_quantity * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.holdings[i] -= sell_quantity
                total_transaction_cost += sell_quantity * price * self.transaction_cost
                trades_executed['sell'] += 1
                self.total_trades += 1
        
        # ========== 第二階段：均等分配資金買入 ==========
        buy_indices = np.where(action == 1)[0]
        
        # 過濾掉無效價格的股票
        valid_buy_indices = []
        for i in buy_indices:
            price = prices[i]
            if price > 0 and not np.isnan(price) and not np.isinf(price):
                valid_buy_indices.append(i)
        
        num_buys = len(valid_buy_indices)
        
        if num_buys > 0:
            # 每支股票分配的最大資金
            budget_per_stock = self.balance / num_buys
            # 也受限於最大持倉比例
            max_budget = self.portfolio_value * self.max_position_ratio
            budget_per_stock = min(budget_per_stock, max_budget)
            
            for i in valid_buy_indices:
                price = prices[i]
                cost_per_unit = price * (1 + self.transaction_cost)
                
                # 確保 cost_per_unit 有效
                if cost_per_unit <= 0:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
                    continue
                
                # 根據預算計算可買數量
                max_by_budget = int(budget_per_stock / cost_per_unit)
                buy_quantity = min(self.trade_unit, max_by_budget)
                
                # 再檢查實際餘額
                actual_affordable = int(self.balance / cost_per_unit)
                buy_quantity = min(buy_quantity, actual_affordable)
                
                if buy_quantity > 0:
                    cost = buy_quantity * cost_per_unit
                    self.balance -= cost
                    self.holdings[i] += buy_quantity
                    total_transaction_cost += buy_quantity * price * self.transaction_cost
                    trades_executed['buy'] += 1
                    self.total_trades += 1
                    self.successful_buys += 1
                else:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
        
        trades_executed['hold'] = np.sum(action == 0)
        
        return {
            'transaction_cost': total_transaction_cost,
            'trades': trades_executed
        }
    
    def _execute_trades_proportional(self, action: np.ndarray, prices: np.ndarray) -> dict:
        """
        比例縮放策略：如果資金不足，按比例減少每支股票的買入量
        
        優點：所有買入信號都會執行（可能數量較少）
        """
        total_transaction_cost = 0.0
        trades_executed = {'buy': 0, 'sell': 0, 'hold': 0, 'failed_buy': 0}
        
        # ========== 第一階段：執行所有賣出 ==========
        sell_indices = np.where(action == -1)[0]
        for i in sell_indices:
            price = prices[i]
            # 跳過無效價格
            if price <= 0 or np.isnan(price) or np.isinf(price):
                continue
                
            sell_quantity = min(self.trade_unit, int(self.holdings[i]))
            
            if sell_quantity > 0:
                revenue = sell_quantity * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.holdings[i] -= sell_quantity
                total_transaction_cost += sell_quantity * price * self.transaction_cost
                trades_executed['sell'] += 1
                self.total_trades += 1
        
        # ========== 第二階段：計算總需求並按比例縮放 ==========
        buy_indices = np.where(action == 1)[0]
        
        # 過濾掉無效價格的股票
        valid_buy_indices = []
        for i in buy_indices:
            price = prices[i]
            if price > 0 and not np.isnan(price) and not np.isinf(price):
                valid_buy_indices.append(i)
        
        if len(valid_buy_indices) > 0:
            # 計算理想情況下的總花費
            total_cost_needed = 0.0
            costs = {}
            for i in valid_buy_indices:
                price = prices[i]
                cost = self.trade_unit * price * (1 + self.transaction_cost)
                costs[i] = cost
                total_cost_needed += cost
            
            # 計算縮放比例
            if total_cost_needed > self.balance and total_cost_needed > 0:
                scale_factor = self.balance / total_cost_needed
            else:
                scale_factor = 1.0
            
            # 按比例執行買入
            for i in valid_buy_indices:
                price = prices[i]
                cost_per_unit = price * (1 + self.transaction_cost)
                
                # 確保 cost_per_unit 有效
                if cost_per_unit <= 0:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
                    continue
                
                # 縮放後的買入數量
                scaled_quantity = int(self.trade_unit * scale_factor)
                
                # 確保至少買 1 股（如果有足夠資金）
                if scaled_quantity == 0 and self.balance >= cost_per_unit:
                    scaled_quantity = 1
                
                # 最終檢查餘額
                actual_affordable = int(self.balance / cost_per_unit)
                buy_quantity = min(scaled_quantity, actual_affordable)
                
                if buy_quantity > 0:
                    cost = buy_quantity * cost_per_unit
                    self.balance -= cost
                    self.holdings[i] += buy_quantity
                    total_transaction_cost += buy_quantity * price * self.transaction_cost
                    trades_executed['buy'] += 1
                    self.total_trades += 1
                    self.successful_buys += 1
                else:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
        
        trades_executed['hold'] = np.sum(action == 0)
        
        return {
            'transaction_cost': total_transaction_cost,
            'trades': trades_executed
        }
    
    def _execute_trades_random(self, action: np.ndarray, prices: np.ndarray) -> dict:
        """
        隨機順序策略：隨機打亂買入順序
        
        優點：長期來看公平，避免固定偏好
        """
        total_transaction_cost = 0.0
        trades_executed = {'buy': 0, 'sell': 0, 'hold': 0, 'failed_buy': 0}
        
        # ========== 第一階段：執行所有賣出 ==========
        sell_indices = np.where(action == -1)[0]
        for i in sell_indices:
            price = prices[i]
            # 跳過無效價格
            if price <= 0 or np.isnan(price) or np.isinf(price):
                continue
                
            sell_quantity = min(self.trade_unit, int(self.holdings[i]))
            
            if sell_quantity > 0:
                revenue = sell_quantity * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.holdings[i] -= sell_quantity
                total_transaction_cost += sell_quantity * price * self.transaction_cost
                trades_executed['sell'] += 1
                self.total_trades += 1
        
        # ========== 第二階段：隨機順序執行買入 ==========
        buy_indices = np.where(action == 1)[0]
        
        # 過濾掉無效價格的股票
        valid_buy_indices = []
        for i in buy_indices:
            price = prices[i]
            if price > 0 and not np.isnan(price) and not np.isinf(price):
                valid_buy_indices.append(i)
        
        if len(valid_buy_indices) > 0:
            valid_buy_indices = np.array(valid_buy_indices)
            # 隨機打亂順序
            np.random.shuffle(valid_buy_indices)
            
            for i in valid_buy_indices:
                price = prices[i]
                cost_per_unit = price * (1 + self.transaction_cost)
                
                # 確保 cost_per_unit 有效
                if cost_per_unit <= 0:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
                    continue
                
                max_affordable = int(self.balance / cost_per_unit)
                buy_quantity = min(self.trade_unit, max_affordable)
                
                if buy_quantity > 0:
                    cost = buy_quantity * cost_per_unit
                    self.balance -= cost
                    self.holdings[i] += buy_quantity
                    total_transaction_cost += buy_quantity * price * self.transaction_cost
                    trades_executed['buy'] += 1
                    self.total_trades += 1
                    self.successful_buys += 1
                else:
                    trades_executed['failed_buy'] += 1
                    self.failed_buys += 1
        
        trades_executed['hold'] = np.sum(action == 0)
        
        return {
            'transaction_cost': total_transaction_cost,
            'trades': trades_executed
        }
    
    def reset(self, seed=None, options=None):
        """重置環境"""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        
        # 重置交易統計
        self.total_trades = 0
        self.successful_buys = 0
        self.failed_buys = 0
        
        # ★ 重置歷史
        self.observation_history = []
        
        # 重置歷史數據（用於 reward 計算）
        self.price_history = []
        self.return_history = []
        
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'holdings': self.holdings.copy(),
            'portfolio_value': self.portfolio_value,
            'reward_type': self.agent_type,
            'use_temporal': self.use_temporal,
            'model_type': self.model_type,
        }
        
        return observation, info
    
    def step(self, action):
        """
        執行一步
        
        :param action: 動作陣列，每支股票 [0, 1, 2] 或 [-1, 0, 1]
        :return: observation, reward, done, truncated, info
        """
        action = np.array(action).flatten()
        
        # 轉換動作格式
        if np.all(action >= 0) and np.all(action <= 2) and action.dtype in [np.int32, np.int64, int]:
            action = self._convert_action(action)
        
        # 使用收盤價進行交易
        current_prices = self._get_current_prices()
        prev_portfolio_value = self.portfolio_value
        
        # 根據策略執行交易
        if self.allocation_strategy == 'priority':
            trade_result = self._execute_trades_priority(action, current_prices)
        elif self.allocation_strategy == 'equal':
            trade_result = self._execute_trades_equal(action, current_prices)
        elif self.allocation_strategy == 'proportional':
            trade_result = self._execute_trades_proportional(action, current_prices)
        elif self.allocation_strategy == 'random':
            trade_result = self._execute_trades_random(action, current_prices)
        else:
            trade_result = self._execute_trades_priority(action, current_prices)
        
        # 移動到下一步
        self.current_step += 1
        
        # 計算新的 portfolio value（使用收盤價）
        new_prices = self._get_current_prices()
            
        self.portfolio_value = self.balance + np.sum(self.holdings * new_prices)
        
        # ← 更新歷史數據
        self.price_history.append(new_prices.copy())
        daily_return = (self.portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1e-8)
        self.return_history.append(daily_return)
        
        # ========== 使用選定的 reward 函數計算獎勵 ==========
        # 獲取當前步的基本面數據
        if self.current_step < len(self.fundamental_data):
            current_fundamental = self.fundamental_data[self.current_step]
        else:
            current_fundamental = None
        
        # 獲取當前步的技術指標
        if self.current_step < len(self.technical_indicators):
            current_technical = self.technical_indicators[self.current_step]
        else:
            current_technical = None
        
        reward_info = {
            'prev_portfolio_value': prev_portfolio_value,
            'portfolio_value': self.portfolio_value,
            'holdings': self.holdings.copy(),
            'prices': new_prices,
            'price_history': self.price_history,
            'return_history': self.return_history,
            'transaction_cost': trade_result['transaction_cost'],
            'action': action,
            'fundamental_data': current_fundamental,
            'technical_indicators': current_technical,
        }
        
        reward = self.reward_function.calculate(reward_info)
        # ====================================================
        
        # 檢查是否結束
        done = self.current_step >= len(self.stock_data) - 1
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'holdings': self.holdings.copy(),
            'portfolio_value': self.portfolio_value,
            'transaction_cost': trade_result['transaction_cost'],
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'trades': trade_result['trades'],
            'total_trades': self.total_trades,
            'successful_buys': self.successful_buys,
            'failed_buys': self.failed_buys,
            'buy_success_rate': self.successful_buys / max(1, self.successful_buys + self.failed_buys),
            'reward_type': self.agent_type,
            'daily_return': daily_return,
        }
        
        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """渲染環境狀態"""
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:,.2f}")
        print(f"Holdings: {self.holdings}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Return: {(self.portfolio_value - self.initial_balance) / self.initial_balance * 100:.2f}%")
        print(f"Buy Success Rate: {self.successful_buys / max(1, self.successful_buys + self.failed_buys) * 100:.1f}%")
        print(f"Reward Type: {self.agent_type}")
        print(f"Daily Return: {self.return_history[-1] * 100:.2f}%")
