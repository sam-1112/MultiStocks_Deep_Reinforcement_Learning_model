import numpy as np
from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Reward 基類"""
    
    @abstractmethod
    def calculate(self, info: dict) -> float:
        """計算獎勵"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """獎勵函數名稱"""
        pass


class DirectionAgentReward(BaseReward):
    """方向預測 Agent 的獎勵 - 關注價格變動方向"""
    
    @property
    def name(self) -> str:
        return "direction"
    
    def calculate(self, info: dict) -> float:
        """
        獎勵公式：預測正確方向獲得正獎勵
        
        R = Σ (action_i * return_i) / num_stocks
        """
        action = info.get('action', np.zeros(1))
        prices = info.get('prices', np.zeros(1))
        price_history = info.get('price_history', [])
        
        if len(price_history) < 2:
            return 0.0
        
        prev_prices = price_history[-2] if len(price_history) >= 2 else prices
        price_returns = (prices - prev_prices) / (prev_prices + 1e-8)
        
        # action: -1 (賣), 0 (持有), 1 (買)
        # 正確方向：買入且價格上漲，或賣出且價格下跌
        direction_reward = np.sum(action * price_returns) / len(action)
        
        return float(np.clip(direction_reward, -1.0, 1.0))


class FundamentalScoreAgentReward(BaseReward):
    """基本面評分 Agent 的獎勵 - 關注基本面品質"""
    
    @property
    def name(self) -> str:
        return "fundamental"
    
    def calculate(self, info: dict) -> float:
        """
        獎勵公式：持有高品質股票獲得正獎勵
        
        R = portfolio_return * fundamental_quality_score
        """
        portfolio_value = info.get('portfolio_value', 0)
        prev_portfolio_value = info.get('prev_portfolio_value', 1)
        holdings = info.get('holdings', np.zeros(1))
        fundamental_data = info.get('fundamental_data', None)
        
        # 基本回報
        portfolio_return = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        
        if fundamental_data is None or np.sum(np.abs(holdings)) == 0:
            return float(portfolio_return)
        
        # 計算持倉的基本面加權分數
        # 假設 fundamental_data 的第一個特徵是品質分數
        if len(fundamental_data.shape) > 1:
            quality_scores = fundamental_data[:, -1]  # 最後一個特徵作為品質分數
        else:
            quality_scores = fundamental_data
        
        # 標準化品質分數到 [0, 1]
        score_min = np.min(quality_scores)
        score_max = np.max(quality_scores)
        if score_max - score_min > 1e-8:
            quality_scores = (quality_scores - score_min) / (score_max - score_min)
        else:
            quality_scores = np.ones_like(quality_scores) * 0.5
        
        # 持倉加權平均品質
        abs_holdings = np.abs(holdings)
        if np.sum(abs_holdings) > 0:
            weighted_quality = np.sum(abs_holdings * quality_scores) / np.sum(abs_holdings)
        else:
            weighted_quality = 0.5
        
        # 組合獎勵：回報 * 品質加成
        reward = portfolio_return * (1 + 0.5 * weighted_quality)
        
        return float(np.clip(reward, -1.0, 1.0))


class RiskRegimeAgentReward(BaseReward):
    """風險狀態 Agent 的獎勵 - 關注風險調整後回報"""
    
    def __init__(self, risk_free_rate: float = 0.02, lookback: int = 20):
        self.risk_free_rate = risk_free_rate / 252  # 日化
        self.lookback = lookback
    
    @property
    def name(self) -> str:
        return "risk_regime"
    
    def calculate(self, info: dict) -> float:
        """
        獎勵公式：Sharpe Ratio 風格的風險調整回報
        
        R = (portfolio_return - risk_free_rate) / volatility
        """
        portfolio_value = info.get('portfolio_value', 0)
        prev_portfolio_value = info.get('prev_portfolio_value', 1)
        return_history = info.get('return_history', [])
        
        daily_return = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        
        # 計算歷史波動率
        if len(return_history) >= self.lookback:
            recent_returns = np.array(return_history[-self.lookback:])
            volatility = np.std(recent_returns) + 1e-8
        else:
            volatility = 0.02  # 默認波動率
        
        # 風險調整回報
        excess_return = daily_return - self.risk_free_rate
        risk_adjusted_return = excess_return / volatility
        
        # 額外獎勵：低波動率時期持有
        volatility_bonus = 0.0
        if volatility < 0.01:  # 低波動率
            volatility_bonus = 0.1
        elif volatility > 0.05:  # 高波動率
            volatility_bonus = -0.1
        
        reward = risk_adjusted_return * 0.1 + volatility_bonus  # 縮放風險調整回報
        
        return float(np.clip(reward, -1.0, 1.0))


class FinalAgentReward(BaseReward):
    """最終決策 Agent 的獎勵 - 綜合考慮多個因素"""
    
    def __init__(self, weights: dict = None):
        self.direction_reward = DirectionAgentReward()
        self.fundamental_reward = FundamentalScoreAgentReward()
        self.risk_reward = RiskRegimeAgentReward()
        
        # 各獎勵的權重
        self.weights = weights or {
            'portfolio': 0.4,
            'direction': 0.2,
            'fundamental': 0.2,
            'risk': 0.2
        }
    
    @property
    def name(self) -> str:
        return "final"
    
    def calculate(self, info: dict) -> float:
        """
        獎勵公式：加權組合多個獎勵信號
        
        R = w1 * portfolio_return + w2 * direction + w3 * fundamental + w4 * risk_adjusted
        """
        portfolio_value = info.get('portfolio_value', 0)
        prev_portfolio_value = info.get('prev_portfolio_value', 1)
        
        # 基本回報
        portfolio_return = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        
        # 各子獎勵
        direction_r = self.direction_reward.calculate(info)
        fundamental_r = self.fundamental_reward.calculate(info)
        risk_r = self.risk_reward.calculate(info)
        
        # 加權組合
        reward = (
            self.weights['portfolio'] * portfolio_return +
            self.weights['direction'] * direction_r +
            self.weights['fundamental'] * fundamental_r +
            self.weights['risk'] * risk_r
        )
        
        return float(np.clip(reward, -1.0, 1.0))


# ============ 獎勵函數工廠 ============

_REWARD_REGISTRY = {
    'direction': DirectionAgentReward,
    'fundamental': FundamentalScoreAgentReward,
    'risk_regime': RiskRegimeAgentReward,
    'final': FinalAgentReward,
}


def get_reward_function(agent_type: str, **kwargs) -> BaseReward:
    """
    根據 agent 類型獲取對應的獎勵函數
    
    :param agent_type: 'direction', 'fundamental', 'risk_regime', 'final'
    :param kwargs: 傳遞給獎勵函數的額外參數
    :return: BaseReward 實例
    """
    if agent_type not in _REWARD_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Available: {list(_REWARD_REGISTRY.keys())}")
    
    reward_class = _REWARD_REGISTRY[agent_type]
    
    # 某些獎勵函數需要額外參數
    if agent_type == 'risk_regime':
        return reward_class(
            risk_free_rate=kwargs.get('risk_free_rate', 0.02),
            lookback=kwargs.get('lookback_period', 20)
        )
    elif agent_type == 'final':
        return reward_class(
            weights=kwargs.get('reward_weights', None)
        )
    
    return reward_class()


def register_reward(name: str, reward_class: type):
    """註冊自定義獎勵函數"""
    if not issubclass(reward_class, BaseReward):
        raise TypeError(f"reward_class must be a subclass of BaseReward")
    _REWARD_REGISTRY[name] = reward_class
    print(f"[RewardRegistry] 已註冊獎勵函數: {name}")


def get_available_rewards() -> list:
    """獲取所有可用的獎勵函數類型"""
    return list(_REWARD_REGISTRY.keys())