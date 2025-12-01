from trader.algos.a2c import A2CStrategy
from trader.algos.ddpg import DDPGStrategy
from trader.algos.base_algo import AlgorithmStrategy

from trader.algos.ddpg import DDPGStrategy
from trader.algos.a2c import A2CStrategy
from trader.algos.ddqn import DDQNStrategy

class AlgorithmFactory:
    """工廠模式 - 所有演算法統一使用離散動作空間"""
    
    _strategies = {
        'ddpg': DDPGStrategy,
        'a2c': A2CStrategy,
        'ddqn': DDQNStrategy,
    }
    
    @staticmethod
    def get_available_algorithms():
        return list(AlgorithmFactory._strategies.keys())
    
    @staticmethod
    def create(algorithm: str, state_dim: int, action_dim: int, 
               model_type: str = 'mlp', k: int = 5, **kwargs):
        """
        創建演算法代理
        
        :param algorithm: 演算法名稱
        :param state_dim: 狀態維度
        :param action_dim: 動作維度（股票數量）
        :param model_type: 模型類型
        :param k: 動作範圍 [-k, k]
        :param kwargs: 其他參數
        :return: 演算法代理實例
        """
        if algorithm not in AlgorithmFactory._strategies:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"[AlgorithmFactory] Creating {algorithm.upper()} agent...")
        print(f"  - State dim: {state_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Action space: [-{k}, ..., 0, ..., {k}]")
        print(f"  - Model type: {model_type}\n")
        
        strategy_class = AlgorithmFactory._strategies[algorithm]
        return strategy_class(state_dim, action_dim, model_type=model_type, k=k, **kwargs)
    

from trader.models.mlp import MLP
from abc import ABC, abstractmethod

class ModelFactory:
    """改進的模型工廠 - 支持不同演算法的不同激活函數"""
    
    # Actor 模型
    _actor_models = {
        'mlp': MLP,
    }
    
    # Critic 模型（用於 A2C、DDPG）
    _critic_models = {
        'mlp': MLP,
    }
    
    # Q-Network 模型（用於 DQN、DDQN）
    _qnetwork_models = {
        'mlp': MLP,
    }
    
    # ← 激活函數映射
    _activations = {
        'ddpg_actor': 'tanh',        # DDPG Actor: 連續值 [-1, 1]
        'a2c_actor': 'none',          # A2C Actor: logits（無激活）
        'ddqn_qnet': 'none',          # DDQN Q-Network: Q值（無激活）
        'critic': 'none',             # Critic: 狀態值（無激活）
        'value': 'none',              # Value Network: 價值（無激活）
    }

    @staticmethod
    def create_actor(model_type: str, input_dim: int, output_dim: int,
                     actor_type: str = 'continuous', hidden_dim: int = 128,
                     n_layers: int = 2, **kwargs):
        """
        創建 Actor 網絡
        
        :param model_type: 模型類型 ('mlp', 'cnn', 'lstm')
        :param input_dim: 輸入維度
        :param output_dim: 輸出維度
        :param actor_type: Actor 類型 ('continuous' for DDPG, 'discrete' for A2C)
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :param kwargs: 其他參數
        :return: Actor 網絡模型
        """
        if model_type not in ModelFactory._actor_models:
            raise ValueError(f"Unknown actor model: {model_type}")
        
        # ← 根據 actor_type 選擇激活函數
        if actor_type == 'continuous':
            activation = 'tanh'  # DDPG Actor 用 Tanh
            print(f"[ModelFactory] 創建連續 Actor (DDPG 風格)")
        else:  # discrete
            activation = 'none'   # A2C Actor 用無激活（logits）
            print(f"[ModelFactory] 創建離散 Actor (A2C 風格)")
        
        actor_class = ModelFactory._actor_models[model_type]
        
        print(f"  - 模型: {model_type.upper()}")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: {output_dim}")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: {activation}\n")
        
        return actor_class(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_activation=activation,
            **kwargs
        )
    
    @staticmethod
    def create_critic(model_type: str, input_dim: int, hidden_dim: int = 128,
                     n_layers: int = 2, **kwargs):
        """
        創建 Critic 網絡
        
        用於 A2C、PPO 等演算法
        Critic 評估狀態價值，輸出無激活函數（可以是任意實數）
        
        :param model_type: 模型類型
        :param input_dim: 輸入維度（通常是狀態維度）
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :return: Critic 網絡模型
        """
        if model_type not in ModelFactory._critic_models:
            raise ValueError(f"Unknown critic model: {model_type}")
        
        critic_class = ModelFactory._critic_models[model_type]
        
        print(f"[ModelFactory] 創建 Critic 網絡")
        print(f"  - 模型: {model_type.upper()}")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: 1 (狀態價值)")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: none\n")
        
        return critic_class(
            input_dim=input_dim,
            output_dim=1,  # ← 輸出單一價值
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_activation='none',  # ← Critic 無激活
            **kwargs
        )
    
    @staticmethod
    def create_qnetwork(model_type: str, input_dim: int, output_dim: int,
                       hidden_dim: int = 128, n_layers: int = 2, **kwargs):
        """
        創建 Q-Network
        
        用於 DQN、DDQN 等演算法
        Q-Network 輸出每個動作的 Q 值，無激活函數
        
        :param model_type: 模型類型
        :param input_dim: 輸入維度
        :param output_dim: 輸出維度（動作空間大小）
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :return: Q-Network 模型
        """
        if model_type not in ModelFactory._qnetwork_models:
            raise ValueError(f"Unknown q-network model: {model_type}")
        
        qnetwork_class = ModelFactory._qnetwork_models[model_type]
        
        print(f"[ModelFactory] 創建 Q-Network")
        print(f"  - 模型: {model_type.upper()}")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: {output_dim} (Q 值)")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: none\n")
        
        return qnetwork_class(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_activation='none',  # ← Q-Network 無激活
            **kwargs
        )
    
    @staticmethod
    def register_actor(name: str, actor_class):
        """動態註冊新的 Actor 模型"""
        ModelFactory._actor_models[name] = actor_class
        print(f"[ModelFactory] 已註冊 Actor 模型: {name}")
    
    @staticmethod
    def register_critic(name: str, critic_class):
        """動態註冊新的 Critic 模型"""
        ModelFactory._critic_models[name] = critic_class
        print(f"[ModelFactory] 已註冊 Critic 模型: {name}")
    
    @staticmethod
    def register_qnetwork(name: str, qnetwork_class):
        """動態註冊新的 Q-Network 模型"""
        ModelFactory._qnetwork_models[name] = qnetwork_class
        print(f"[ModelFactory] 已註冊 Q-Network 模型: {name}")
    
    @staticmethod
    def get_available_models():
        """獲取所有可用的模型類型"""
        return {
            'actors': list(ModelFactory._actor_models.keys()),
            'critics': list(ModelFactory._critic_models.keys()),
            'qnetworks': list(ModelFactory._qnetwork_models.keys()),
        }