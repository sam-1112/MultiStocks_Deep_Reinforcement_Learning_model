"""
模型工廠 - 支持多種模型類型（MLP、LSTM、BiLSTM、TimesNet、MSCNN）
"""

import torch
import torch.nn as nn
from trader.models.mlp import MLP
from trader.models.lstm import LSTM
from trader.models.timesnet import TimesNet
from trader.models.mscnn import MSCNN
from trader.models.bilstm import BiLSTM, BiLSTMActor, BiLSTMCritic


class ModelFactory:
    """改進的模型工廠 - 支持不同演算法的不同激活函數"""
    
    # Actor 模型
    _actor_models = {
        'mlp': MLP,
        'lstm': LSTM,
        'bilstm': BiLSTMActor,  
        'timesnet': TimesNet,
        'mscnn': MSCNN,
    }
    
    # Critic 模型（用於 A2C、DDPG）
    _critic_models = {
        'mlp': MLP,
        'lstm': LSTM,
        'bilstm': BiLSTMCritic,
        'timesnet': TimesNet,
        'mscnn': MSCNN,
    }
    
    # Q-Network 模型（用於 DQN、DDQN）
    _qnetwork_models = {
        'mlp': MLP,
        'lstm': LSTM,
        'bilstm': BiLSTM,
        'timesnet': TimesNet,
        'mscnn': MSCNN,
    }
    
    # 激活函數映射
    _activations = {
        'ddpg_actor': 'tanh',        # DDPG Actor: 連續值 [-1, 1]
        'a2c_actor': 'none',         # A2C Actor: logits（無激活）
        'ddqn_qnet': 'none',         # DDQN Q-Network: Q值（無激活）
        'critic': 'none',            # Critic: 狀態值（無激活）
        'value': 'none',             # Value Network: 價值（無激活）
    }

    @staticmethod
    def create_actor(model_type: str, input_dim: int, output_dim: int,
                     actor_type: str = 'continuous', hidden_dim: int = 128,
                     n_layers: int = 2, use_attention: bool = False,
                     num_heads: int = 4, attention_type: str = 'simple',
                     configs: dict = None, **kwargs):
        """
        創建 Actor 網絡
        
        :param model_type: 模型類型 ('mlp', 'bilstm', 'lstm', 'timesnet', 'mscnn')
        :param input_dim: 輸入維度（特徵數）
        :param output_dim: 輸出維度（股票數 × 動作數 或 動作維度）
        :param actor_type: Actor 類型 ('continuous' for DDPG, 'discrete' for A2C)
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :param use_attention: 是否使用注意力機制
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型 ('simple', 'sequence')
        :param configs: 完整配置字典
        :param kwargs: 其他參數（dropout、action_range 等）
        :return: Actor 網絡模型
        """
        if model_type not in ModelFactory._actor_models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(ModelFactory._actor_models.keys())}")
        
        # 根據 actor_type 選擇激活函數
        if actor_type == 'continuous':
            activation = ModelFactory._activations['ddpg_actor']  # tanh
        else:
            activation = ModelFactory._activations['a2c_actor']  # none
        
        print(f"\n[ModelFactory] 創建 Actor 網絡")
        print(f"  - 模型類型: {model_type.upper()}")
        print(f"  - Actor 類型: {actor_type} ({'連續動作' if actor_type == 'continuous' else '離散動作'})")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: {output_dim}")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: {activation}")
        if use_attention:
            print(f"  - 注意力機制: ✓ ({attention_type}, {num_heads} heads)")
        print()
        
        # ========== BiLSTM 特殊處理 ==========
        if model_type == 'bilstm':
            if actor_type == 'discrete':
                # 離散 Actor（A2C）- BiLSTMActor
                return BiLSTMActor(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=kwargs.get('dropout', 0.2),
                    configs=configs
                )
            else:
                # 連續 Actor（DDPG）- BiLSTM with Tanh
                return BiLSTM(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    output_activation=activation,  # tanh
                    dropout=kwargs.get('dropout', 0.2),
                    configs=configs
                )
        
        # ========== 其他模型 ==========
        actor_class = ModelFactory._actor_models[model_type]
        
        return actor_class(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_activation=activation,
            configs=configs,
            **kwargs
        )
    
    @staticmethod
    def create_critic(model_type: str, input_dim: int, hidden_dim: int = 128,
                     n_layers: int = 2, use_attention: bool = False,
                     num_heads: int = 4, attention_type: str = 'simple',
                     configs: dict = None, **kwargs):
        """
        創建 Critic 網絡
        
        用於 A2C、PPO 等演算法，評估狀態價值（無輸出層激活）
        
        :param model_type: 模型類型 ('mlp', 'bilstm', 'lstm', etc.)
        :param input_dim: 輸入維度（狀態維度）
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :param use_attention: 是否使用注意力機制
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型
        :param configs: 完整配置字典
        :param kwargs: 其他參數
        :return: Critic 網絡模型
        """
        if model_type not in ModelFactory._critic_models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(ModelFactory._critic_models.keys())}")
        
        activation = ModelFactory._activations['critic']  # none
        
        print(f"\n[ModelFactory] 創建 Critic 網絡")
        print(f"  - 模型類型: {model_type.upper()}")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: 1（狀態價值）")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: {activation}")
        if use_attention:
            print(f"  - 注意力機制: ✓ ({attention_type}, {num_heads} heads)")
        print()
        
        # ========== BiLSTM 特殊處理 ==========
        if model_type == 'bilstm':
            return BiLSTMCritic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=kwargs.get('dropout', 0.2),
                configs=configs
            )
        
        # ========== 其他模型 ==========
        critic_class = ModelFactory._critic_models[model_type]
        
        return CriticWrapper(
            critic_class(
                input_dim=input_dim,
                output_dim=1,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                output_activation=activation,
                configs=configs,
                **kwargs
            )
        )
    
    @staticmethod
    def create_qnetwork(model_type: str, input_dim: int, output_dim: int,
                       hidden_dim: int = 128, n_layers: int = 2,
                       use_attention: bool = False, num_heads: int = 4,
                       attention_type: str = 'simple',
                       configs: dict = None, **kwargs):
        """
        創建 Q-Network
        
        用於 DQN、DDQN 等演算法，輸出每個動作的 Q 值（無激活）
        
        :param model_type: 模型類型 ('mlp', 'bilstm', 'lstm', etc.)
        :param input_dim: 輸入維度
        :param output_dim: 輸出維度（動作空間大小）
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層層數
        :param use_attention: 是否使用注意力機制
        :param num_heads: 注意力頭數
        :param attention_type: 注意力類型
        :param configs: 完整配置字典
        :param kwargs: 其他參數
        :return: Q-Network 模型
        """
        if model_type not in ModelFactory._qnetwork_models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(ModelFactory._qnetwork_models.keys())}")
        
        activation = ModelFactory._activations['ddqn_qnet']  # none
        
        print(f"\n[ModelFactory] 創建 Q-Network (DDQN 風格)")
        print(f"  - 模型類型: {model_type.upper()}")
        print(f"  - 輸入維度: {input_dim}")
        print(f"  - 輸出維度: {output_dim}（Q 值）")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 隱藏層數: {n_layers}")
        print(f"  - 輸出激活: {activation}")
        if use_attention:
            print(f"  - 注意力機制: ✓ ({attention_type}, {num_heads} heads)")
        print()
        
        # ========== BiLSTM 特殊處理 ==========
        if model_type == 'bilstm':
            return BiLSTM(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                output_activation=activation,
                dropout=kwargs.get('dropout', 0.2),
                configs=configs
            )
        
        # ========== 其他模型 ==========
        qnetwork_class = ModelFactory._qnetwork_models[model_type]
        
        return qnetwork_class(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_activation=activation,
            configs=configs,
            **kwargs
        )
    
    @staticmethod
    def register_actor(name: str, actor_class):
        """動態註冊新的 Actor 模型"""
        ModelFactory._actor_models[name] = actor_class
        print(f"[ModelFactory] ✓ 已註冊 Actor 模型: {name}")
    
    @staticmethod
    def register_critic(name: str, critic_class):
        """動態註冊新的 Critic 模型"""
        ModelFactory._critic_models[name] = critic_class
        print(f"[ModelFactory] ✓ 已註冊 Critic 模型: {name}")
    
    @staticmethod
    def register_qnetwork(name: str, qnetwork_class):
        """動態註冊新的 Q-Network 模型"""
        ModelFactory._qnetwork_models[name] = qnetwork_class
        print(f"[ModelFactory] ✓ 已註冊 Q-Network 模型: {name}")
    
    @staticmethod
    def get_available_models():
        """獲取所有可用的模型類型"""
        return {
            'actors': list(ModelFactory._actor_models.keys()),
            'critics': list(ModelFactory._critic_models.keys()),
            'qnetworks': list(ModelFactory._qnetwork_models.keys()),
        }
    
    @staticmethod
    def print_model_info():
        """打印可用的模型信息"""
        available = ModelFactory.get_available_models()
        print("\n[ModelFactory] 可用的模型類型:")
        print(f"  - Actors: {', '.join(available['actors'])}")
        print(f"  - Critics: {', '.join(available['critics'])}")
        print(f"  - Q-Networks: {', '.join(available['qnetwork'])}")


class CriticWrapper(nn.Module):
    """Critic 包裝器 - 自動處理不同維度的輸入"""
    
    def __init__(self, base_critic):
        """
        :param base_critic: 基礎 Critic 模型
        """
        super().__init__()
        self.base_critic = base_critic
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入張量
                - 平面: (batch, state_dim + action_dim)
                - 時序: (batch, window_size, state_dim + action_dim)
        
        Returns:
            價值輸出 (batch, 1)
        """
        return self.base_critic(x)
    
    def state_dict(self):
        """獲取狀態字典"""
        return self.base_critic.state_dict()
    
    def load_state_dict(self, state_dict):
        """加載狀態字典"""
        self.base_critic.load_state_dict(state_dict)
    
    def parameters(self):
        """獲取參數"""
        return self.base_critic.parameters()
    
    def to(self, device):
        """移動到設備"""
        self.base_critic.to(device)
        return self