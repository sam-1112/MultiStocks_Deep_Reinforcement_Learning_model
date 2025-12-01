import torch
import torch.nn as nn

class MLP(nn.Module):
    """多層感知機 - 支持不同的激活函數"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none'):
        """
        初始化 MLP
        
        :param input_dim: 輸入維度
        :param output_dim: 輸出維度
        :param hidden_dim: 隱藏層維度
        :param n_layers: 隱藏層數量
        :param output_activation: 輸出層激活函數
                - 'none': 無激活（用於 Critic、Q-Network）
                - 'tanh': Tanh 激活（用於 DDPG Actor，限制在 [-1, 1]）
                - 'sigmoid': Sigmoid 激活（用於概率輸出）
                - 'relu': ReLU 激活（用於正值輸出）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_activation = output_activation
        
        # 驗證激活函數
        valid_activations = ['none', 'tanh', 'sigmoid', 'relu']
        if output_activation not in valid_activations:
            raise ValueError(f"Invalid activation: {output_activation}. "
                           f"Must be one of {valid_activations}")
        
        # 構建隱藏層
        layers = []
        prev_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 輸出層（先不加激活函數）
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 根據需要添加輸出層激活函數
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
            print(f"  - Output activation: Tanh (for DDPG Actor)")
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            print(f"  - Output activation: Sigmoid")
        elif output_activation == 'relu':
            layers.append(nn.ReLU())
            print(f"  - Output activation: ReLU")
        else:  # 'none'
            print(f"  - Output activation: None")
        
        self.net = nn.Sequential(*layers)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """前向傳播"""
        return self.net(x)
    
    def get_config(self):
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation
        }