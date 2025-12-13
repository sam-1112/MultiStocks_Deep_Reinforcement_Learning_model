import torch
import torch.nn as nn

class MLP(nn.Module):
    """多層感知機 - 支持不同的激活函數"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none',
                 configs: dict = None, **kwargs):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入張量
            - 2D: (batch_size, input_dim)
            - 3D: (batch_size, seq_len, input_dim) -> 自動展平
        :return: 輸出張量 (batch_size, output_dim)
        """
        # ★ 如果輸入是 3D（時序），展平為 2D
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.size()
            x = x.reshape(batch_size, -1)  # (batch, seq_len * feature_dim)
            
            # 如果展平後維度不匹配，使用線性投影
            flattened_dim = seq_len * feature_dim
            if flattened_dim != self.input_dim:
                if not hasattr(self, 'input_projection'):
                    self.input_projection = nn.Linear(flattened_dim, self.input_dim).to(x.device)
                    nn.init.xavier_uniform_(self.input_projection.weight)
                    nn.init.zeros_(self.input_projection.bias)
                x = self.input_projection(x)
        
        # 使用 self.net 進行前向傳播
        x = self.net(x)
        
        return x
    
    def get_config(self):
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation
        }