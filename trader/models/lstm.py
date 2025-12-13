import torch
import torch.nn as nn

class LSTM(nn.Module):
    """長短期記憶網絡 - 支持不同的激活函數"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none',
                 configs: dict = None, **kwargs):
        super(LSTM, self).__init__()
        
        # 保存配置
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_activation = output_activation
        
        print(f"[LSTM] 初始化:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Layers: {n_layers}")
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0
        )
        
        # 全連接輸出層
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 根據需要添加輸出層激活函數
        if output_activation == 'tanh':
            self.output_act_fn = nn.Tanh()
            print(f"  - Output activation: Tanh (for DDPG Actor)")
        elif output_activation == 'sigmoid':
            self.output_act_fn = nn.Sigmoid()
            print(f"  - Output activation: Sigmoid")
        elif output_activation == 'relu':
            self.output_act_fn = nn.ReLU()
            print(f"  - Output activation: ReLU")
        else:  # 'none'
            self.output_act_fn = None
            print(f"  - Output activation: None")
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入張量
            - 2D: (batch_size, input_dim) -> 自動擴展為 (batch, 1, input_dim)
            - 3D: (batch_size, seq_len, input_dim) -> 直接使用
        :return: 輸出張量 (batch_size, output_dim)
        """
        # ★ 如果輸入是 2D，擴展為 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # 如果特徵維度不匹配，投影
        if x.size(2) != self.input_dim:
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(x.size(2), self.input_dim).to(x.device)
                nn.init.xavier_uniform_(self.input_projection.weight)
                nn.init.zeros_(self.input_projection.bias)
            x = self.input_projection(x)
        
        # LSTM 前向傳播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最後一個時間步的輸出
        out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # 全連接層
        out = self.fc(out)
        
        # 輸出激活
        if self.output_act_fn is not None:
            out = self.output_act_fn(out)
        
        return out
    
    def get_config(self):
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation
        }