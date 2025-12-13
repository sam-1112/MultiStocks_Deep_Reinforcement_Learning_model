import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv1D(nn.Module):
    """多尺度 1D 卷積層"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list = [3, 5, 7]):
        super(MultiScaleConv1D, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.out_channels = out_channels * len(kernel_sizes)
    
    def forward(self, x):
        """
        :param x: (batch, channels, seq_len)
        :return: (batch, out_channels * num_scales, seq_len)
        """
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        return F.relu(out)


class MSCNN(nn.Module):
    """Multi-Scale CNN - 支持標準化介面的多尺度卷積神經網絡"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none',
                 kernel_sizes: list = None, seq_len: int = 10,
                 dropout: float = 0.1, configs: dict = None, **kwargs):
        """
        初始化 MSCNN
        
        :param input_dim: 輸入維度（特徵數量）
        :param output_dim: 輸出維度（動作維度）
        :param hidden_dim: 隱藏維度
        :param n_layers: 卷積層數
        :param output_activation: 輸出激活函數 ('tanh', 'sigmoid', 'relu', 'none')
        :param kernel_sizes: 多尺度卷積核大小列表
        :param seq_len: 序列長度
        :param dropout: Dropout 比例
        :param configs: 完整配置字典（用於讀取 mscnn 特有參數）
        """
        super(MSCNN, self).__init__()
        
        # 保存配置（與 MLP、LSTM、TimesNet 統一介面）
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_activation = output_activation
        self.seq_len = seq_len
        self.dropout_rate = dropout
        
        # 從 configs 讀取 MSCNN 特有參數
        if configs is not None:
            mscnn_cfg = configs.get('mscnn', {})
            env_cfg = configs.get('env', {})
            
            # 多尺度卷積參數
            multi_scale_cfg = mscnn_cfg.get('multi_scale', {})
            backbone_cfg = mscnn_cfg.get('backbone', {})
            action_cfg = mscnn_cfg.get('action', {})
            
            in_channels = multi_scale_cfg.get('in_channels', 5)
            out_channels = multi_scale_cfg.get('out_channels', hidden_dim)
            backbone_in = backbone_cfg.get('in_channels', out_channels)
            backbone_out = backbone_cfg.get('out_channels', hidden_dim)
            
            self.seq_len = env_cfg.get('window_size', seq_len)
        else:
            in_channels = input_dim
            out_channels = hidden_dim
            backbone_in = hidden_dim
            backbone_out = hidden_dim
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.kernel_sizes = kernel_sizes
        
        print(f"[MSCNN] 初始化:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Layers: {n_layers}")
        print(f"  - Kernel sizes: {kernel_sizes}")
        print(f"  - Seq len: {self.seq_len}")
        print(f"  - Output activation: {output_activation if output_activation != 'none' else 'None'}")
        
        # 輸入投影層（將 input_dim 投影到 hidden_dim）
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度卷積層
        num_scales = len(kernel_sizes)
        self.multi_scale_conv = MultiScaleConv1D(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_scales,
            kernel_sizes=kernel_sizes
        )
        
        # 計算多尺度卷積輸出的通道數
        ms_out_channels = (hidden_dim // num_scales) * num_scales
        
        # 堆疊卷積層
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(ms_out_channels if i == 0 else hidden_dim, hidden_dim, 
                         kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # 全局平均池化後的維度
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全連接層
        fc_input_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 輸出激活函數
        if output_activation == 'tanh':
            self.output_act_fn = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_act_fn = nn.Sigmoid()
        elif output_activation == 'relu':
            self.output_act_fn = nn.ReLU()
        else:
            self.output_act_fn = None
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入張量
            - 2D: (batch_size, input_dim) -> 自動擴展
            - 3D: (batch_size, seq_len, input_dim)
        :return: 輸出張量 (batch_size, output_dim)
        """
        # 處理 2D 輸入
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, seq_len, input_dim)
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        batch_size, seq_len, features = x.size()
        
        # 輸入投影: (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # 轉換為 Conv1D 格式: (batch, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 多尺度卷積
        x = self.multi_scale_conv(x)
        
        # 堆疊卷積層
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 全局平均池化: (batch, hidden_dim, 1)
        x = self.global_pool(x)
        
        # 展平: (batch, hidden_dim)
        x = x.squeeze(-1)
        
        # 全連接層: (batch, output_dim)
        x = self.fc(x)
        
        # 應用輸出激活函數
        if self.output_act_fn is not None:
            x = self.output_act_fn(x)
        
        return x
    
    def get_config(self):
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation,
            'kernel_sizes': self.kernel_sizes,
            'seq_len': self.seq_len,
        }