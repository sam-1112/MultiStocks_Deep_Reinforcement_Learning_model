import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        """
        Forward pass for the token embedding layer.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return x: 
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionEmbedding layer with sinusoidal position encodings.

        :param d_model: Dimension of the output embedding
        :param max_len: Maximum length of the input sequence
        """
        super(PositionEmbedding, self).__init__()
        self.outputDim = d_model
        self.maxlen = max_len

        pe = torch.zeros(self.maxlen, self.outputDim).float()
        pe.requires_grad = False

        position = torch.arange(0, self.maxlen).unsqueeze(1).float()
        div_term = (torch.arange(0, self.outputDim, 2).float() * -(np.log(10000.0) / self.outputDim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass for the position embedding layer.

        :param x: Input tensor of shape (batch_size, seq_length, d_model)
        :return: Position embeddings of shape (1, seq_length, d_model)
        """
        return self.pe[:, :x.size(1)]
        

class EmbeddingBlock(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        """
        Data embedding = Position embedding + Token embedding

        :param c_in: Dimension of the input data
        :param d_model: Dimension of the output embedding
        :param dropout: Dropout rate
        """
        super(EmbeddingBlock, self).__init__()
        self.inputDim = c_in
        self.outputDim = d_model
        self.tokenEmbedding = TokenEmbedding(c_in, d_model)
        self.positionEmbedding = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        """
        Forward pass for the embedding block.
        
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Output tensor of shape (batch_size, seq_length, d_model)
        """
        x = self.tokenEmbedding(x) + self.positionEmbedding(x)
        return self.dropout(x)

    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Initializes the InceptionBlock with multiple convolutional kernels.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_kernels: Number of different kernel sizes to use
        :param init_weight: Whether to initialize weights (default: True)
        """
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initializes the weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for the Inception block.

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return res: Output tensor
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        return res
        

class TimesBlock(nn.Module):

    def __init__(self, seq_len, pred_len, d_model, d_ff, num_kernels, top_k):
        """
        Initializes the TimesBlock.

        :param seq_len: Sequence length
        :param pred_len: Prediction length
        :param d_model: Model dimension
        :param d_ff: Feed-forward dimension
        :param num_kernels: Number of kernels in InceptionBlock
        :param top_k: Top k periods for FFT
        """
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels=num_kernels)
        )
    
    def forward(self, x):
        """
        Forward pass for the TimesBlock.

        :param x: Input tensor of shape (batch_size, seq_len, d_model)
        :return res: Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, N = x.size()
        period_list, period_weight = self.FFTforPeriod(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 確保 period 至少為 1
            period = max(1, int(period))
            
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros((B, length - T, N)).to(x.device)
                out = torch.cat((x, padding), dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, dim=-1)
        res = res + x 
        return res         

    def FFTforPeriod(self, x, k=2):
        """
        Computes the FFT to identify the top k periods.

        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :param k: Number of top periods to identify
        :return period: Array of top k periods
        :return frequency_weights: Tensor of frequency weights
        """
        xf = torch.fft.rfft(x.float(), dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        
        # 確保 k 不超過可用的頻率數量
        k = min(k, len(frequency_list))
        
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        
        # 避免除以零
        top_list = np.maximum(top_list, 1)
        period = x.shape[1] // top_list
        period = np.maximum(period, 1)  # 確保週期至少為 1
        
        return period, abs(xf).mean(-1)[:, :k]
    

class TimesNet(nn.Module):
    """TimesNet - 支持標準化介面的時序模型"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none',
                 seq_len: int = 10, top_k: int = 5, num_kernels: int = 6,
                 dropout: float = 0.1, configs: dict = None, **kwargs):
        """
        初始化 TimesNet
        
        支援兩種初始化方式：
        1. 標準化介面（與 MLP、LSTM 一致）
        2. 從 configs 字典讀取參數
        
        :param input_dim: 輸入維度（特徵數量）
        :param output_dim: 輸出維度（動作維度）
        :param hidden_dim: 隱藏維度 (d_model)
        :param n_layers: 層數 (e_layers)
        :param output_activation: 輸出激活函數 ('none', 'tanh', 'sigmoid', 'relu')
        :param seq_len: 序列長度
        :param top_k: FFT top-k 週期
        :param num_kernels: Inception block 的 kernel 數量
        :param dropout: Dropout 比例
        :param configs: 完整配置字典（可選，會覆蓋上述參數）
        """
        super(TimesNet, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果提供了 configs，從中提取 TimesNet 特有參數
        if configs is not None and 'timesnet' in configs:
            timesnet_cfg = configs['timesnet']
            training_cfg = configs.get('training', {})
            env_cfg = configs.get('env', {})
            
            # 從 timesnet 配置讀取
            hidden_dim = timesnet_cfg.get('d_model', hidden_dim)
            n_layers = timesnet_cfg.get('e_layers', n_layers)
            top_k = timesnet_cfg.get('top_k', top_k)
            num_kernels = timesnet_cfg.get('num_kernels', num_kernels)
            d_ff = timesnet_cfg.get('d_ff', hidden_dim * 4)
            
            # 從 training/env 配置讀取
            seq_len = env_cfg.get('window_size', training_cfg.get('seq_len', seq_len))
            dropout = training_cfg.get('dropout', dropout)
        else:
            # d_ff 預設為 d_model 的 4 倍
            d_ff = kwargs.get('d_ff', hidden_dim * 4)
        
        # 保存配置（與 MLP、LSTM 統一介面）
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_activation = output_activation
        
        # TimesNet 特有參數
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout_rate = dropout
        self.d_ff = d_ff
        
        print(f"[TimesNet] 初始化:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Hidden dim (d_model): {hidden_dim}")
        print(f"  - d_ff: {d_ff}")
        print(f"  - Layers: {n_layers}")
        print(f"  - Seq len: {seq_len}")
        print(f"  - Top-k: {top_k}")
        print(f"  - Num kernels: {num_kernels}")
        print(f"  - Output activation: {output_activation if output_activation != 'none' else 'None'}")
        
        # Embedding 層
        self.embedding = EmbeddingBlock(input_dim, hidden_dim, dropout)
        
        # TimesBlock 層
        self.model = nn.ModuleList([
            TimesBlock(seq_len, 1, hidden_dim, d_ff, num_kernels, top_k)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 分類/輸出層
        self.activation = F.gelu
        self.dropout = nn.Dropout(p=dropout)
        
        # 計算展平後的維度
        flatten_dim = hidden_dim * seq_len
        projection_hidden = max(flatten_dim // 4, 64)  # 確保至少有 64 維
        
        self.projection1 = nn.Linear(flatten_dim, projection_hidden)
        self.projection2 = nn.Linear(projection_hidden, output_dim)
        
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
        """初始化全連接層權重"""
        nn.init.xavier_uniform_(self.projection1.weight)
        nn.init.zeros_(self.projection1.bias)
        nn.init.xavier_uniform_(self.projection2.weight)
        nn.init.zeros_(self.projection2.bias)
    
    def forward(self, x: torch.Tensor, x_mark_enc: torch.Tensor = None) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入張量
            - 2D: (batch_size, input_dim) -> 自動擴展
            - 3D: (batch_size, seq_len, input_dim)
        :param x_mark_enc: 時間標記（可選）
        :return: 輸出張量 (batch_size, output_dim)
        """
        # 處理 2D 輸入（與 MLP、LSTM 統一）
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, seq_len, input_dim)
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        B, T, C = x.size()
        
        # 如果輸入特徵維度與預期不符，進行調整
        if C != self.input_dim:
            # 使用線性投影調整維度（懶初始化）
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(C, self.input_dim).to(x.device)
                nn.init.xavier_uniform_(self.input_projection.weight)
                nn.init.zeros_(self.input_projection.bias)
            x = self.input_projection(x)
            C = self.input_dim
        
        # 如果序列長度不匹配，進行調整
        if T != self.seq_len:
            if T < self.seq_len:
                # 補零
                padding = torch.zeros(B, self.seq_len - T, C, device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # 截斷（取最後 seq_len 個時間步）
                x = x[:, -self.seq_len:, :]
        
        # Embedding
        enc_out = self.embedding(x)  # [B, seq_len, hidden_dim]
        
        # TimesBlock layers
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # 輸出處理
        output = self.activation(enc_out)
        output = self.dropout(output)
        
        # 時間標記加權（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        
        # 展平
        output = output.reshape(B, -1)  # [B, seq_len * hidden_dim]
        
        # 全連接層
        output = self.projection1(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.projection2(output)  # [B, output_dim]
        
        # 應用輸出激活函數
        if self.output_act_fn is not None:
            output = self.output_act_fn(output)
        
        return output
    
    def classification(self, x_enc, x_mark_enc=None):
        """保持向後相容的分類方法"""
        return self.forward(x_enc, x_mark_enc)
    
    def get_config(self):
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation,
            'seq_len': self.seq_len,
            'top_k': self.top_k,
            'num_kernels': self.num_kernels,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
        }