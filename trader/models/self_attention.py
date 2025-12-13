import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """
    多頭自注意力機制
    
    用於整合多個 Sub-Agent 的輸出信號
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        :param dim: 輸入維度
        :param num_heads: 注意力頭數
        :param dropout: Dropout 比率
        """
        super().__init__()
        
        assert dim % num_heads == 0, f"dim ({dim}) 必須能被 num_heads ({num_heads}) 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value 投影層
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 輸出投影層
        self.proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入 (batch_size, seq_len, dim) 或 (batch_size, dim)
        :param mask: 注意力遮罩 (可選)
        :return: 輸出 (同 x 形狀)
        """
        # 處理 2D 輸入 (batch_size, dim)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, dim = x.shape
        
        # 投影到 Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 計算注意力權重
        attn = (q @ k.transpose(-2, -1)) * (1.0 / self.scale)  # (batch_size, num_heads, seq_len, seq_len)
        
        # 應用遮罩
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 應用注意力到 values
        x_attn = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)
        
        # 輸出投影
        x_attn = self.proj(x_attn)
        x_attn = self.proj_dropout(x_attn)
        
        # 還原 2D 輸入的形狀
        if squeeze_output:
            x_attn = x_attn.squeeze(1)  # (batch_size, dim)
        
        return x_attn


class AttentionFusionLayer(nn.Module):
    """
    注意力融合層
    
    融合 State 特徵和 Sub-Agent 信號
    """
    
    def __init__(self, state_dim: int, signal_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        :param state_dim: 狀態維度
        :param signal_dim: Sub-Agent 信號維度
        :param hidden_dim: 隱藏層維度
        :param num_heads: 注意力頭數
        :param dropout: Dropout 比率"""
        super().__init__()
        
        self.state_dim = state_dim
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        
        # 狀態編碼器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 信號編碼器
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合層：使用注意力機制
        self.attention = MultiHeadSelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 融合後的特徵層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param state: 狀態向量 (batch_size, state_dim)
        :param signal: Sub-Agent 信號 (batch_size, signal_dim)
        :return: 融合特徵 (batch_size, hidden_dim)
        """
        # 編碼
        state_feat = self.state_encoder(state)  # (batch_size, hidden_dim)
        signal_feat = self.signal_encoder(signal)  # (batch_size, hidden_dim)
        
        # 注意力加權融合
        state_attn = self.attention(state_feat)  # (batch_size, hidden_dim)
        
        # 連接特徵
        fused = torch.cat([state_attn, signal_feat], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # 融合層
        fused = self.fusion_layer(fused)  # (batch_size, hidden_dim)
        
        # 殘差連接
        fused = fused + state_feat
        fused = self.output_norm(fused)
        
        return fused


class StockSequenceAttention(nn.Module):
    """
    股票序列注意力層
    
    將股票狀態視為序列，使用多頭注意力處理
    """
    
    def __init__(self, num_stocks: int, feature_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) 必須能被 num_heads ({num_heads}) 整除"
        
        # 特徵投影
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多頭自注意力（在股票維度上）
        self.attention = MultiHeadSelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 聚合層
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 股票特徵 (batch_size, num_stocks, feature_dim)
        :return: 聚合特徵 (batch_size, hidden_dim)
        """
        batch_size, num_stocks, feature_dim = x.shape
        
        # 特徵投影
        x = self.feature_proj(x)  # (batch_size, num_stocks, hidden_dim)
        
        # 多頭注意力
        x = self.attention(x)  # (batch_size, num_stocks, hidden_dim)
        
        # 平均聚合
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 聚合層
        x = self.aggregate(x)  # (batch_size, hidden_dim)
        
        return x


class SimpleAttentionActor(nn.Module):
    """
    簡單注意力 Actor Network
    
    使用簡單注意力融合狀態特徵
    適用於較小的狀態空間
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 狀態編碼
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 簡單注意力層
        self.attention = MultiHeadSelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 動作輸出
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入狀態 (batch_size, state_dim)
        :return: 動作 logits (batch_size, action_dim)
        """
        # 狀態編碼
        h = self.state_encoder(x)  # (batch_size, hidden_dim)
        
        # 注意力
        attn_out = self.attention(h)  # (batch_size, hidden_dim)
        h = h + attn_out
        h = self.layer_norm(h)
        
        # 動作輸出
        out = self.action_head(h)  # (batch_size, action_dim)
        
        return out


class SequenceAttentionActor(nn.Module):
    """
    序列注意力 Actor Network
    
    將狀態視為特徵序列，使用多頭注意力處理
    適用於較大的狀態空間
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 狀態編碼（投影到 hidden_dim）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 多頭自注意力
        self.attention = MultiHeadSelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 動作輸出
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入狀態 (batch_size, state_dim)
        :return: 動作 logits (batch_size, action_dim)
        """
        # 狀態編碼
        h = self.state_encoder(x)  # (batch_size, hidden_dim)
        
        # 多頭自注意力
        attn_out = self.attention(h)  # (batch_size, hidden_dim)
        h = h + attn_out
        h = self.layer_norm(h)
        
        # 動作輸出
        out = self.action_head(h)  # (batch_size, action_dim)
        
        return out


class ContinuousAttentionActor(nn.Module):
    """
    連續動作注意力 Actor Network
    
    用於 DDPG 等連續控制算法
    輸出範圍 [-1, 1]
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 狀態編碼
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 多頭自注意力
        self.attention = MultiHeadSelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 連續動作輸出 [-1, 1]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # ← 輸出範圍 [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入狀態 (batch_size, state_dim)
        :return: 連續動作 (batch_size, action_dim) in [-1, 1]
        """
        # 狀態編碼
        h = self.state_encoder(x)  # (batch_size, hidden_dim)
        
        # 多頭自注意力
        attn_out = self.attention(h)  # (batch_size, hidden_dim)
        h = h + attn_out
        h = self.layer_norm(h)
        
        # 連續動作輸出
        out = self.action_head(h)  # (batch_size, action_dim) in [-1, 1]
        
        return out