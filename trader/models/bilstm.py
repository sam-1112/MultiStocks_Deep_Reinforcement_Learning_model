import torch
import torch.nn as nn
import numpy as np


class BiLSTM(nn.Module):
    """雙向 LSTM 模型 - 時間序列模式"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, output_activation: str = 'none', 
                 dropout: float = 0.2, configs: dict = None, **kwargs):
        """
        初始化 BiLSTM
        
        :param input_dim: 輸入維度（特徵數）
        :param output_dim: 輸出維度
        :param hidden_dim: LSTM 隱藏層維度
        :param n_layers: LSTM 層數
        :param output_activation: 輸出層激活函數 ('none', 'tanh', 'sigmoid', 'relu')
        :param dropout: Dropout 比率
        :param configs: 配置字典（用於兼容）
        :param kwargs: 其他參數（向後相容）
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
        
        print(f"[BiLSTM] 初始化 (時間序列模式)")
        print(f"  - 輸入維度: {input_dim} (特徵數)")
        print(f"  - 輸出維度: {output_dim}")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - LSTM 層數: {n_layers}")
        print(f"  - 雙向: ✓")
        print(f"  - Dropout: {dropout}")
        print(f"  - 輸出激活: {output_activation}")
        print(f"  - 時間序列模式: ✓ (保持時間維度)\n")
        
        # 雙向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # BiLSTM 輸出維度是 hidden_dim * 2（前向 + 反向）
        lstm_output_dim = hidden_dim * 2
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 輸出層激活函數
        if output_activation == 'tanh':
            self.activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif output_activation == 'relu':
            self.activation = nn.ReLU()
        else:  # 'none'
            self.activation = None
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        # LSTM 權重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # FC 層權重初始化
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入張量
                - 2D: (batch_size, input_dim) -> 轉為 (batch_size, 1, input_dim)
                - 3D: (batch_size, seq_len, input_dim) -> 直接使用
        :return: 輸出張量 (batch_size, output_dim)
        """
        # 處理 2D 輸入（添加序列維度）
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM 前向傳播
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        
        # 使用最後時間步的輸出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # 通過全連接層
        output = self.fc(last_output)  # (batch_size, output_dim)
        
        # 應用輸出激活函數
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_config(self) -> dict:
        """返回配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_activation': self.output_activation,
            'model_type': 'bilstm'
        }


class BiLSTMActor(nn.Module):
    """BiLSTM Actor Network - 時間序列模式，用於 A2C 演算法"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.2,
                 configs: dict = None, **kwargs):
        """
        初始化 BiLSTM Actor
        
        :param input_dim: 輸入維度（特徵數）
        :param output_dim: 輸出維度 = num_stocks * 3
        :param hidden_dim: LSTM 隱藏層維度
        :param n_layers: LSTM 層數
        :param dropout: Dropout 比率
        :param configs: 配置字典（向後相容）
        :param kwargs: 其他參數（向後相容）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        print(f"[BiLSTMActor] 初始化 (時間序列模式)")
        print(f"  - 輸入維度: {input_dim} (特徵數)")
        print(f"  - 輸出維度: {output_dim}")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - LSTM 層數: {n_layers}")
        print(f"  - Dropout: {dropout}\n")
        
        # 雙向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # 動作輸出層
        self.action_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化權重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        for module in self.action_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
        :return: 動作 logits (batch_size, output_dim)
        """
        # 確保是 3D 張量
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # LSTM 前向傳播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # 使用最後時間步
        last_output = lstm_out[:, -1, :]
        
        # 動作輸出
        action_logits = self.action_head(last_output)
        return action_logits
    
    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'model_type': 'bilstm_actor'
        }


class BiLSTMCritic(nn.Module):
    """BiLSTM Critic Network - 時間序列模式，用於 A2C 演算法"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2, configs: dict = None,
                 **kwargs):
        """
        初始化 BiLSTM Critic
        
        :param input_dim: 輸入維度（狀態維度）
        :param hidden_dim: LSTM 隱藏層維度
        :param n_layers: LSTM 層數
        :param dropout: Dropout 比率
        :param configs: 配置字典（向後相容）
        :param kwargs: 其他參數（向後相容）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        print(f"[BiLSTMCritic] 初始化 (時間序列模式)")
        print(f"  - 輸入維度: {input_dim} (特徵數)")
        print(f"  - 隱藏維度: {hidden_dim}")
        print(f"  - 輸出維度: 1 (狀態價值)")
        print(f"  - LSTM 層數: {n_layers}")
        print(f"  - Dropout: {dropout}\n")
        
        # 雙向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # 價值輸出層
        self.value_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化權重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        :param x: 輸入狀態 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
        :return: 狀態價值 (batch_size, 1)
        """
        # 確保是 3D 張量
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # LSTM 前向傳播
        lstm_out, _ = self.lstm(x)
        
        # 使用最後時間步
        last_output = lstm_out[:, -1, :]
        
        # 價值輸出
        value = self.value_head(last_output)
        return value
    
    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'model_type': 'bilstm_critic'
        }