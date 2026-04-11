import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Hàm Loss triệt tiêu sự mờ nhạt của Data Imbalance cực độ.
    Alpha: Cân bằng trọng số tỉ lệ Fraud so với Non-Fraud.
    Gamma: Phạt gắt các Normal Transaction giả dạng Fraud (Giảm false positives).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

class LSTMSequenceDetector(nn.Module):
    """
    Nhận chuỗi Data Sequence từ PaySim (T=10) để phân tích hành vi dòng tiền rò rỉ.
    Cấu trúc: LSTM(Layer=2) -> Fully connected (32) -> LoGits
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Shape đầu vào chuẩn nén: (Batch_size, Sequence_len=10, Channels=All_Features)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Chỉ tập trung bóc tách trọng tâm của điểm chốt chuỗi thời gian cuối (Bước số 10)
        last_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out) # Đưa ra nhãn thô (Chưa đi qua Sigmoid vì dùng BCEWithLogits)
        return logits.squeeze(-1)
