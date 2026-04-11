import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_lstm_pipeline(model: nn.Module, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                        epochs=10, batch_size=2048, lr=1e-3, device='cuda'):
    """
    Vòng lặp đào tạo cho LSTM (Siêu tốc bằng CUDA/GPU nếu có).
    Xử lý mất cân bằng Pytorch thông qua tự động căn chỉnh pos_weight.
    """
    logger.info(f"Đang chuẩn bị TensorDataset | Epochs={epochs} | BatchSize={batch_size} | Thiết bị={device.upper()}")
    
    # 1. Ép Tensor
    X_tr = torch.tensor(X_train_seq, dtype=torch.float32)
    y_tr = torch.tensor(y_train_seq, dtype=torch.float32)
    X_va = torch.tensor(X_val_seq, dtype=torch.float32)
    y_va = torch.tensor(y_val_seq, dtype=torch.float32)
    
    # 2. Dataloader song song
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Tự động chèn trọng số Class Imbalance (Kích thích tìm kiếm Tội phạm)
    fraud_count = y_train_seq.sum()
    weight_ratio = (len(y_train_seq) - fraud_count) / max(fraud_count, 1)
    pos_weight = torch.tensor([weight_ratio], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Thanh load tiến độ (Tqdm) chuyên nghiệp
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Chặn Gradient Exploding quá mức trong time-series
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        # Kiểm tra vắng mặt (Validation không kích hoạt đồ thị đạo hàm)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
        t_loss = running_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        logger.info(f"Epoch {epoch+1:02d} => Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
    return model, train_losses, val_losses
