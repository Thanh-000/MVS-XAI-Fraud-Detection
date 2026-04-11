"""
Mạng Nơ-ron Chuỗi (Sequential LSTM Neural Network)
Sử dụng PyTorch để nhúng Dữ liệu Cửa sổ Thời gian (Sliding Window)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMFraud(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.lstm = nn.LSTM(in_features, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze(-1)

class SequentialTrainer:
    def __init__(self, in_features, seq_len=5):
        self.seq_len = seq_len
        self.model = LSTMFraud(in_features)
        
    def build_cube(self, X):
        # Trượt ma trận thành Khung thời gian (Cube 3D cho LSTM)
        X_val = X.values if hasattr(X, 'values') else X
        X_pad = np.pad(X_val, ((self.seq_len-1, 0), (0, 0)), mode='constant')
        strides = (X_pad.strides[0], X_pad.strides[0], X_pad.strides[1])
        return np.lib.stride_tricks.as_strided(X_pad, shape=(len(X), self.seq_len, X.shape[1]), strides=strides)

    def train_and_predict_oof(self, X_train, y_train):
        X_cube = self.build_cube(X_train)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        loader = DataLoader(TensorDataset(torch.tensor(X_cube, dtype=torch.float32), 
                            torch.tensor(y_train, dtype=torch.float32)), batch_size=4096, shuffle=True)
        
        print("[LSTM] Kích hoạt Mạng Nơ-ron Chuỗi không lỗi (Epochs)...")
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(self.model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            oof = torch.sigmoid(self.model(torch.tensor(X_cube, dtype=torch.float32))).numpy()
        print("✅ Đã đúc kết Quyết định (OOF) của Khối Mạng Nơ-ron/PyTorch.")
        return oof
        
    def predict(self, X_test):
        X_cube = self.build_cube(X_test)
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(X_cube, dtype=torch.float32))).numpy()
