"""
LSTM Time-Sequence Model with Focal Loss — PyTorch Implementation.
Matches notebook 06_MVS_XAI_Ultimate_IEEE_CIS.ipynb (v4.3.4).

Architecture: LSTM(in_f, 64, num_layers=2, dropout=0.3)
              → Linear(64,32) → ReLU → Dropout(0.3) → Linear(32,1)
Loss: Focal Loss (γ=2.0, α=0.75)
Training: Full epochs + best-state tracking (NO early stopping).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for sequential fraud detection.
    Shared implementation with BehavioralMLP for consistency.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class LSTMFraud(nn.Module):
    """2-layer LSTM for sequential transaction pattern detection.

    Input shape: (batch, seq_len=10, n_features)
    Uses the last hidden state from the top LSTM layer.
    """
    def __init__(self, in_features):
        super().__init__()
        self.lstm = nn.LSTM(in_features, 64, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])  # Last hidden state of top layer


def train_lstm_fold(X_seq_trn, y_trn, X_seq_val, y_val, n_features, device,
                    epochs=12, lr=0.002, batch_size=4096):
    """Train LSTM with Focal Loss for one CV fold.

    Strategy (v4.3.2): Train ALL epochs, restore best-epoch model.
    Early stopping was removed because it caused AUC~0.50 on imbalanced data.

    Args:
        X_seq_trn: Training sequences (numpy, shape: [N, T, F]).
        y_trn: Training labels (numpy array).
        X_seq_val: Validation sequences (numpy, shape: [N, T, F]).
        y_val: Validation labels (numpy array).
        n_features: Number of input features per timestep.
        device: torch.device ('cuda' or 'cpu').
        epochs: Number of training epochs (default: 12).
        lr: Learning rate (default: 0.002).
        batch_size: Batch size (default: 4096).

    Returns:
        Tuple of (validation predictions, trained model).
    """
    model = LSTMFraud(n_features).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_seq_trn, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_t = torch.tensor(X_seq_val, dtype=torch.float32)
    val_y_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float('inf')
    best_ep = 0
    best_state = None

    for ep in range(epochs):
        # --- Training ---
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # --- Best-state tracking (NO early stopping) ---
        model.eval()
        with torch.no_grad():
            val_ds_tmp = TensorDataset(val_t, val_y_t)
            val_loader_tmp = DataLoader(val_ds_tmp, batch_size=batch_size, shuffle=False)
            val_loss = 0
            for X_b, y_b in val_loader_tmp:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b.unsqueeze(1)).item()
            val_loss /= len(val_loader_tmp)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ep = ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    [LSTM] Best epoch: {best_ep}/{epochs} (val_loss={best_val_loss:.4f})")

    # --- Predict validation set ---
    model.eval()
    val_ds = TensorDataset(val_t, torch.zeros(len(val_t)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            preds.append(torch.sigmoid(model(X_b.to(device))).cpu().numpy())

    return np.concatenate(preds).flatten(), model
