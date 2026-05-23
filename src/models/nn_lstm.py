"""
LSTM Time-Sequence Model with Focal Loss — PyTorch Implementation.
Matches notebook 06_MVS_XAI_Ultimate_IEEE_CIS.ipynb (v4.3.4).

Architecture: LSTM(in_f, 64, num_layers=2, dropout=0.3)
              → Linear(64,32) → ReLU → Dropout(0.3) → Linear(32,1)
Loss: Focal Loss (γ=2.0, α=0.75)
Training: Full epochs + best-state tracking (NO early stopping).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ZeroSequenceDataset(Dataset):
    """PyTorch dataset for lazy all-zero sequence tensors."""

    def __init__(self, n_samples, seq_len, n_features, y=None):
        self.n_samples = int(n_samples)
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.y = y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.zeros((self.seq_len, self.n_features), dtype=torch.float32)
        if self.y is None:
            return x, torch.tensor(0.0, dtype=torch.float32)
        return x, torch.tensor(float(self.y[idx]), dtype=torch.float32)


def make_sequence_dataset(X_seq, y=None):
    if getattr(X_seq, "is_zero_sequence", False):
        return ZeroSequenceDataset(
            n_samples=X_seq.shape[0],
            seq_len=X_seq.shape[1],
            n_features=X_seq.shape[2],
            y=y,
        )

    X_tensor = torch.as_tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.zeros(len(X_seq), dtype=torch.float32) if y is None else torch.as_tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def iter_sequence_batches(X_seq, y=None, batch_size=4096, shuffle=False, seed=42):
    """Yield dense sequence batches without requiring a full dense tensor.

    Indexed/lazy sequence arrays are materialized one batch at a time. This is
    the main fast path for PaySim-scale runs where (N, T, F) would require many
    GB of RAM.
    """
    n_samples = len(X_seq)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        if getattr(X_seq, "is_zero_sequence", False):
            X_batch = np.zeros((len(batch_idx), X_seq.shape[1], X_seq.shape[2]), dtype=np.float32)
        elif getattr(X_seq, "is_indexed_sequence", False):
            X_batch = X_seq.get_batch(batch_idx)
        else:
            X_batch = X_seq[batch_idx]

        if y is None:
            y_batch = np.zeros(len(batch_idx), dtype=np.float32)
        else:
            y_batch = np.asarray(y, dtype=np.float32)[batch_idx]
        yield torch.as_tensor(X_batch, dtype=torch.float32), torch.as_tensor(y_batch, dtype=torch.float32)


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
                    epochs=12, lr=0.002, batch_size=4096, seed=42):
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = LSTMFraud(n_features).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_ep = 0
    best_state = None

    for ep in range(epochs):
        # --- Training ---
        model.train()
        for X_b, y_b in iter_sequence_batches(
            X_seq_trn, y_trn, batch_size=batch_size, shuffle=True, seed=seed + ep
        ):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # --- Best-state tracking (NO early stopping) ---
        model.eval()
        with torch.no_grad():
            val_loss = 0
            n_val_batches = 0
            for X_b, y_b in iter_sequence_batches(X_seq_val, y_val, batch_size=batch_size, shuffle=False):
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b.unsqueeze(1)).item()
                n_val_batches += 1
            val_loss /= max(n_val_batches, 1)

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
    preds = []
    with torch.no_grad():
        for X_b, _ in iter_sequence_batches(X_seq_val, batch_size=batch_size, shuffle=False):
            preds.append(torch.sigmoid(model(X_b.to(device))).cpu().numpy())

    return np.concatenate(preds).flatten(), model
