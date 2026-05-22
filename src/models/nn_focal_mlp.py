"""
Behavioral MLP with Focal Loss — PyTorch Implementation.
Matches notebook 06_MVS_XAI_Ultimate_IEEE_CIS.ipynb (v4.3.4).

Architecture: Linear(256)→ReLU→Dropout(0.3)→Linear(128)→ReLU→Dropout(0.3)
              →Linear(64)→ReLU→Dropout(0.2)→Linear(1)
Loss: Focal Loss (γ=2.0, α=0.75)
Training: Full epochs + best-state tracking (NO early stopping).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Focal Loss as defined in proposal_revised.tex Eq. (2):
    L_FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    γ=2 penalizes easily classified legitimate transactions.
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


class BehavioralMLP(nn.Module):
    """Multi-Layer Perceptron for behavioral feature view.
    3-layer funnel architecture with dropout for regularization.
    """
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_focal(X_trn, y_trn, X_val, y_val, device, epochs=15, lr=0.001, batch_size=4096, seed=42):
    """Train Behavioral MLP with Focal Loss.

    Strategy (v4.3.2): Train ALL epochs, restore best-epoch model.
    Early stopping was removed because it caused AUC~0.50 on imbalanced data.

    Args:
        X_trn: Training features (numpy array).
        y_trn: Training labels (numpy array).
        X_val: Validation features (numpy array).
        y_val: Validation labels (numpy array).
        device: torch.device ('cuda' or 'cpu').
        epochs: Number of training epochs (default: 15).
        lr: Learning rate (default: 0.001).
        batch_size: Batch size (default: 4096).

    Returns:
        Tuple of (validation predictions, trained model).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    n_features = X_trn.shape[1]
    model = BehavioralMLP(n_features).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.as_tensor(X_trn, dtype=torch.float32),
        torch.as_tensor(y_trn, dtype=torch.float32)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    val_t = torch.as_tensor(X_val, dtype=torch.float32)
    val_y_t = torch.as_tensor(y_val, dtype=torch.float32)

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
    print(f"    [MLP] Best epoch: {best_ep}/{epochs} (val_loss={best_val_loss:.4f})")

    # --- Predict validation set ---
    model.eval()
    val_ds = TensorDataset(val_t, torch.zeros(len(val_t)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            preds.append(torch.sigmoid(model(X_b.to(device))).cpu().numpy())

    return np.concatenate(preds).flatten(), model
