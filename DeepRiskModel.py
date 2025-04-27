import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import numpy as np

class DeepRiskDataset(Dataset):
    """
    Dataset for deep risk model.
    - features: Tensor of shape [T, N, P]
    - returns:  Tensor of shape [T, N]
    - horizon: number of forward days for prediction
    """
    def __init__(self, features: torch.Tensor, returns: torch.Tensor, horizon: 20):
        assert features.dim() == 3  # T x N x P
        assert returns.dim() == 2   # T x N
        self.X = features
        self.Y = returns
        self.h = horizon

    def __len__(self):
        return self.X.size(0) - self.h

    def __getitem__(self, idx: int):
        x_t = self.X[idx]  # [N, P]
        y_fwd = self.Y[idx + 1 : idx + 1 + self.h]
        return x_t, y_fwd

class DeepRiskModel(nn.Module):
    def __init__(self, P: int, hidden_dim: int, K: int, gat_heads: int = 1):
        super().__init__()
        self.gru_temp = nn.GRU(input_size=P, hidden_size=hidden_dim,
                               num_layers=2, batch_first=True)
        self.gat = GATConv(in_channels=P, out_channels=P // 2,
                           heads=gat_heads, concat=False)
        self.gru_cs = nn.GRU(input_size=P, hidden_size=hidden_dim,
                             num_layers=2, batch_first=True)
        self.K1 = K // 2
        self.K2 = K - self.K1
        self.proj_temp = nn.Linear(hidden_dim, self.K1)
        self.proj_cs   = nn.Linear(hidden_dim, self.K2)
        self.bn1 = nn.BatchNorm1d(self.K1)
        self.bn2 = nn.BatchNorm1d(self.K2)

    def forward(self, X_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # X_seq: [N, T, P] where T=sequence length (often 1)
        out_t, _ = self.gru_temp(X_seq)
        h_t = out_t[:, -1, :]  # [N, hidden]
        # Cross-sectional path
        X_t = X_seq[:, -1, :]  # [N, P]
        gat_out = self.gat(X_t, edge_index)  # [N, P//2]
        # pad back to P dims
        pad = torch.zeros(X_t.size(0), X_t.size(1) - gat_out.size(1), device=X_t.device)
        residual = X_t - torch.cat([gat_out, pad], dim=1)
        out_cs, _ = self.gru_cs(residual.unsqueeze(1))  # [N, 1, hidden]
        h_cs = out_cs[:, -1, :]  # [N, hidden]
        # Project & normalize
        f1 = self.bn1(self.proj_temp(h_t))  # [N, K1]
        f2 = self.bn2(self.proj_cs(h_cs))    # [N, K2]
        return torch.cat([f1, f2], dim=1)    # [N, K]

def r2_loss(F: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    FtF = F.T @ F + eps * torch.eye(F.shape[1], device=F.device)
    beta = torch.linalg.solve(FtF, F.T @ y) 
    y_hat = F @ beta
    mse = torch.mean((y - y_hat)**2)
    var = torch.mean((y - y.mean())**2).clamp_min(eps)
    return mse / var


def collin_reg(F, eps=1e-6):
    FtF = F.T @ F + eps * torch.eye(F.shape[1], device=F.device)
    inv = torch.inverse(FtF)
    return torch.trace(inv) / F.shape[1]

def train_epoch(model, loader, edge_index, optimizer, lambda_reg, device):
    model.train()
    running_loss = 0.0
    for X_t, y_fwd in loader:
        # X_t: [1, N, P], y_fwd: [1, horizon, N]
        X_np = X_t.squeeze(0).to(device)       # [N, P]
        X_seq = X_np.unsqueeze(1)              # [N, 1, P]
        y = y_fwd.squeeze(0).to(device)        # [horizon, N]
        # Forward
        F_t = model(X_seq, edge_index.to(device))  # [N, K]
        # Multi-horizon R2 + collinearity regularization
        r2_vals = [r2_loss(F_t, y[i]) for i in range(y.size(0))]
        loss = torch.stack(r2_vals).mean() + lambda_reg * collin_reg(F_t)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, edge_index, lambda_reg, device):
    model.eval()
    running_loss = 0.0
    for X_t, y_fwd in loader:
        X_np = X_t.squeeze(0).to(device)
        X_seq = X_np.unsqueeze(1)
        y = y_fwd.squeeze(0).to(device)
        F_t = model(X_seq, edge_index.to(device))
        r2_vals = [r2_loss(F_t, y[i]) for i in range(y.size(0))]
        running_loss += (torch.stack(r2_vals).mean() + lambda_reg * collin_reg(F_t)).item()
    return running_loss / len(loader)

if __name__ == '__main__':
    epochs = 50
    horizon = 20
    hidden_dim = 64
    K = 5 
    lr = 1e-4
    lambda_reg = 1e-3
    batch_size = 1
    device = 'cuda'

    df = pd.read_csv('beta_factors.csv', parse_dates=['Date'])
    dates = sorted(df['Date'].unique())
    comps = sorted(df['Company'].unique())
    feat_cols = [c for c in df.columns if c not in ['Date','Company']]
    T, N, P = len(dates), len(comps), len(feat_cols)
    feats = np.zeros((T, N, P), np.float32)
    for i, d in enumerate(dates):
        tmp = df[df['Date']==d].set_index('Company')
        feats[i] = tmp.reindex(comps)[feat_cols].values
    
    df_r = pd.read_csv('returns.csv', parse_dates=['Date'])
    df_r = df_r.dropna(subset=['Return'])
    R = np.zeros((T, N), np.float32)
    for i, d in enumerate(dates):
        tmp = df_r[df_r['Date']==d].set_index('Company')
        R[i] = tmp.reindex(comps)['Return'].fillna(1e-6).values

    valid_mask = ~np.isnan(feats).any(axis=(1,2))
    feats = feats[valid_mask]
    R = R[valid_mask]

    full_ds = DeepRiskDataset(torch.from_numpy(feats), torch.from_numpy(R), horizon)
    total_samples = len(full_ds)

    i1 = int(0.7 * total_samples)
    i2 = int(0.85 * total_samples)
    train_ds = Subset(full_ds, range(0, i1))
    val_ds   = Subset(full_ds, range(i1, i2))
    test_ds  = Subset(full_ds, range(i2, total_samples))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # Prepare graph for GAT (fully-connected)
    adj = torch.ones(N, N)
    edge_index, _ = dense_to_sparse(adj)

    # Instantiate model & optimizer
    model = DeepRiskModel(P, hidden_dim, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, edge_index, optimizer, lambda_reg, device)
        val_loss   = eval_epoch(model,   val_loader,   edge_index, lambda_reg, device)
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        
    test_loss = eval_epoch(model, test_loader, edge_index, lambda_reg, device)
    print(f"Final Test Loss: {test_loss:.3f}")
    
    torch.save(model.state_dict(), 'drm_model.pth')
    print("Model weights saved to drm_model.pth")