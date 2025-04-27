import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from DeepRiskModel import DeepRiskModel

P, hidden_dim, K, horizon = 5, 64, 5, 20
model_path = 'drm_model.pth'
feat_csv = 'beta_factors.csv'
ret_csv = 'returns.csv'

device = 'cuda'
model = DeepRiskModel(P, hidden_dim, K).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

df_feat = pd.read_csv(feat_csv, parse_dates=['Date'])
df_ret = pd.read_csv(ret_csv,  parse_dates=['Date'])
comps = sorted(df_feat['Company'].unique())
feat_cols = [c for c in df_feat.columns if c not in ('Date','Company')]

last_date  = df_feat['Date'].max()
start_date = last_date - pd.Timedelta(days=30)
mask       = (df_feat['Date'] >= start_date) & (df_feat['Date'] <= last_date)
dates_test = sorted(df_feat.loc[mask, 'Date'].unique())

print(f"Plotting {len(dates_test)} days from {dates_test[0].date()} to {dates_test[-1].date()}")

N  = len(comps)
adj = torch.ones(N, N)
edge_index, _ = dense_to_sparse(adj)

def compute_r2(F, y, eps=1e-6):
    FtF  = F.T @ F + eps * np.eye(F.shape[1])
    beta = np.linalg.solve(FtF, F.T @ y)
    yhat = F @ beta
    return 1 - np.mean((y - yhat)**2) / np.var(y)

r2_raw, r2_drm = [], []
betas = []
ics = []

for d in dates_test:
    X = ( df_feat[df_feat['Date']==d]
          .set_index('Company').reindex(comps)[feat_cols]
          .values )
    y = ( df_ret[df_ret['Date']==d]
          .set_index('Company')['Return']
          .reindex(comps).fillna(0).values )
    
    r2_raw.append(compute_r2(X, y))
    Xt = torch.from_numpy(X).float().to(device)
    F  = model(Xt.unsqueeze(1), edge_index.to(device)).cpu().detach().numpy()
    r2_drm.append(compute_r2(F, -y))

    β_d = np.linalg.solve(
        F.T @ F + 1e-6 * np.eye(K),
        F.T @ y
    )
    betas.append(β_d)
    
    ic_d = [ np.corrcoef(F[:, k], y)[0,1] for k in range(K) ]
    ics.append(ic_d)

plt.figure(figsize=(10,4))
plt.plot(dates_test, r2_raw, '-o', label='Raw Factors R²')
plt.plot(dates_test, r2_drm, '-o', label='DRM Factors R²')
plt.title('Daily Explained Variance (R²)')
plt.xlabel('Date'); plt.ylabel('R²')
plt.xticks(rotation=45); plt.grid(True)
plt.legend(); plt.tight_layout()
plt.show()

betas = np.stack(betas, axis=0) 
cum_betas = np.cumsum(betas, axis=0) 

plt.figure(figsize=(10,6))
for k in range(K):
    plt.plot(
        dates_test,
        cum_betas[:, k],
        label=f'Factor {k+1}'
    )

plt.title('Cumulative DRM Factor Returns (Regression β)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc='upper left', ncol=2)
plt.tight_layout()
plt.show()

ics     = np.stack(ics, axis=0)
ic_cols = [f'Factor {i+1} IC' for i in range(K)]

df_daily_ic = pd.DataFrame(ics, index=dates_test, columns=ic_cols)
print("\nDaily Information Coefficients:")
print(df_daily_ic)

mean_ic = df_daily_ic.mean()
ir_ic   = mean_ic / df_daily_ic.std()
df_summary_ic = pd.DataFrame({
    'Mean IC': mean_ic,
    'IR': ir_ic
})
print("\nInformation Coefficient Summary:")
print(df_summary_ic)

plt.figure(figsize=(10,4))
for col in df_daily_ic.columns:
    plt.plot(df_daily_ic.index, df_daily_ic[col], '-o', label=col)

plt.axhline(0, linestyle='--')
plt.title('Daily Information Coefficient (IC)')
plt.xlabel('Date')
plt.ylabel('IC')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()