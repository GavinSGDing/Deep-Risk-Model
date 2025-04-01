import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

df_stocks = pd.read_csv(r'\Deep-Risk-Model\Big 500 data\Big_500.csv')
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])

df_factors = pd.read_csv(r'\Desktop\Deep-Risk-Model\Filtered_F&F\BE-ME.csv')
df_factors['Date'] = pd.to_datetime(df_factors['Date'])

df_merged = pd.merge(df_stocks, df_factors, on='Date', how='left')

valid_features = ['Lo 30', 'Med 40', 'Hi 30']

def get_risk_factor_value(row):
    feature = row['Feature']
    if feature in valid_features:
        return row[feature]
    else:
        return None

df_merged['RiskFactor'] = df_merged.apply(get_risk_factor_value, axis=1)

df_merged = df_merged.dropna(subset=['Return', 'RiskFactor'])

def rolling_ols(sub_df, window):
    sub_df = sub_df.sort_values('Date').set_index('Date')
    
    betas = []
    idx = []

    for end_idx in tqdm(range(window, len(sub_df)), desc="Rolling OLS", leave=False):
        window_slice = sub_df.iloc[end_idx - window: end_idx]
        
        y = window_slice['Return']
        X = window_slice['RiskFactor']
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        betas.append(model.params)
        idx.append(sub_df.index[end_idx])
    
    return pd.DataFrame(betas, index=idx)

results = []
unique_companies = df_merged['Company'].nunique()
for company, group in tqdm(df_merged.groupby('Company'), total=unique_companies, desc="Companies"):
    beta_df = rolling_ols(group, window=120)
    beta_df['Company'] = company
    results.append(beta_df)

df_betas = pd.concat(results)
df_betas_reset = df_betas.reset_index().rename(columns={'index': 'Date'})

df_betas_reset.to_csv('BE-ME-beta.csv', index=False)