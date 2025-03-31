import pandas as pd
import statsmodels.api as sm

df_stocks = pd.read_csv(r'\Deep-Risk-Model\S&P 500 data\final_sp500.csv')
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])
df_stocks = df_stocks[['Company', 'Date', 'Return']]

df_mom = pd.read_csv(r'\Deep-Risk-Model\Filtered_F&F\MOM.csv')
df_mom['Date'] = pd.to_datetime(df_mom['Date'])

df_merged = pd.merge(df_stocks, df_mom, on='Date', how='left')

def rolling_ols(sub_df, window):
    sub_df = sub_df.sort_values('Date').set_index('Date')
    
    betas = []
    idx = []
    
    for end_idx in range(window, len(sub_df)):
        window_slice = sub_df.iloc[end_idx - window : end_idx]
        
        y = window_slice['Return']
        X = window_slice[['Mom']]
        X = sm.add_constant(X)  
        
        model = sm.OLS(y, X).fit()
        betas.append(model.params)
        idx.append(sub_df.index[end_idx])
    
    betas_df = pd.DataFrame(betas, index=idx)
    return betas_df

df_betas = df_merged.groupby('Company').apply(rolling_ols, window=120)

df_betas_reset = df_betas.reset_index()
df_betas_reset.to_csv('daily_beta_values.csv', index=False)