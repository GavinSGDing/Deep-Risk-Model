import pandas as pd
df = pd.read_csv('sp500.csv') # r"absolute path"
df['Date'] = pd.to_datetime(df['Date'], utc=True)

df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

df = df[['Company', 'Date', 'Return']]

def assign_feature(group):
    group = group.sort_values('Return')
    try:
        group['Feature'] = pd.qcut(group['Return'], q=[0, 0.3, 0.7, 1], labels=["Lo 30", "Med 40", "Hi 30"])
    except ValueError:
        group['Feature'] = "Undefined"
    return group

df = df.groupby('Date').apply(assign_feature).reset_index(drop=True)
df = df.sort_values(by=['Company', 'Date'])
df.to_csv('final_sp500.csv', index=False) # r"absolute path"