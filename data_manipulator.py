import pandas as pd
df = pd.read_csv('stock_details_5_years.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df = df.sort_values(by=['Company', 'Date'])

new_order = ['Company', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
df = df[new_order]
df['Return'] = df.groupby('Company')['Close'].pct_change()
df.to_csv('Return_500.csv', index=False)