import pandas as pd
input_file = 'F-F_Momentum_Factor_daily.csv' # INV, ME, OP
output_file = 'MOM.csv' #INV, ME, OP

df = pd.read_csv(input_file)

df['Date'] = pd.to_datetime(
    df['Date'].astype(str).str.strip(), 
    format='%Y%m%d', 
    errors='coerce'
)
#df = df[['Date', 'Lo 30', 'Med 40', 'Hi 30']]

start_date = pd.to_datetime('20181129')
end_date   = pd.to_datetime('20231129')

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

df.to_csv(output_file, index=False)