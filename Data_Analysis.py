import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r'\Deep-Risk-Model\Beta_val\OP-beta.csv')
df['Date'] = pd.to_datetime(df['Date'])

df = df[['Date', 'OP-beta', 'Company']]

group_stats = df.groupby('Company')['OP-beta'].describe()

selected_company = "GOOGL"
company_data = df[df['Company'] == selected_company]

profile = ProfileReport(
    company_data,
    tsmode=True,
    sortby="Date",
    title="Time-Series EDA for Company GOOGL (OP-beta)"
)

profile.to_file(f"{selected_company}_OP_beta_profile.html")