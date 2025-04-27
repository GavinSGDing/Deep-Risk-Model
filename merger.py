import pandas as pd

# 1) load your merged CSV
df = pd.read_csv("merged_factors.csv", parse_dates=["Date"])

# 2) specify the exact order you want
beta_cols = ["BE-ME-beta", "INV-beta", "ME-beta", "Mom-beta", "OP-beta"]
new_cols  = ["Date", "Company"] + beta_cols

# 3) reorder the columns
df = df[new_cols]

# 4) sort the rows by Date then Company
df = df.sort_values(by=["Date", "Company"]).reset_index(drop=True)

# 5) (optional) inspect
print(df.head())

# 6) save back out if you like
df.to_csv("merged_ordered.csv", index=False)