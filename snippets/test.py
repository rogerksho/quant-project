import pandas as pd

df = pd.read_csv("data/sp500_cleaned.txt", index_col=0, parse_dates=True, usecols=[0,1,2,3,4])

print(df.head)