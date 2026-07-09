import pandas as pd

df = pd.read_csv("data/raw/train.csv")
cols = [c for c in df.columns if c != "SalePrice"]
for c in cols:
    dtype = df[c].dtype
    print(f"{c}|{dtype}")
