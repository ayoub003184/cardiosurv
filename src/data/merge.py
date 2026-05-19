import pandas as pd

# Load datasets
df1 = pd.read_csv("src/data/data/raw/heart.csv")
df2 = pd.read_csv("dataset2.csv")

# Merge on common column
merged = pd.merge(df1, df2, on="id", how="inner")

# Save merged dataset
merged.to_csv("merged_dataset.csv", index=False)

print(merged.head())
