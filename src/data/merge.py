import pandas as pd

# Load datasets
df1 = pd.read_csv("src/data/data/raw/heart.csv")
df2 = pd.read_csv("src/data/data/raw/processed.cleveland_labeled.csv")
df3 = pd.read_csv("src/data/data/raw/processed.hungarian_labeled.csv")
df4 = pd.read_csv("src/data/data/raw/processed.switzerland_labeled.csv")
df5 = pd.read_csv("src/data/data/raw/processed.va_labeled.csv")

# Merge on common column
merged = pd.merge(df1, df2, df3, df4, df4 on="id", how="inner")

# Save merged dataset
merged.to_csv("merged_dataset.csv", index=False)

print(merged.head())
