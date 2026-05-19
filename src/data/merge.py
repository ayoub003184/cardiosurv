import pandas as pd

# Load datasets
df1 = pd.read_csv("src/data/data/raw/heart.csv")
df2 = pd.read_csv("src/data/data/raw/processed.cleveland_labeled.csv")
df3 = pd.read_csv("src/data/data/raw/processed.hungarian_labeled.csv")
df4 = pd.read_csv("src/data/data/raw/processed.switzerland_labeled.csv")
df5 = pd.read_csv("src/data/data/raw/processed.va_labeled.csv")

# Combine datasets
merged = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Save output
merged.to_csv("data/processed/merged.csv", index=False)

print(merged.head())
print(merged.shape)
