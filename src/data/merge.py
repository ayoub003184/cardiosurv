import pandas as pd

# Load datasets
df1 = pd.read_csv("src/data/data/raw/heart.csv")
df2 = pd.read_csv("src/data/data/raw/processed.cleveland_labeled.csv")
df3 = pd.read_csv("src/data/data/raw/processed.hungarian_labeled.csv")
df4 = pd.read_csv("src/data/data/raw/processed.switzerland_labeled.csv")
df5 = pd.read_csv("src/data/data/raw/processed.va_labeled.csv")

# Rename columns in processed datasets to match heart.csv
rename_dict = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'ChestPainType',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'fbs': 'FastingBS',
    'restecg': 'RestingECG',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'oldpeak': 'Oldpeak',
    'slope': 'ST_Slope',
    'target': 'HeartDisease'
}

# Apply renaming to datasets 2-5
df2 = df2.rename(columns=rename_dict)
df3 = df3.rename(columns=rename_dict)
df4 = df4.rename(columns=rename_dict)
df5 = df5.rename(columns=rename_dict)

# Select only the common columns from heart.csv
common_cols = df1.columns
df2 = df2[common_cols]
df3 = df3[common_cols]
df4 = df4[common_cols]
df5 = df5[common_cols]

# Combine datasets
merged = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Save output
merged.to_csv("src/data/merged.csv", index=False)

print(merged.head())
print(merged.shape)
