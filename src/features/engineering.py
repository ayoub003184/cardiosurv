import pandas as pd

# 1. Age Bining
def age_bin(age: int) -> str:
  if age < 40: return "<40"
  if age < 50: return "40-49"
  if age < 60: return "50-59"
  if age < 70: return "60-69"
  return "70+"

# 2. Blood Pressure Risk Level
def bp_risk_level(resting_bp: int) -> str:
  if resting_bp < 120: return "Normal"
  if resting_bp < 130: return "Elevated"
  if resting_bp < 140: return "Stage1"
  if resting_bp < 180: return "Stage2"
  return "Crisis"

# 3. Heart Rate Stress Index
def hr_stress_index(max_hr: int, age: int) -> float:
  return round(max_hr / (220 - age), 3)

#4. Final Risk Label
def derive_risk_label(row) -> str:
  # 1. HeartDisease = 0 -> Low Risk
  if row['HeartDisease'] == 0:
    return "Low"
    
  # If HeartDisease = 1, check the severe conditions
  elif row['HeartDisease'] == 1:
    has_high_oldpeak = row['Oldpeak'] >= 2.0
    has_angina = row['ExerciseAngina'] == 'Y'
    has_severe_bp = row['bp_risk_level'] in ['Stage2', 'Crisis']
    
    # 3. Any of the severe conditions are met -> High Risk
    if has_high_oldpeak or has_angina or has_severe_bp:
      return "High"
      
    # 2. None of the severe conditions are met -> Medium Risk
    else:
      return "Medium"
      
  return "Low"

# 5. The Main Engineering Pipeline
def engineer(df):
  df['age_bin'] = df['age'].apply(age_bin)
  df['bp_risk_level'] = df['resting_bp'].apply(bp_risk_level)
  df['hr_stress_index'] = df.apply(lamda row: hr_stress_index(row['max_hr'], row['age']), axis=1)
  df['RiskCategory'] = df.apply(derive_risk_label, axis=1)

  return df

# 6. Execution
def main():
  # Load the cleaned data
  df = pd.read_csv('data/processed/cleaned.csv')
  print(f"[features] Input shape: {df.shape}")

  # Run the pipeline to add the 4 new columns
  df_engineered = engineer(df)

  # Print out the required console logs
  print(f"[features] Output shape: {df_engineered.shape}")
  print("[features] Risk label distribution:")

  # Safely get counts, defaulting to 0 if a category is missing
  counts = df_engineered['RiskCategory'].value_counts()
  print(f"Low:\t{counts.get('Low', 0)}")
  print(f"Medium:\t{counts.get('Medium', 0)}")
  print(f"High:\t{counts.get('High', 0)}")

  # Save the new CSV file
  df_engineered.to_csv('data/processed/features.csv', index=False)
  print("[features] Wrote data/processed/features.csv")

if __name__ == "__main__":
  main()
    
