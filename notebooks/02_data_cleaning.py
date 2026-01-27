import pandas as pd
from pathlib import Path

# -------------------------------
# STEP 1: Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------
# STEP 2: Load data
# -------------------------------
df = pd.read_csv(DATA_DIR / "drugsComTrain_raw.csv")
print("Original shape:", df.shape)

# -------------------------------
# STEP 3: Select useful columns
# -------------------------------
df = df[['drugName', 'condition', 'review', 'rating']]
print("After column selection:", df.shape)

# -------------------------------
# STEP 4: Handle missing values
# -------------------------------
print(df.isnull().sum())
df.dropna(inplace=True)
print("After dropping missing values:", df.shape)

# -------------------------------
# STEP 5: Create target label
# -------------------------------
def label_rating(rating):
    if rating >= 7:
        return 0   # Safe
    elif rating <= 4:
        return 1   # Adverse
    else:
        return None

df['label'] = df['rating'].apply(label_rating)

# Remove neutral ratings (5 & 6)
df.dropna(subset=['label'], inplace=True)

print(df['label'].value_counts())

# -------------------------------
# STEP 6: Save cleaned dataset
# -------------------------------
clean_path = DATA_DIR / "clean_drug_data.csv"
df.to_csv(clean_path, index=False)

print("Cleaned dataset saved successfully!")
