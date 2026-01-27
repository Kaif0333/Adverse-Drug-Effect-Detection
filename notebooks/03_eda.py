import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------
# STEP 1: Paths
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -----------------------------------
# STEP 2: Load cleaned dataset
# -----------------------------------
df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------------
# STEP 3: Class Distribution (Safe vs Adverse)
# -----------------------------------
label_counts = df['label'].value_counts()

plt.figure()
label_counts.plot(kind='bar')
plt.title("Safe vs Adverse Drug Experiences")
plt.xlabel("Label (0 = Safe, 1 = Adverse)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------------
# STEP 4: Rating Distribution
# -----------------------------------
plt.figure()
df['rating'].hist()
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -----------------------------------
# STEP 5: Top 10 Drugs with Adverse Effects
# -----------------------------------
risky_drugs = df[df['label'] == 1]['drugName'].value_counts().head(10)

plt.figure()
risky_drugs.plot(kind='bar')
plt.title("Top 10 Drugs with Adverse Effects")
plt.xlabel("Drug Name")
plt.ylabel("Number of Adverse Reviews")
plt.tight_layout()
plt.show()
