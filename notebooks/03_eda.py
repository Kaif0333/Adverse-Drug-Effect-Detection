import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===============================
# PATH SETUP (IMPORTANT)
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load cleaned data
df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

# ===============================
# BASIC INFO
# ===============================
print("Dataset shape:", df.shape)
print("Adverse count:", df['label'].sum())

# ===============================
# 1. Class Distribution
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Safe vs Adverse Drug Reviews")
plt.show()

# ===============================
# 2. Top Adverse Drugs
# ===============================
top_drugs = df[df['label'] == 1]['drugName'].value_counts().head(10)

plt.figure(figsize=(8,5))
top_drugs.plot(kind='bar')
plt.title("Top 10 Drugs with Adverse Effects")
plt.xlabel("Drug Name")
plt.ylabel("Count")
plt.show()

# ===============================
# 3. Rating vs Adverse Effect
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x='label', y='rating', data=df)
plt.title("Rating vs Adverse Effects")
plt.show()

# ===============================
# 4. Review Length Analysis
# ===============================
df['review_length'] = df['review'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(6,4))
sns.boxplot(x='label', y='review_length', data=df)
plt.title("Review Length vs Adverse Effects")
plt.show()
