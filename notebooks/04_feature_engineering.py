import pandas as pd
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -----------------------------------
# STEP 1: Paths
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# STEP 2: Load cleaned dataset
# -----------------------------------
df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

print("Dataset shape:", df.shape)

# -----------------------------------
# STEP 3: Basic text cleaning function
# -----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_review"] = df["review"].apply(clean_text)

print("Sample cleaned review:")
print(df["clean_review"].iloc[0])

# -----------------------------------
# STEP 4: Features and Target
# -----------------------------------
X = df["clean_review"]
y = df["label"]

# -----------------------------------
# STEP 5: TF-IDF Vectorization
# -----------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(X)

print("TF-IDF feature shape:", X_tfidf.shape)

# -----------------------------------
# STEP 6: Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------------
# STEP 7: Save processed data
# -----------------------------------
import joblib

joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump((X_train, X_test, y_train, y_test), MODELS_DIR / "processed_data.pkl")

print("Feature engineering completed and saved!")
