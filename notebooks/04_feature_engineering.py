import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

# ===============================
# LOAD CLEAN DATA
# ===============================
df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

# ===============================
# FEATURE 1: TF-IDF
# ===============================
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_text = tfidf.fit_transform(df["clean_review"])

# ===============================
# FEATURE 2: REVIEW LENGTH
# ===============================
df["review_length"] = df["clean_review"].apply(lambda x: len(x.split()))
X_length = df[["review_length"]].values

# ===============================
# COMBINE FEATURES
# ===============================
X = hstack([X_text, X_length])
y = df["label"]

# ===============================
# SAVE FEATURES & VECTORIZER
# ===============================
joblib.dump(X, MODELS_DIR / "processed_data.pkl")
joblib.dump(y, MODELS_DIR / "labels.pkl")
joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")

print("Feature engineering completed successfully")
print("Feature shape:", X.shape)
