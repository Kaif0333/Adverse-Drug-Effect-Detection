import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_features(base_dir: Path, max_features=5000):
    df = pd.read_csv(base_dir / "data" / "clean_drug_data.csv")
    df['clean_review'] = df['review'].apply(clean_text)
    X = df['clean_review']
    y = df['label']
    
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save
    joblib.dump(tfidf, base_dir / "models" / "tfidf_vectorizer.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), base_dir / "models" / "processed_data.pkl")
    
    return X_train, X_test, y_train, y_test, tfidf
