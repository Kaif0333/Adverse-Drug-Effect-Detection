import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load clean dataset
# ----------------------------
df = pd.read_csv("data/clean_drug_data.csv")

X = df["review"]
y = df["label"]

# ----------------------------
# Stratified Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Pipeline
# ----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# ----------------------------
# Correct Parameter Grid
# ----------------------------
param_grid = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "rf__n_estimators": [200, 300],
    "rf__max_depth": [15, 20]
}

# ----------------------------
# GridSearch with CV
# ----------------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2
)

# ----------------------------
# Train
# ----------------------------
grid.fit(X_train, y_train)

# ----------------------------
# Best Model
# ----------------------------
best_model = grid.best_estimator_

print("\n✅ BEST PARAMETERS FOUND:")
print(grid.best_params_)

# ----------------------------
# Save Best Model
# ----------------------------
joblib.dump(best_model, "models/best_rf_gridsearch.pkl")

print("\n💾 Best tuned model saved successfully")
