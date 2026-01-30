import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/clean_drug_data.csv")

X = df["review"]
y = df["label"]

# ----------------------------
# Stratified split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Advanced Random Forest
# ----------------------------
advanced_rf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# ----------------------------
# Train
# ----------------------------
advanced_rf.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = advanced_rf.predict(X_test)
y_prob = advanced_rf.predict_proba(X_test)[:, 1]

print("\n🚀 ADVANCED RANDOM FOREST RESULTS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Advanced Random Forest - Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Advanced Random Forest - ROC Curve")
plt.show()

# ----------------------------
# Save model
# ----------------------------
joblib.dump(advanced_rf, "models/advanced_random_forest.pkl")
print("\n✅ Advanced model saved successfully")
