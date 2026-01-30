import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ----------------------------
# Load model
# ----------------------------
model = joblib.load("models/best_rf_gridsearch.pkl")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/clean_drug_data.csv")

X = df["review"]
y = df["label"]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Predict probabilities
# ----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Default threshold (0.5)
# ----------------------------
y_pred_default = (y_prob >= 0.5).astype(int)

# ----------------------------
# Optimized threshold (0.3)
# ----------------------------
threshold = 0.3
y_pred_optimized = (y_prob >= threshold).astype(int)

# ----------------------------
# Metrics
# ----------------------------
print("\n📊 DEFAULT THRESHOLD (0.5)")
print("Precision:", precision_score(y_test, y_pred_default))
print("Recall   :", recall_score(y_test, y_pred_default))
print("F1 Score :", f1_score(y_test, y_pred_default))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

print("\n📊 OPTIMIZED THRESHOLD (0.3)")
print("Precision:", precision_score(y_test, y_pred_optimized))
print("Recall   :", recall_score(y_test, y_pred_optimized))
print("F1 Score :", f1_score(y_test, y_pred_optimized))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_optimized))

# ----------------------------
# Business Explanation
# ----------------------------
print("\n💼 BUSINESS INSIGHT:")
print(
    "Lowering the threshold increases recall, ensuring fewer adverse drug effects are missed.\n"
    "This is critical in healthcare where false negatives pose higher risk than false positives."
)
