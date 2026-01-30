import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------
# Load models
# ----------------------------
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_pipeline.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_pipeline.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "Advanced RF (Tuned)": joblib.load("models/best_rf_gridsearch.pkl")
}

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/clean_drug_data.csv")

X = df["review"]
y = df["label"]

# ----------------------------
# Train-test split (unseen test)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=99,  # different seed = unseen
    stratify=y
)

# ----------------------------
# Evaluate all models
# ----------------------------
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

# ----------------------------
# Display results
# ----------------------------
results_df = pd.DataFrame(results).T
print("\n📊 FINAL MODEL COMPARISON:")
print(results_df)

# ----------------------------
# Select best model (highest F1)
# ----------------------------
best_model_name = results_df["F1"].idxmax()
best_model = models[best_model_name]

print(f"\n🏆 SELECTED FINAL MODEL: {best_model_name}")

# ----------------------------
# Save final model
# ----------------------------
joblib.dump(best_model, "models/final_adverse_effect_model.pkl")
print("\n💾 Final model saved for deployment")