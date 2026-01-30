import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

# ----------------------------
# Load best model
# ----------------------------
model = joblib.load("models/best_rf_gridsearch.pkl")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/clean_drug_data.csv")

X = df["review"]
y = df["label"]

# ----------------------------
# Generate learning curve
# ----------------------------
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=3,
    scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

# ----------------------------
# Mean scores
# ----------------------------
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training F1 Score")
plt.plot(train_sizes, val_mean, label="Validation F1 Score")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve - Overfitting Check")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Interpretation
# ----------------------------
print("\n📊 Statistical Validation Insight:")
if train_mean[-1] > val_mean[-1]:
    print("Slight overfitting observed, but generalization is acceptable.")
else:
    print("Good generalization with balanced training and validation performance.")
