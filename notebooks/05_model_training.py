import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------
# STEP 1: Paths
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# STEP 2: Load processed data
# -----------------------------------
X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "processed_data.pkl")

# -----------------------------------
# STEP 3: Logistic Regression
# -----------------------------------
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Save model
joblib.dump(lr, MODELS_DIR / "logistic_regression_model.pkl")

# -----------------------------------
# STEP 4: Random Forest Classifier
# -----------------------------------
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save model
joblib.dump(rf, MODELS_DIR / "random_forest_model.pkl")

# -----------------------------------
# STEP 5: Save confusion matrices
# -----------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(RESULTS_DIR / "confusion_matrix_lr.png")
plt.close()

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(RESULTS_DIR / "confusion_matrix_rf.png")
plt.close()

print("Model training completed. Models and results saved!")
