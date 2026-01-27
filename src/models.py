from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb

def train_models(X_train, y_train, X_test, y_test, base_dir: Path):
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    
    joblib.dump(rf, base_dir / "models" / "random_forest_model.pkl")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    
    joblib.dump(xgb_model, base_dir / "models" / "xgboost_model.pkl")
    
    # Confusion matrices
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(results_dir / "confusion_matrix_rf.png")
    plt.close()
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens')
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(results_dir / "confusion_matrix_xgb.png")
    plt.close()
    
    print("Models and confusion matrices saved.")
    return rf, xgb_model
