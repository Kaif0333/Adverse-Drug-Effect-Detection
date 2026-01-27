from pathlib import Path
from src.data_processing import load_raw_data, clean_data, save_clean_data
from src.feature_engineering import process_features
from src.models import train_models

BASE_DIR = Path(__file__).resolve().parent

# Step 1: Load & clean data
train, test = load_raw_data(BASE_DIR)
clean_train = clean_data(train)
save_clean_data(clean_train, BASE_DIR)

# Step 2: Feature engineering
X_train, X_test, y_train, y_test, tfidf = process_features(BASE_DIR)

# Step 3: Train professional models
rf_model, xgb_model = train_models(X_train, y_train, X_test, y_test, BASE_DIR)
