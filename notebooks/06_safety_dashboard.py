import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------------
# STEP 1: Paths
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# -----------------------------------
# STEP 2: Load models & data
# -----------------------------------
tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
lr_model = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")
rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")

df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

# -----------------------------------
# STEP 3: Function to predict adverse effect
# -----------------------------------
def predict_adverse(drug_review, model="rf"):
    """
    drug_review : string, the review or description
    model       : "rf" or "lr"
    """
    review_clean = drug_review.lower().strip()
    X_vect = tfidf.transform([review_clean])
    
    if model == "rf":
        prob = rf_model.predict_proba(X_vect)[0][1]
    else:
        prob = lr_model.predict_proba(X_vect)[0][1]
    
    return prob

# -----------------------------------
# STEP 4: Function to show top risky drugs
# -----------------------------------
def top_risky_drugs(n=10):
    risky = df[df['label']==1]['drugName'].value_counts().head(n)
    plt.figure(figsize=(8,5))
    risky.plot(kind='bar', color='red')
    plt.title(f"Top {n} Drugs with Adverse Effects")
    plt.xlabel("Drug Name")
    plt.ylabel("Number of Adverse Reviews")
    plt.tight_layout()
    plt.show()

# -----------------------------------
# STEP 5: Interactive Dashboard
# -----------------------------------
print("=== Safety Analytics Dashboard ===")
print("Options:")
print("1. Predict adverse effect probability for a review")
print("2. Show top risky drugs")
print("3. Exit")

while True:
    choice = input("Enter option (1/2/3): ").strip()
    
    if choice == "1":
        review = input("Enter drug review text: ").strip()
        model_choice = input("Choose model - Logistic Regression (lr) or Random Forest (rf): ").strip().lower()
        prob = predict_adverse(review, model=model_choice)
        print(f"Predicted probability of adverse effect: {prob*100:.2f}%")
        
    elif choice == "2":
        n = input("How many top risky drugs to show? (default 10): ").strip()
        n = int(n) if n.isdigit() else 10
        top_risky_drugs(n=n)
        
    elif choice == "3":
        print("Exiting dashboard. Goodbye!")
        break
    else:
        print("Invalid option. Choose 1, 2, or 3.")
