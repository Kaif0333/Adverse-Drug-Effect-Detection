import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# -----------------------------------
# Paths
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# -----------------------------------
# Load data and models
# -----------------------------------
df = pd.read_csv(DATA_DIR / "clean_drug_data.csv")

# Load Tfidf vectorizer
tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
rf_model_path = MODELS_DIR / "random_forest_model.pkl"

tfidf = joblib.load(tfidf_path)
rf_model = joblib.load(rf_model_path)

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(page_title="Adverse Drug Effect Dashboard", layout="wide")

st.title("💊 Adverse Drug Effect Detection & Safety Analytics")
st.markdown("""
Interactive dashboard to predict adverse effects from drug reviews.  
Visualize which words contribute most to risk using SHAP explanations.
""")

# -----------------------------------
# Sidebar Options
# -----------------------------------
st.sidebar.header("Options")
option = st.sidebar.radio("Choose Option", ["Predict Review", "Top Risky Drugs"])

# -----------------------------------
# Predict Review Section
# -----------------------------------
if option == "Predict Review":
    st.subheader("Predict Probability of Adverse Effect")
    review_input = st.text_area("Enter drug review text here")
    
    if st.button("Predict"):
        if review_input.strip() == "":
            st.warning("Please enter a review text!")
        else:
            # Clean input
            review_clean = review_input.lower().strip()
            X_vect = tfidf.transform([review_clean])
            X_vect_dense = X_vect.toarray()  # Fix SHAP issue

            # Predict probability using Random Forest
            prob = rf_model.predict_proba(X_vect_dense)[0][1]
            st.success(f"Predicted probability of adverse effect: {prob*100:.2f}%")

            # ---------------------------
            # SHAP explanation
            # ---------------------------
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_vect_dense)

            st.subheader("Top words influencing prediction")
            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values,
                features=X_vect_dense,
                feature_names=tfidf.get_feature_names_out(),
                show=False,
                plot_type="bar",
                max_display=10
            )
            st.pyplot(fig)

# -----------------------------------
# Top Risky Drugs Section
# -----------------------------------
else:
    st.subheader("Top N Drugs with Adverse Effects")
    n = st.slider("Select number of top drugs", min_value=5, max_value=20, value=10)
    
    risky = df[df['label']==1]['drugName'].value_counts().head(n)
    
    st.bar_chart(risky)
    
    st.write("Data Table:")
    st.dataframe(
        risky.reset_index().rename(columns={'index':'Drug Name', 'drugName':'Adverse Count'})
    )
