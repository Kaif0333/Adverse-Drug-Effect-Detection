import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Adverse Drug Effect Detection",
    layout="wide"
)

st.title("💊 Adverse Drug Effect Detection & Safety Analytics")

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    with open("models/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Predict Review"]
)

# ---------------------------
# Overview Page
# ---------------------------
if page == "Overview":
    st.subheader("📊 Project Overview")

    st.markdown("""
    **Project Name:** Adverse Drug Effect Detection  
    **Domain:** Data Analytics & Machine Learning  

    ### What this system does:
    - Takes a drug review as input
    - Uses NLP + ML to detect adverse effects
    - Helps in drug safety monitoring

    ### Model Used:
    - TF-IDF Vectorization
    - Random Forest Classifier
    """)

# ---------------------------
# Prediction Page
# ---------------------------
elif page == "Predict Review":
    st.subheader("🧪 Predict Adverse Drug Effect")

    review_text = st.text_area(
        "Enter Drug Review:",
        height=150,
        placeholder="Type a patient drug review here..."
    )

    if st.button("Predict"):
        if review_text.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            X_input = vectorizer.transform([review_text])
            prediction = model.predict(X_input)[0]

            if prediction == 1:
                st.error("🚨 Adverse Drug Effect Detected")
            else:
                st.success("✅ No Adverse Drug Effect Detected")
