# Adverse Drug Effect Detection & Safety Analytics

A machine learning and NLP project that analyzes drug reviews to identify possible adverse drug effects from patient-written text. The project combines data cleaning, exploratory analysis, TF-IDF feature extraction, model training, and a Streamlit dashboard for real-time prediction.

---

## Project Overview

Adverse Drug Effect Detection & Safety Analytics focuses on detecting negative drug reactions from patient reviews using **Data Analytics, Natural Language Processing, and Machine Learning**.

The goal is to support healthcare analytics by identifying review patterns that may indicate adverse drug reactions.

---

## Objectives

- Analyze drug review data
- Clean and preprocess text-based medical review data
- Perform exploratory data analysis on drug ratings and review patterns
- Train a machine learning model for adverse-effect classification
- Build an interactive Streamlit dashboard for real-time prediction

---

## Domain

- Data Analytics
- Machine Learning
- Natural Language Processing
- Healthcare Safety Analytics

---

## Tech Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| NLP | TF-IDF Vectorization |
| Machine Learning | scikit-learn, Random Forest Classifier |
| Dashboard | Streamlit |
| Visualization | Matplotlib / EDA notebooks |

---

## Project Structure

```text
Adverse_Drug_Effect_Project/
│
├── data/
│   ├── raw_drug_data.csv
│   └── clean_drug_data.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   ├── adverse_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses a drug review dataset with fields such as:

- `drugName`
- `condition`
- `review`
- `rating`
- `label` where `0 = No Adverse Effect` and `1 = Adverse Effect`

---

## Data Processing

- Handled missing values
- Cleaned and normalized review text
- Corrected data types
- Prepared clean datasets for model training
- Converted text reviews into numerical features using TF-IDF

---

## Exploratory Data Analysis

The EDA phase includes:

- Distribution of adverse vs non-adverse reviews
- Rating patterns across drug reviews
- Review length comparison
- Basic statistical insights for healthcare safety analysis

---

## Machine Learning Approach

The model pipeline uses:

1. Text cleaning and preprocessing
2. TF-IDF vectorization
3. Random Forest classification
4. Model evaluation
5. Saved model and vectorizer for dashboard prediction

---

## Streamlit Dashboard

The dashboard allows users to enter a drug review and receive a real-time prediction indicating whether the review may contain an adverse drug effect signal.

```bash
streamlit run app.py
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Kaif0333/Adverse-Drug-Effect-Detection.git
cd Adverse-Drug-Effect-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## Results

- Detects possible adverse drug reactions from review text
- Provides analytical insights into drug safety reviews
- Demonstrates end-to-end ML workflow from data cleaning to deployment-ready dashboard

---

## Project Demo

▶️ Screen Recording:  
https://drive.google.com/file/d/10uDarsyf8AddMCxoA0QDjtwyfUt0Z9Df/view

---

## Future Enhancements

- Improve model performance using class balancing
- Add model explainability using SHAP
- Deploy the Streamlit dashboard online
- Add more advanced transformer-based NLP models

---

## Author

**S. Mohammed Kaif Basha**  
B.Tech – Computer Science & Engineering (AI & ML)
