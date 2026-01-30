# Adverse Drug Effect Detection & Safety Analytics

## 📌 Project Overview
This project focuses on detecting **adverse drug effects** from patient reviews using **Data Analytics, NLP, and Machine Learning**.  
It helps identify negative drug reactions early to support healthcare safety decisions.

---

## 🎯 Objectives
- Analyze drug review data
- Detect adverse drug reactions from text
- Perform exploratory data analysis (EDA)
- Build and evaluate machine learning models
- Deploy a simple interactive dashboard

---

## 🧠 Domain
- Data Analytics
- Machine Learning
- Natural Language Processing (NLP)

---

## 🗂 Project Structure
Adverse_Drug_Effect_Project/


│── data/

│ ├── raw_drug_data.csv

│ ├── clean_drug_data.csv

│

│── notebooks/

│ ├── 01_data_loading.ipynb

│ ├── 02_data_cleaning.ipynb

│ ├── 03_eda.ipynb

│ ├── 04_model_training.ipynb

│

│── src/

│ ├── preprocess.py

│ ├── train_model.py

│ ├── predict.py

│

│── models/

│ ├── adverse_model.pkl

│ ├── tfidf_vectorizer.pkl

│

│── app.py

│── requirements.txt

│── README.md

---

## 📊 Dataset
- Source: Drug review dataset
- Features:
  - drugName
  - condition
  - review
  - rating
  - label (0 = No Adverse Effect, 1 = Adverse Effect)

---

## 🧹 Data Processing
- Missing values handled
- Text cleaned and normalized
- Data types corrected
- Clean dataset saved separately

---

## 📈 Exploratory Data Analysis (EDA)
- Distribution of adverse vs non-adverse reviews
- Rating vs adverse effect analysis
- Review length comparison
- Statistical insights documented

---

## 🤖 Machine Learning
- TF-IDF for text feature extraction
- Random Forest Classifier
- Model evaluation and saving

---

## 🖥 Dashboard
- Built using **Streamlit**
- User enters a drug review
- Model predicts adverse effect in real-time

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
streamlit run app.py
```
✅ Results

Successfully detects adverse drug reactions

Provides analytical insights into drug safety

Can assist healthcare analytics systems

📌 Future Enhancements

Improve accuracy using class balancing

Add model explainability (SHAP)

Deploy on cloud platform

👨‍🎓 Academic Use

This project is developed as part of B.Tech Final Year Project.
📬 Author

S. Mohammed Kaif Basha
B.Tech – Computer Science ( AI & ML )