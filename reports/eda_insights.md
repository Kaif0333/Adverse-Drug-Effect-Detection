# Exploratory Data Analysis (EDA) – Adverse Drug Effect Detection

## 1. Dataset Overview
- The cleaned dataset contains drug reviews along with ratings and labels.
- Label `0` indicates safe experience.
- Label `1` indicates adverse drug reaction.

## 2. Class Distribution
- Majority of reviews fall under the safe category.
- Adverse reactions form a smaller but significant portion.
- This indicates **class imbalance**, which must be handled during modeling.

## 3. Drug-wise Adverse Effects
- Certain drugs appear frequently in adverse reactions.
- This suggests that specific medications have higher risk profiles.
- These drugs should be closely monitored in real-world use.

## 4. Rating vs Adverse Effect
- Reviews labeled as adverse typically have lower ratings.
- Safe drug experiences show higher median ratings.
- Rating is a strong indicator of adverse reactions.

## 5. Review Length Analysis
- Adverse reviews are generally longer.
- Users tend to explain negative experiences in more detail.
- Text length can be a useful feature for prediction models.

## 6. Key Insights
- Adverse reactions are predictable using text and ratings.
- Review content provides valuable safety signals.
- Data quality is sufficient for machine learning modeling.
