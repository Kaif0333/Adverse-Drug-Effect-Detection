import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load best tuned model
# ----------------------------
model = joblib.load("models/best_rf_gridsearch.pkl")

# ----------------------------
# Access TF-IDF and RF parts
# ----------------------------
tfidf = model.named_steps["tfidf"]
rf = model.named_steps["rf"]

feature_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_

# ----------------------------
# Create importance dataframe
# ----------------------------
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).head(20)

# ----------------------------
# Plot Feature Importance
# ----------------------------
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)
plt.gca().invert_yaxis()
plt.title("Top 20 Important Words Influencing Adverse Drug Effect")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ----------------------------
# Print insights
# ----------------------------
print("\n🧠 Model Interpretation Insights:")
print("Top words contributing to adverse drug effect prediction:")
print(importance_df)
