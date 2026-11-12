
import os
import csv
import joblib
import matplotlib.pyplot as plt

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/best_model.joblib")
vectorizer = joblib.load(
    "data/processed/features/tfidf_vectorizer.joblib"
)

# Get feature names
feature_names = list(vectorizer.get_feature_names_out())
importances = model.feature_importances_

# Add names for any remaining Word2Vec dimensions
if len(feature_names) < len(importances):
    remaining = len(importances) - len(feature_names)
    feature_names.extend(
        [f"word2vec_dimension_{i}" for i in range(remaining)]
    )

if len(feature_names) != len(importances):
    raise ValueError("Feature names and importance count do not match.")

# Select top 20 features
indices = importances.argsort()[-20:]
top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]

# Create output directory
os.makedirs("outputs/plots", exist_ok=True)

# Save chart
plt.figure(figsize=(10, 8))
plt.barh(top_features, top_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 20 Important Features - Extra Trees")
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png", dpi=300)
plt.show()

# Save feature importance data
with open("outputs/feature_importance.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Feature", "Importance"])
    for i in indices[::-1]:
        writer.writerow([feature_names[i], importances[i]])

print("Feature importance chart saved successfully!")
print("CSV saved successfully!")
print("\nTop 10 Important Features:")
for i in indices[::-1][:10]:
    print(f"{feature_names[i]}: {importances[i]:.6f}")
