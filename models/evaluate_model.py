import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("../data/extracted_features.csv")
labels = pd.read_csv("../data/labels.csv")

# Load Models
svm_model = joblib.load("../models/svm_model.joblib")
rf_model = joblib.load("../models/rf_model.joblib")

# Predictions
y_pred_svm = svm_model.predict(df)
y_pred_rf = rf_model.predict(df)

# Evaluate Performance
print("SVM Accuracy:", accuracy_score(labels, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(labels, y_pred_rf))

print("\nSVM Classification Report:\n", classification_report(labels, y_pred_svm))
print("\nRandom Forest Classification Report:\n", classification_report(labels, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(labels, y_pred_svm), annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()
