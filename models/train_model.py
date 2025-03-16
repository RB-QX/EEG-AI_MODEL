import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump

# Load Features
df = pd.read_csv("../data/extracted_features.csv")
labels = pd.read_csv("../data/labels.csv")  # Assume a labels file with cognitive load levels

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

# Train ML Models
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
dump(svm_model, "../models/svm_model.joblib")

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
dump(rf_model, "../models/rf_model.joblib")

print("Models trained and saved!")
