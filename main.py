from preprocessing.preprocess import load_eeg
from features.extract_features import extract_features
from models.train_model import *
from models.evaluate_model import *

# Run Preprocessing
print("Loading and Preprocessing EEG Data...")
eeg_data = load_eeg("./data/eeg_data.csv")

# Extract Features
print("Extracting Features...")
features = extract_features(eeg_data)
features.to_csv("./data/extracted_features.csv", index=False)

# Train Model
print("Training Models...")
train_models()

# Evaluate
print("Evaluating Models...")
evaluate_models()
