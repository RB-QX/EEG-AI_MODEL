import numpy as np
import pandas as pd
from scipy.signal import welch

# Compute Power Spectral Density (PSD)
def compute_psd(eeg_signal, fs=256):
    freqs, psd = welch(eeg_signal, fs, nperseg=fs*2)
    return np.mean(psd), np.std(psd)  # Return Mean and Std of PSD

# Extract Features (Time + Frequency domain)
def extract_features(eeg_df):
    features = []
    for col in eeg_df.columns[1:]:  # Skip Time column
        mean_psd, std_psd = compute_psd(eeg_df[col].values)
        features.append([col, mean_psd, std_psd])  # Include column name for debugging
    
    feature_df = pd.DataFrame(features, columns=["Channel", "Mean_PSD", "Std_PSD"])
    print("\nExtracted Features (First 5 Rows):")
    print(feature_df.head())  # Debugging print
    
    return feature_df

if __name__ == "__main__":
    eeg_data = pd.read_csv("../data/eeg_data.csv")
    
    # Check if data exists
    if eeg_data.empty:
        print("❌ ERROR: EEG Data is empty! Check eeg_data.csv")
    else:
        features = extract_features(eeg_data)
        features.to_csv("../data/extracted_features.csv", index=False)
        print("✅ Features successfully extracted and saved!")
