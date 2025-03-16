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
        features.append([mean_psd, std_psd])
    
    feature_df = pd.DataFrame(features, columns=["Mean_PSD", "Std_PSD"])
    return feature_df

if __name__ == "__main__":
    eeg_data = pd.read_csv("../data/eeg_data.csv")
    features = extract_features(eeg_data)
    features.to_csv("../data/extracted_features.csv", index=False)
    print(features.head())
