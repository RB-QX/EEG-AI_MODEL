import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, lfilter

# Bandpass Filter (0.5 - 45 Hz)
def bandpass_filter(data, lowcut=0.5, highcut=45, fs=256, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)

# Load EEG Data
def load_eeg(file_path):
    df = pd.read_csv(file_path)  # Ensure dataset has a time column + EEG channels
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: bandpass_filter(x))
    return df

if __name__ == "__main__":
    eeg_data = load_eeg("../data/eeg_data.csv")
    print(eeg_data.head())
