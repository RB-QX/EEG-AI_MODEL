import pandas as pd
import matplotlib.pyplot as plt

# Load EEG Data
df = pd.read_csv("../data/eeg_data.csv")

# Plot Sample EEG Channels
plt.figure(figsize=(12, 6))
for i in range(3):  # Plot first 3 channels
    plt.plot(df.iloc[:, i+1], label=f"Channel {i+1}")

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("EEG Signals")
plt.legend()
plt.show()
