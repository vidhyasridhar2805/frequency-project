import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Load audio files
audio1, sr1 = librosa.load("audio1.wav.mp3", sr=None)
audio2, sr2 = librosa.load("audio2.wav.mp3", sr=None)

# Apply FFT
fft1 = np.abs(np.fft.fft(audio1))
fft2 = np.abs(np.fft.fft(audio2))

# Keep half spectrum
fft1 = fft1[:len(fft1)//2]
fft2 = fft2[:len(fft2)//2]

# Make same length
min_len = min(len(fft1), len(fft2))
fft1 = fft1[:min_len]
fft2 = fft2[:min_len]

# Plot graph
plt.figure(figsize=(10,5))
plt.plot(fft1, label="Audio 1")
plt.plot(fft2, label="Audio 2")
plt.title("Frequency Comparison")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

# Similarity score
similarity = 1 - cosine(fft1, fft2)
print("Similarity Score:", similarity)
