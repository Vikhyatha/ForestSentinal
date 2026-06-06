import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file = "audio/1-7057-A-12.wav"

y, sr = librosa.load(audio_file)

S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10,4))
librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.tight_layout()

plt.savefig("spectrogram.png")
plt.show()