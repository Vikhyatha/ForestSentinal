import librosa
import numpy as np
import joblib

model = joblib.load("sound_model.pkl")

file_path = input("Enter audio file path: ")

audio, sr = librosa.load(file_path, sr=None)

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

mfcc_mean = np.mean(mfcc.T, axis=0)

mfcc_mean = mfcc_mean.reshape(1, -1)

prediction = model.predict(mfcc_mean)

print("Predicted Sound:", prediction[0])