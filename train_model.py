import os
import pickle
import numpy as np
import librosa

from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================
# CONFIG
# =========================
TRAIN_PATH = "dataset_split/train"
MODEL_PATH = "sound_model.pkl"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
RANDOM_STATE = 42

# Class-wise augmentation
AUG_MAP = {
    "bird": 2,
    "animal": 2,
    "chainsaw": 6   # boost minority / weak class
}
# =========================


def mfcc_feature_from_audio(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


def extract_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def augment_noise(y, noise_level=0.005):
    noise = noise_level * np.random.randn(len(y))
    return y + noise


def augment_shift(y):
    if len(y) <= 20:
        return np.roll(y, 1)
    shift = np.random.randint(len(y) // 20, len(y) // 5 + 1)
    return np.roll(y, shift)


def augment_pitch(y, sr):
    n_steps = np.random.choice([-2, -1, 1, 2])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_gain(y):
    gain = np.random.uniform(0.8, 1.2)
    return y * gain


def augment_time_stretch(y):
    rate = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y, rate=rate)


def make_augmented_audio(audio, sr, aug_idx):
    # rotate augmentation types
    aug_type = aug_idx % 5
    if aug_type == 0:
        return augment_noise(audio)
    if aug_type == 1:
        return augment_shift(audio)
    if aug_type == 2:
        return augment_pitch(audio, sr)
    if aug_type == 3:
        return augment_gain(audio)
    return augment_time_stretch(audio)


def load_train_data(train_path):
    X, y_labels = [], []

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Train folder not found: {train_path}")

    for label in os.listdir(train_path):
        class_dir = os.path.join(train_path, label)
        if not os.path.isdir(class_dir):
            continue

        aug_count = AUG_MAP.get(label, 2)

        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                continue

            fpath = os.path.join(class_dir, fname)
            try:
                audio, sr = extract_audio(fpath)

                # Original
                base_feat = mfcc_feature_from_audio(audio, sr)
                X.append(base_feat)
                y_labels.append(label)

                # Class-wise augmented samples
                for i in range(aug_count):
                    aug_audio = make_augmented_audio(audio, sr, i)
                    aug_feat = mfcc_feature_from_audio(aug_audio, sr)
                    X.append(aug_feat)
                    y_labels.append(label)

            except Exception as e:
                print(f"Skipping {fpath} -> {e}")

    return np.array(X), np.array(y_labels)


def main():
    np.random.seed(RANDOM_STATE)

    X_train, y_train = load_train_data(TRAIN_PATH)

    if len(X_train) == 0:
        raise ValueError("No valid training samples found.")

    print("Total training samples (after augmentation):", len(X_train))
    print("Training class distribution:", Counter(y_train))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=12,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved as: {MODEL_PATH}")


if __name__ == "__main__":
    main()