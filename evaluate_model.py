import os
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
TEST_PATH = "dataset_split/test"
MODEL_PATH = "sound_model.pkl"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
CM_SAVE_PATH = "static/confusion_matrix.png"
# =========================


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


def load_test_data(test_path):
    X, y = [], []

    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Test folder not found: {test_path}")

    for label in os.listdir(test_path):
        class_dir = os.path.join(test_path, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                continue

            fpath = os.path.join(class_dir, fname)
            try:
                feat = extract_features(fpath)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"Skipping {fpath} -> {e}")

    return np.array(X), np.array(y)


def save_confusion_matrix(y_true, y_pred, out_path):
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = load_test_data(TEST_PATH)

    if len(X_test) == 0:
        raise ValueError("No valid test samples found.")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Total test samples:", len(X_test))
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    save_confusion_matrix(y_test, y_pred, CM_SAVE_PATH)
    print(f"\nConfusion matrix saved to: {CM_SAVE_PATH}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        conf = np.max(probs, axis=1)
        correct = (y_pred == y_test)

        avg_conf_correct = np.mean(conf[correct]) * 100 if np.any(correct) else 0
        avg_conf_wrong = np.mean(conf[~correct]) * 100 if np.any(~correct) else 0

        print(f"Average confidence (correct): {avg_conf_correct:.2f}%")
        print(f"Average confidence (wrong):   {avg_conf_wrong:.2f}%")
    else:
        print("Model does not support predict_proba().")


if __name__ == "__main__":
    main()