import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, output_image):

    # load audio
    y, sr = librosa.load(audio_path)

    # create spectrogram
    plt.figure(figsize=(10,4))

    S = librosa.feature.melspectrogram(y=y, sr=sr)

    S_dB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+2.0f dB')

    plt.title('Mel Spectrogram')

    plt.tight_layout()

    plt.savefig(output_image)

    plt.close()