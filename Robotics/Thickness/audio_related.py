import numpy as np
import librosa

def preprocess_audio(file_path, sr=16000):

    y, sr = librosa.load(file_path, sr=sr, mono=True)

    y, _ = librosa.effects.trim(y, top_db=30)

    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms

    return y, sr


def extract_features(file_path, sr=16000):

    y, sr = preprocess_audio(file_path, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)

    decay_rate = np.mean(np.diff(rms))

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(centroid),
        np.mean(bandwidth),
        np.mean(rolloff),
        np.mean(flatness),
        np.mean(zcr),

        np.mean(rms),
        decay_rate
    ])

    return features