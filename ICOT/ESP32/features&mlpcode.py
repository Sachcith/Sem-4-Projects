import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.stats import entropy
import librosa

def spectral_centroid(spec):
    freqs = np.arange(spec.shape[0])
    spec_sum = np.sum(spec, axis=0)
    spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
    centroid = np.sum(freqs[:, np.newaxis] * spec, axis=0) / spec_sum
    return np.mean(centroid), np.std(centroid)

def spectral_bandwidth(spec):
    freqs = np.arange(spec.shape[0])
    spec_sum = np.sum(spec, axis=0)
    spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
    centroid = np.sum(freqs[:, np.newaxis] * spec, axis=0) / spec_sum
    bandwidth = np.sqrt(np.sum(((freqs[:, np.newaxis] - centroid) ** 2) * spec, axis=0) / spec_sum)
    return np.mean(bandwidth), np.std(bandwidth)

def spectral_rolloff(spec, percentile=0.85):
    spec_sum = np.sum(spec, axis=0, keepdims=True)
    spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
    cumsum = np.cumsum(spec, axis=0) / spec_sum
    rolloff = np.argmax(cumsum >= percentile, axis=0)
    return np.mean(rolloff), np.std(rolloff)

def spectral_flatness(spec):
    spec_positive = np.maximum(spec, 1e-10)
    geo_mean = np.exp(np.mean(np.log(spec_positive), axis=0))
    arith_mean = np.mean(spec_positive, axis=0)
    flatness = geo_mean / np.maximum(arith_mean, 1e-10)
    return np.mean(flatness), np.std(flatness)

def spectral_contrast(spec, n_bands=4):
    n_freq = spec.shape[0]
    band_size = n_freq // n_bands
    contrasts = []
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else n_freq
        band = spec[start:end, :]
        peak = np.max(band, axis=0)
        valley = np.min(band, axis=0)
        contrasts.append(np.mean(peak - valley))
    return contrasts

def zero_crossing_rate(spec):
    diff = np.diff(spec, axis=1)
    zcr = np.mean(np.abs(np.sign(diff[:, 1:]) - np.sign(diff[:, :-1])) / 2)
    return zcr

def frequency_band_energies(spec, n_bands=5):
    n_freq = spec.shape[0]
    band_size = n_freq // n_bands
    energies = []
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else n_freq
        band_energy = np.mean(spec[start:end, :] ** 2)
        energies.append(band_energy)
    return energies

def temporal_features(spec):
    frame_energy = np.sum(spec ** 2, axis=0)
    onset_strength = np.diff(frame_energy)
    onset_strength = np.maximum(onset_strength, 0)
    
    return [
        np.mean(frame_energy),
        np.std(frame_energy),
        np.max(frame_energy),
        np.mean(onset_strength) if len(onset_strength) > 0 else 0.0,
        np.std(onset_strength) if len(onset_strength) > 0 else 0.0,
    ]

def statistical_moments(spec):
    flat = spec.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    
    if std > 0:
        skewness = np.mean(((flat - mean) / std) ** 3)
        kurtosis = np.mean(((flat - mean) / std) ** 4) - 3
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    return skewness, kurtosis

def compute_mfcc_stats(spec, n_mfcc=13):
    try:
        mfcc = librosa.feature.mfcc(S=spec, n_mfcc=n_mfcc)
        return float(np.mean(mfcc)), float(np.std(mfcc))
    except:
        return 0.0, 0.0

def spectral_entropy_feature(spec):
    psd_norm = spec / (np.sum(spec) + 1e-10)
    ent = entropy(psd_norm.flatten())
    return float(ent)

def spectral_crest_factor(spec):
    peak = np.max(spec)
    mean = np.mean(spec)
    return float(peak / (mean + 1e-10))

def extract_handcrafted_features(X_stft, X_dwt):
    """
    Per spectrogram type:
    - Basic statistics: 6 (mean, std, max, min, q25, q75)
    - Spectral centroid: 2 (mean, std)
    - Spectral bandwidth: 2 (mean, std)
    - Spectral rolloff: 2 (mean, std)
    - Spectral flatness: 2 (mean, std)
    - Spectral contrast: 4 (4 bands)
    - Zero-crossing rate: 1
    - Frequency band energies: 5 (5 bands)
    - Temporal features: 5 (energy stats + onset)
    - Statistical moments: 2 (skewness, kurtosis)
    - MFCC stats: 2 (mean, std)
    - Spectral entropy: 1
    - Crest factor: 1 
    
    Total there are 35 × 2 (STFT + DWT) = 70 features 
    """

    n_samples = X_stft.shape[0]
    features_list = []
    
    for i in range(n_samples):
        stft_spec = np.nan_to_num(X_stft[i], nan=0.0, posinf=1.0, neginf=-1.0)
        dwt_spec = np.nan_to_num(X_dwt[i], nan=0.0, posinf=1.0, neginf=-1.0) 
        stft_spec = np.clip(stft_spec, -100, 100)
        dwt_spec = np.clip(dwt_spec, -100, 100)
        stft_positive = np.abs(stft_spec) + 1e-10
        dwt_positive = np.abs(dwt_spec) + 1e-10
        features = []
        
        for spec, spec_pos in [(stft_spec, stft_positive), (dwt_spec, dwt_positive)]:
            features.extend([
                float(np.mean(spec)),
                float(np.std(spec)) if np.std(spec) > 0 else 0.0,
                float(np.max(spec)),
                float(np.min(spec)),
                float(np.percentile(spec, 25)),
                float(np.percentile(spec, 75)),
            ])
            cent_mean, cent_std = spectral_centroid(spec_pos)
            features.extend([cent_mean, cent_std])
            bw_mean, bw_std = spectral_bandwidth(spec_pos)
            features.extend([bw_mean, bw_std])
            roll_mean, roll_std = spectral_rolloff(spec_pos)
            features.extend([roll_mean, roll_std])
            flat_mean, flat_std = spectral_flatness(spec_pos)
            features.extend([flat_mean, flat_std])
            contrasts = spectral_contrast(spec_pos, n_bands=4)
            features.extend(contrasts)
            zcr = zero_crossing_rate(spec)
            features.append(zcr)
            band_energies = frequency_band_energies(spec_pos, n_bands=5)
            features.extend(band_energies)
            temp_feats = temporal_features(spec)
            features.extend(temp_feats)
            skew, kurt = statistical_moments(spec)
            features.extend([skew, kurt])
            mfcc_mean, mfcc_std = compute_mfcc_stats(spec, n_mfcc=13)
            features.extend([mfcc_mean, mfcc_std])
            features.append(spectral_entropy_feature(spec_pos))
            features.append(spectral_crest_factor(spec_pos))
        features = [0.0 if (np.isnan(f) or np.isinf(f)) else float(f) for f in features]
        features_list.append(features)
    features = np.array(features_list, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
    features = np.clip(features, -1000, 1000)
    return features


### MLP
def build_student_mlp(num_features=70, num_classes=6, hidden_units=[64, 32]):
    inputs = layers.Input(shape=(num_features,), name='feature_input')
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.001), 
                        name=f'hidden_{i+1}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001), 
                          name='student_output')(x)
    model = Model(inputs=inputs, outputs=outputs, name='Student_MLP')
    return model

if __name__ == "__main__":
    model = build_student_mlp(num_features=70, num_classes=6)
    model.summary()