import random
import numpy as np
from scipy.stats import entropy
import librosa
from scipy.signal import butter, filtfilt
import os, glob, gc, time
# import numpy as np
# import librosa
try:
    import torch
    TORCH_AVAILABLE = True
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    USE_CUDA = False
    DEVICE = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split as sklearn_split
import pywt 
try:
    from pytorch_wavelets import DWTForward
    PWAVELETS_AVAILABLE = True
except Exception:
    DWTForward = None
    PWAVELETS_AVAILABLE = False

# Change these paths
input_file_path = "input.txt"
output_file_path = "expected_output.txt"

TEST_CASES = 1
SIZE = 32000


# def mean(arr):
#     if len(arr) == 0:
#         return 0
#     total = 0
#     for x in arr:
#         total += x
#     return total / len(arr)

# def standard_deviation(arr):
#     arr = np.array(arr, dtype=np.float32)
#     return np.std(arr)

# def max_value(arr):
#     arr = np.array(arr,dtype=np.float32)
#     return np.max(arr)

# def min_value(arr):
#     arr = np.array(arr,dtype=np.float32)
#     return np.min(arr)

# def kth_smallest(arr, k):
#     arr = sorted(arr)
#     return arr[k-1]

# def skewness(arr):
#     arr = np.array(arr, dtype=np.float32)
#     mean = np.mean(arr)
#     std = np.std(arr)
#     return np.mean(((arr - mean) / std) ** 3)

# def kurtosis(arr):
#     arr = np.array(arr, dtype=np.float32)
#     mean = np.mean(arr)
#     std = np.std(arr)
#     return np.mean(((arr - mean) / std) ** 4)

# def zero_crossing_rate(arr):
#     arr = np.array(arr, dtype=np.float32).reshape(10, 10)
#     diff = np.diff(arr, axis=1)
#     zcr = np.mean(np.abs(np.sign(diff[:, 1:]) - np.sign(diff[:, :-1])) / 2)
#     return zcr

# def normalize(data):
#     data = np.array(data, dtype=np.float32)
#     rms = np.sqrt(np.mean(data**2, dtype=np.float32), dtype=np.float32)
#     return data / rms if rms > 0 else data

# def spectral_centroid(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     freqs = np.arange(spec.shape[0])
#     spec_sum = np.sum(spec, axis=0)
#     spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
#     centroid = np.sum(freqs[:, np.newaxis] * spec, axis=0) / spec_sum
#     return np.mean(centroid), np.std(centroid)

# def spectral_bandwidth(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     freqs = np.arange(spec.shape[0])
#     spec_sum = np.sum(spec, axis=0)
#     spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
#     centroid = np.sum(freqs[:, np.newaxis] * spec, axis=0) / spec_sum
#     bandwidth = np.sqrt(np.sum(((freqs[:, np.newaxis] - centroid) ** 2) * spec, axis=0) / spec_sum)
#     return np.mean(bandwidth), np.std(bandwidth)

# def spectral_rolloff(spec, percentile=0.85):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     spec_sum = np.sum(spec, axis=0, keepdims=True)
#     spec_sum = np.where(spec_sum == 0, 1e-10, spec_sum)
#     cumsum = np.cumsum(spec, axis=0) / spec_sum
#     rolloff = np.argmax(cumsum >= percentile, axis=0)
#     return np.mean(rolloff), np.std(rolloff)

# def spectral_flatness(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     spec_positive = np.maximum(spec, 1e-10)
#     geo_mean = np.exp(np.mean(np.log(spec_positive), axis=0))
#     arith_mean = np.mean(spec_positive, axis=0)
#     flatness = geo_mean / np.maximum(arith_mean, 1e-10)
#     return np.mean(flatness), np.std(flatness)

# def spectral_contrast(spec, n_bands=4):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     n_freq = spec.shape[0]
#     band_size = n_freq // n_bands
#     contrasts = []
#     for b in range(n_bands):
#         start = b * band_size
#         end = (b + 1) * band_size if b < n_bands - 1 else n_freq
#         band = spec[start:end, :]
#         peak = np.max(band, axis=0)
#         valley = np.min(band, axis=0)
#         contrasts.append(np.mean(peak - valley))
#     return contrasts

# def frequency_band_energies(spec, n_bands=5):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     n_freq = spec.shape[0]
#     band_size = n_freq // n_bands
#     energies = []
#     for b in range(n_bands):
#         start = b * band_size
#         end = (b + 1) * band_size if b < n_bands - 1 else n_freq
#         band_energy = np.mean(spec[start:end, :] ** 2)
#         energies.append(band_energy)
#     return energies

# def temporal_features(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     frame_energy = np.sum(spec ** 2, axis=0)
#     onset_strength = np.diff(frame_energy)
#     onset_strength = np.maximum(onset_strength, 0)
    
#     return [
#         np.mean(frame_energy),
#         np.std(frame_energy),
#         np.max(frame_energy),
#         np.mean(onset_strength) if len(onset_strength) > 0 else 0.0,
#         np.std(onset_strength) if len(onset_strength) > 0 else 0.0,
#     ]

# def statistical_moments(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     flat = spec.flatten()
#     mean = np.mean(flat)
#     std = np.std(flat)
    
#     if std > 0:
#         skewness = np.mean(((flat - mean) / std) ** 3)
#         kurtosis = np.mean(((flat - mean) / std) ** 4) - 3
#     else:
#         skewness = 0.0
#         kurtosis = 0.0
    
#     return skewness, kurtosis

# def spectral_entropy_feature(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     psd_norm = spec / (np.sum(spec) + 1e-10)
#     ent = entropy(psd_norm.flatten())
#     return float(ent)

# def spectral_crest_factor(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     peak = np.max(spec)
#     mean = np.mean(spec)
#     return float(peak / (mean + 1e-10))

# def spectral_crest_factor(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     spec = spec.T
#     peak = np.max(spec)
#     mean = np.mean(spec)
#     return float(peak / (mean + 1e-10))


# def normalize_mean_std(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     reshaped_spec = spec.T
#     spec_mean = np.mean(reshaped_spec)
#     spec_std = np.std(reshaped_spec) + 1e-8
#     reshaped_spec = (reshaped_spec - spec_mean) / spec_std
#     return reshaped_spec.flatten(order="F")

# def clip(spec):
#     spec = np.array(spec,dtype=np.float32).reshape(5,20)
#     stft_spec = spec.T
#     stft_spec = np.clip(stft_spec, -100, 100)
#     return stft_spec.flatten(order="F")

# class FeatureExtractor:
#     SPEC_SHAPE = (128, 100)

#     def __init__(self, sr=16000, n_fft=1024, hop_length=256, win_length=1024, device=None):
#         self.sr = sr
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         if device is not None:
#             self.device = device
#         else:
#             self.device = DEVICE if TORCH_AVAILABLE else None

#     def compute_stft_spectrogram(self, audio_segment):
#         audio_segment = np.array(audio_segment,dtype=np.float32)
#         print(audio_segment.shape)
#         if TORCH_AVAILABLE and self.device is not None:
#             x = torch.from_numpy(audio_segment).float().to(self.device)
#             if self.win_length is not None and self.win_length > 0:
#                 win = torch.hann_window(self.win_length, device=self.device)
#             else:
#                 win = None
#             try:
#                 stft_result = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=win, return_complex=True)
#                 magnitude = stft_result.abs().cpu().numpy()
#             except Exception:
#                 stft_result = librosa.stft(audio_segment, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
#                 magnitude = np.abs(stft_result)
#         else:
#             stft_result = librosa.stft(audio_segment, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
#             magnitude = np.abs(stft_result)
#         log_spectrogram = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
#         target_shape = self.SPEC_SHAPE
#         freq_factor = target_shape[0] / log_spectrogram.shape[0]
#         time_factor = target_shape[1] / log_spectrogram.shape[1]
#         reshaped_spec = zoom(log_spectrogram, (freq_factor, time_factor), order=1)
#         reshaped_spec = reshaped_spec[:target_shape[0], :target_shape[1]]
#         if reshaped_spec.shape != target_shape:
#             padded = np.zeros(target_shape, dtype=np.float32)
#             padded[:reshaped_spec.shape[0], :reshaped_spec.shape[1]] = reshaped_spec
#             reshaped_spec = padded
#         spec_mean = np.mean(reshaped_spec)
#         spec_std = np.std(reshaped_spec) + 1e-8
#         reshaped_spec = (reshaped_spec - spec_mean) / spec_std
#         print(reshaped_spec.shape)
#         return reshaped_spec.astype(np.float16).flatten(order="F")

#     def compute_dwt_scalogram(self, audio_segment, wavelet='db4', level=6, shape=(128, 100)):
#         if TORCH_AVAILABLE and PWAVELETS_AVAILABLE and self.device is not None:
#             x = torch.from_numpy(audio_segment).float().unsqueeze(0).unsqueeze(0).to(self.device)
#             try:
#                 dwt = DWTForward(J=level, wave=wavelet).to(self.device)
#                 Yl, Yh = dwt(x)
#                 scalogram_list = []
#                 scalogram_list.append(Yl.squeeze().cpu().numpy())
#                 for yh in Yh:
#                     arr = yh.squeeze().cpu().numpy()
#                     if arr.ndim > 1:
#                         arr = arr.reshape(arr.shape[0], -1)
#                         arr = arr.mean(axis=0)
#                     scalogram_list.append(arr)
#                 max_len = max(len(c) for c in scalogram_list)
#                 scalogram_raw = np.zeros((len(scalogram_list), max_len), dtype=np.float32)
#                 for i, coeff in enumerate(scalogram_list):
#                     scalogram_raw[i, :len(coeff)] = coeff[:]
#                 magnitude = np.abs(scalogram_raw)
#                 scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
#             except Exception:
#                 # fallback to pywt
#                 coeffs = pywt.wavedec(audio_segment, wavelet, level=level)
#                 scalogram_list = []
#                 for coeff in coeffs:
#                     scalogram_list.append(coeff)
#                 max_len = max(len(c) for c in scalogram_list)
#                 scalogram_raw = np.zeros((len(scalogram_list), max_len), dtype=np.float32)
#                 for i, coeff in enumerate(scalogram_list):
#                     scalogram_raw[i, :len(coeff)] = coeff[:]
#                 magnitude = np.abs(scalogram_raw)
#                 scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
#         else:
#             coeffs = pywt.wavedec(audio_segment, wavelet, level=level)
#             scalogram_list = []
#             for coeff in coeffs:
#                 scalogram_list.append(coeff)
#             max_len = max(len(c) for c in scalogram_list)
#             scalogram_raw = np.zeros((len(scalogram_list), max_len), dtype=np.float32)
#             for i, coeff in enumerate(scalogram_list):
#                 scalogram_raw[i, :len(coeff)] = coeff[:]
#             magnitude = np.abs(scalogram_raw)
#             scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
#         # return scalogram_db.flatten()
#         if scalogram_db.shape != shape:
#             from scipy.ndimage import zoom
#             zoom_factor = (shape[0] / scalogram_db.shape[0], shape[1] / scalogram_db.shape[1])
#             scalogram_db = zoom(scalogram_db, zoom_factor, order=1)
#         answer = scalogram_db[:shape[0], :shape[1]].astype(np.float16)
#         print(answer.shape)
#         return answer.flatten()

def compute_mfcc_stats(spec, n_mfcc=13):
    spec = np.array(spec,dtype=np.float32).reshape(5,20)
    spec = spec.T
    try:
        mfcc = librosa.feature.mfcc(S=spec, n_mfcc=n_mfcc,sr=10)
        return float(np.mean(mfcc)), float(np.std(mfcc))
    except:
        return 0.0, 0.0


def apply_bandpass_filter(data, order=5):
        nyq = 0.5 * 16000
        low = 100 / nyq
        high = 1000 / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

def compute_stft_spectrogram(audio, sr=16000, n_fft=1024,
                             hop_length=256,
                             win_length=1024,
                             target_shape=(128,100)):
    audio = np.array(audio,dtype=np.float32)
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length)
    magnitude = np.abs(stft_result)
    log_spec = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
    freq_factor = target_shape[0] / log_spec.shape[0]
    time_factor = target_shape[1] / log_spec.shape[1]
    resized = zoom(log_spec, (freq_factor, time_factor), order=1)
    resized = resized[:target_shape[0], :target_shape[1]]
    if resized.shape != target_shape:
        padded = np.zeros(target_shape, dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        resized = padded
    mean, std = np.mean(resized), np.std(resized) + 1e-8
    resized = (resized - mean) / std
    return resized.flatten(order="F")
    print("#"*40)
    print(mean,std)
    print("STFT")
    print(temp[:25])
    print("#"*40)
    return resized.astype(np.float32)

with open(input_file_path, "w") as fin, open(output_file_path, "w") as fout:

    fin.write(str(TEST_CASES) + "\n")

    for _ in range(TEST_CASES):
        arr = [random.random()-1 for _ in range(SIZE)]

        # write input
        fin.write(" ".join(map(str, arr)) + "\n")

        # compute expected output
        # temp = FeatureExtractor()
        result = compute_stft_spectrogram(arr)
        # result = list(result)
        # result = apply_bandpass_filter(arr)
        # result = round(result,5)
        # result = [round(i,4) for i in result]

        result = [str(i) for i in result]
        # write output
        fout.write(" ".join(result) + " \n")
        # fout.write(str(result) + "\n")