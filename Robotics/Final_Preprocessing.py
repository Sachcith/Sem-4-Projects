import numpy as np
import librosa
from scipy.signal import butter, filtfilt


class Preprocessing:

    def __init__(self, frame_length=2048, hop_length=512, fixed_sample_rate=48000, fixed_size=1.5, target_rms=0.1, low=200, high=8000, pre_emphasis_coef=0.97, n_mels=128):

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fixed_sample_rate = fixed_sample_rate
        self.fixed_size = fixed_size
        self.fixed_length = int(fixed_size * fixed_sample_rate)
        self.target_rms = target_rms
        self.low = low
        self.high = high
        self.pre_emphasis_coef = pre_emphasis_coef
        self.n_mels = n_mels

    def trim_pad_standardize(self, audio_files):

        AF = audio_files
        raw_datas = []
        trim_datas = []
        start_samples = []
        end_samples = []

        for i in range(len(AF)):

            raw_data, sr = librosa.load(
                audio_files[i],
                sr=self.fixed_sample_rate,
                mono=True
            )

            raw_datas.append(raw_data)

            rms = librosa.feature.rms(
                y=raw_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]

            times = librosa.frames_to_samples(
                np.arange(len(rms)),
                hop_length=self.hop_length
            )

            noise_floor = np.percentile(rms, 15)
            threshold = max(noise_floor * 2, np.mean(rms))

            spike = np.where(rms > threshold)[0]

            if len(spike) == 0:
                trim_datas.append(np.zeros(self.fixed_length))
                start_samples.append(0)
                end_samples.append(self.fixed_length)
                continue

            start_frame = spike[0]
            end_frame = spike[-1]

            start_sample = times[start_frame]
            end_sample = min(times[end_frame] + self.frame_length, len(raw_data))

            onsets = librosa.onset.onset_detect(
                y=raw_data,
                sr=self.fixed_sample_rate,
                units='samples'
            )

            if len(onsets) > 0:
                nearest_onset = onsets[np.argmin(np.abs(onsets - start_sample))]
                start_sample = min(start_sample, nearest_onset)

            margin = int(0.01 * sr)
            start_sample = max(0, start_sample - margin)
            end_sample = min(len(raw_data), end_sample + margin)

            start_samples.append(start_sample)
            end_samples.append(end_sample)

            RawData_trimmed = raw_data[start_sample:end_sample]

            if len(RawData_trimmed) < self.fixed_length:
                RawData_trimmed = np.pad(
                    RawData_trimmed,
                    (0, self.fixed_length - len(RawData_trimmed))
                )
            else:
                RawData_trimmed = RawData_trimmed[:self.fixed_length]

            trim_datas.append(RawData_trimmed)

        return raw_datas, trim_datas, start_samples, end_samples

    def rms_normalization(self, trims_datasets):

        TD = trims_datasets

        for i in range(len(TD)):
            TD[i] = TD[i] * (
                self.target_rms /
                (np.sqrt(np.mean(TD[i] ** 2)) + 1e-8)
            )

        return TD

    def bandpass(self, trimmed_norm_data):

        b, a = butter(
            4,
            [
                self.low * 2 / self.fixed_sample_rate,
                self.high * 2 / self.fixed_sample_rate
            ],
            btype='band'
        )

        return filtfilt(b, a, trimmed_norm_data)

    def pre_emphasis(self, denoised_data):

        DD = denoised_data

        for i in range(len(DD)):
            DD[i] = librosa.effects.preemphasis(
                DD[i],
                coef=self.pre_emphasis_coef
            )

        return DD

    def feature_extraction(self, final_trim_data):

        FD = librosa.feature.melspectrogram(
            y=final_trim_data,
            sr=self.fixed_sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        FD_PtoD = librosa.power_to_db(FD)

        return FD_PtoD
    
    def STFT(self, Normalized_data, window_type="hamming", spectrum_type="power", scaling="log", keep_positive_freq=True, center=False):

        frame_size = self.frame_length
        hop_length = self.hop_length
        n_fft = self.frame_length

        Ndata = Normalized_data
        No_of_frames = 1 + int(
            np.floor((len(Normalized_data) - frame_size) / hop_length)
        )

        Sliced_NData_frames = []

        temp = 0
        for _ in range(No_of_frames):
            k = Ndata[temp:temp + frame_size]
            Sliced_NData_frames.append(k)
            temp += hop_length

        N = frame_size
        for i in range(len(Sliced_NData_frames)):
            HWindow = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / N)
            Sliced_NData_frames[i] *= HWindow

        STFT_Data = []
        for frame in Sliced_NData_frames:

            if N < n_fft:
                padded_frame = np.zeros(n_fft)
                padded_frame[:N] = frame
            else:
                padded_frame = frame

            X = []
            for k in range(n_fft):
                cp_n = 0 + 0j
                for n in range(len(padded_frame)):
                    cp_n += padded_frame[n] * np.exp(
                        -1j * 2 * np.pi * k * n / n_fft
                    )
                X.append(cp_n)

            X = np.array(X)

            if keep_positive_freq:
                X = X[:n_fft // 2 + 1]

            if spectrum_type == "power":
                P = np.abs(X) ** 2
            else:
                P = np.abs(X)

            if scaling == "log":
                P = np.log(P + 1e-10)

            STFT_Data.append(P)

        return np.array(STFT_Data)

    def STFT_fast(self, normalized_data, window_type="hamming", spectrum_type="power", scaling="log", center=False):

        frame_size = self.frame_length
        hop_length = self.hop_length
        n_fft = self.frame_length

        if window_type == "hamming":
            window = np.hamming(frame_size)
        elif window_type == "hann":
            window = np.hanning(frame_size)
        else:
            raise ValueError("Unsupported window type")

        if center:
            pad = n_fft // 2
            normalized_data = np.pad(
                normalized_data,
                (pad, pad),
                mode="reflect"
            )

        num_frames = 1 + (len(normalized_data) - frame_size) // hop_length

        frames = np.lib.stride_tricks.as_strided(
            normalized_data,
            shape=(num_frames, frame_size),
            strides=(
                normalized_data.strides[0] * hop_length,
                normalized_data.strides[0]
            )
        ).copy()

        frames *= window

        stft = np.fft.rfft(frames, n=n_fft, axis=1)

        if spectrum_type == "power":
            spec = np.abs(stft) ** 2
        else:
            spec = np.abs(stft)

        if scaling == "log":
            spec = np.log(spec + 1e-10)

        return spec