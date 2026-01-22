import os, glob, gc, time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch
import pywt  # For Discrete Wavelet Transform

# Dataset Configuration
TARGET_SR = 16000  # Target Sampling Rate
LOW_CUT = 100  # Low Cut Frequency for Bandpass Filter
HIGH_CUT = 1000     # High Cut Frequency for Bandpass Filter
SEGMENT_DURATION = 3.0  
SEGMENT_OVERLAP = 0.25  #(25%)

STFT_N_FFT = 512    #  number of FFT components
STFT_HOP_LENGTH = 64 # number of samples between successive frames
STFT_WIN_LENGTH = 256 # number of samples in each window

# DWT Configuration
COMPUTE_DWT_SCALOGRAM = False  
COMPUTE_DWT_ENERGY = False     

DATASET_PATH = "../dataset"
OUTPUT_FOLDER = "../dataset_processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_class_map(root_path):
    try:
        items = os.listdir(root_path)
    except FileNotFoundError:
        return {}
    classes = [d for d in items if os.path.isdir(os.path.join(root_path, d))]
    classes.sort()
    return {class_name: i for i, class_name in enumerate(classes)}

class_map = get_class_map(DATASET_PATH)
inv_class_map = {v: k for k, v in class_map.items()}

class AudioCleaner:
    def __init__(self, sr=16000, low_cut=100, high_cut=1000):
        self.sr = sr
        self.low_cut = low_cut
        self.high_cut = high_cut

    def resample_audio(self, file_path, target_sr=None):
        sr = target_sr if target_sr else self.sr
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    
    def fix_length(self, data, target_length):
        if len(data) < target_length:
            return np.pad(data, (0, target_length - len(data)))
        return data[:target_length]
    
    def apply_pre_emphasis(self, data, coef=0.97):
        return np.append(data[0], data[1:] - coef * data[:-1])

    def apply_bandpass_filter(self, data, order=5):
        nyq = 0.5 * self.sr
        low = self.low_cut / nyq
        high = self.high_cut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def normalize(self, data):
        rms = np.sqrt(np.mean(data**2))
        return data / rms if rms > 0 else data

    def segment_audio(self, data, duration=3.0, overlap=0.25, class_name=None):
        if class_name:
            overlap = 0.0 if class_name == 'Active' else 0.90
            
        n = int(duration * self.sr)
        if len(data) < n:
            return []
            
        step = max(1, int(n * (1 - overlap)))
        return [data[i:i+n] for i in range(0, len(data) - n + 1, step)]

class FeatureExtractor:
    SPEC_SHAPE = (128, 100)
    
    def __init__(self, sr=16000, n_fft=512, hop_length=64, win_length=256):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def compute_stft_spectrogram(self, audio_segment):
        stft_result = librosa.stft(audio_segment, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        magnitude = np.abs(stft_result)
        log_spectrogram = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
        reshaped_spec = np.zeros(self.SPEC_SHAPE, dtype=np.float32)
        nf = min(log_spectrogram.shape[0], self.SPEC_SHAPE[0])
        nt = min(log_spectrogram.shape[1], self.SPEC_SHAPE[1])
        reshaped_spec[:nf, :nt] = log_spectrogram[:nf, :nt]
        
        return reshaped_spec.astype(np.float16)

    def compute_scalar_features(self, audio_segment):
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr)
        centroid_mean = float(np.mean(centroid))
        
        fft_spectrum = np.abs(np.fft.rfft(audio_segment))
        freqs = np.fft.rfftfreq(len(audio_segment), 1/self.sr)
        
        b1 = np.where((freqs >= 150) & (freqs <= 350))[0]
        b2 = np.where((freqs >= 400) & (freqs <= 500))[0]
        
        e1 = float(np.sum(fft_spectrum[b1]**2))
        e2 = float(np.sum(fft_spectrum[b2]**2))
        ber = e2 / e1 if e1 > 0 else 0.0
        
        zcr = librosa.feature.zero_crossing_rate(audio_segment)
        zcr_mean = float(np.mean(zcr))
        
        return np.array([centroid_mean, ber, zcr_mean], dtype=np.float16)
    
    def compute_dwt_scalogram(self, audio_segment, wavelet='db4', level=6, shape=(128, 100)):
        coeffs = pywt.wavedec(audio_segment, wavelet, level=level)
        scalogram_list = []
        for coeff in coeffs:
            scalogram_list.append(coeff)
        max_len = max(len(c) for c in scalogram_list)
        scalogram_raw = np.zeros((len(scalogram_list), max_len), dtype=np.float32)
        for i, coeff in enumerate(scalogram_list):
            scalogram_raw[i, :len(coeff)] = coeff[:]
        magnitude = np.abs(scalogram_raw)
        scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
        if scalogram_db.shape != shape:
            from scipy.ndimage import zoom
            zoom_factor = (shape[0] / scalogram_db.shape[0], shape[1] / scalogram_db.shape[1])
            scalogram_db = zoom(scalogram_db, zoom_factor, order=1)
        
        return scalogram_db[:shape[0], :shape[1]].astype(np.float16)
    
    def compute_cwt_scalogram_viz_only(self, audio_segment, wavelet='morl', scales=None, shape=(128, 100)):
        if scales is None:
            center_freq = pywt.central_frequency(wavelet)
            frequencies = np.logspace(np.log10(LOW_CUT), np.log10(HIGH_CUT), num=shape[0])
            scales = (center_freq * self.sr) / frequencies
            scales = scales[::-1]
        
        coefficients, _ = pywt.cwt(audio_segment, scales, wavelet, sampling_period=1.0/self.sr)
        magnitude = np.abs(coefficients)
        scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
        
        if scalogram_db.shape != shape:
            from scipy.ndimage import zoom
            zoom_factor = (shape[0] / scalogram_db.shape[0], shape[1] / scalogram_db.shape[1])
            scalogram_db = zoom(scalogram_db, zoom_factor, order=1)
        
        return scalogram_db[:shape[0], :shape[1]].astype(np.float16)
    
    def compute_discrete_dwt_features(self, audio_segment, wavelet='db4', level=6):
        coeffs = pywt.wavedec(audio_segment, wavelet, level=level)
        energies = []
        for coeff in coeffs:
            energy = float(np.sum(coeff**2))
            energies.append(energy)
        total_energy = sum(energies) + 1e-10
        normalized_energies = [e / total_energy for e in energies]
        
        return np.array(normalized_energies, dtype=np.float16)

def process_single_file(file_path):
    try:
        cleaner = AudioCleaner(sr=TARGET_SR, low_cut=LOW_CUT, high_cut=HIGH_CUT)
        extractor = FeatureExtractor(sr=TARGET_SR, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH, win_length=STFT_WIN_LENGTH)
        raw = cleaner.resample_audio(file_path, target_sr=TARGET_SR)
        min_len = int(TARGET_SR * SEGMENT_DURATION)
        if len(raw) < min_len:
            raw = np.pad(raw, (0, min_len - len(raw)))
        class_name = os.path.basename(os.path.dirname(file_path))
        chunks = cleaner.segment_audio(raw, duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP, class_name=class_name)
        out = []
        for chunk in chunks:
            try:
                filtered = cleaner.apply_bandpass_filter(chunk)
                normalized = cleaner.normalize(filtered)
                spec = extractor.compute_stft_spectrogram(normalized)
                feats = extractor.compute_scalar_features(normalized)
                
                # Conditionally compute DWT features based on config
                if COMPUTE_DWT_SCALOGRAM:
                    dwt_scalogram = extractor.compute_dwt_scalogram(normalized, wavelet='db4', level=6)
                else:
                    dwt_scalogram = None
                
                if COMPUTE_DWT_ENERGY:
                    dwt_feats = extractor.compute_discrete_dwt_features(normalized, wavelet='db4', level=6)
                else:
                    dwt_feats = None
                
                out.append((spec, dwt_scalogram, feats, dwt_feats))
            except Exception:
                continue
        return out if out else None
    except Exception:
        return None

def preprocess_and_save_incrementally(dataset_path, output_path, batch_size=200):
    os.makedirs(output_path, exist_ok=True)
    cmap = get_class_map(dataset_path)
    
    all_files, all_labels = [], []
    for cname, cid in cmap.items():
        files = sorted(glob.glob(os.path.join(dataset_path, cname, '**', '*.wav'), recursive=True) +
                       glob.glob(os.path.join(dataset_path, cname, '**', '*.mp3'), recursive=True))
        all_files.extend(files)
        all_labels.extend([cid]*len(files))
    print(f"Found {len(all_files)} audio files across {len(cmap)} classes")
    
    # Print file distribution per class
    for cname in cmap.keys():
        count = sum(1 for f, l in zip(all_files, all_labels) if l == cmap[cname])
        print(f"  {cname}: {count} files")
    
    Xs_buf, Xf_buf, y_buf = [], [], []
    Xdwt_buf, Xdwt_feat_buf = [], []  
    part = 0
    total_segments = 0
    processed_files = 0
    failed_files = 0
    
    for i, fp in enumerate(all_files):
        res = process_single_file(fp)
        if res:
            processed_files += 1
            lab = all_labels[i]
            for sp, dwt_sc, ft, dwt_ft in res:
                Xs_buf.append(sp)
                if dwt_sc is not None:
                    Xdwt_buf.append(dwt_sc) 
                Xf_buf.append(ft)
                if dwt_ft is not None:
                    Xdwt_feat_buf.append(dwt_ft)  
                y_buf.append(lab)
                total_segments += 1
        else:
            failed_files += 1
        
        if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
            print(f"Progress: {i+1}/{len(all_files)} files | Segments: {total_segments} | Failed: {failed_files}")
        
        if len(Xs_buf) >= batch_size or (i == len(all_files)-1 and len(Xs_buf)>0):
            save_data = {
                'feat': np.array(Xf_buf, dtype=np.float16),
                'labels': np.array(y_buf, dtype=np.int8)
            }
            if Xdwt_feat_buf:
                save_data['dwt_feat'] = np.array(Xdwt_feat_buf, dtype=np.float16)
            
            np.savez_compressed(os.path.join(output_path, f"data_batch_{part}.npz"), **save_data)
            np.save(os.path.join(output_path, f"X_spec_batch_{part}.npy"), np.array(Xs_buf, dtype=np.float16))
            
            if Xdwt_buf:
                np.save(os.path.join(output_path, f"X_dwt_batch_{part}.npy"), np.array(Xdwt_buf, dtype=np.float16))
            
            rng = np.array([np.min(Xs_buf), np.max(Xs_buf)], dtype=np.float32) if len(Xs_buf)>0 else np.array([0,0], dtype=np.float32)
            np.save(os.path.join(output_path, f"X_spec_range_batch_{part}.npy"), rng)
            
            saved_count = len(Xs_buf)
            features_saved = "STFT" + (" + DWT scalogram" if Xdwt_buf else "") + (" + DWT energy" if Xdwt_feat_buf else "")
            Xs_buf, Xf_buf, y_buf = [], [], []
            Xdwt_buf, Xdwt_feat_buf = [], []  
            part += 1
            print(f"Saved batch {part} with {saved_count} segments ({features_saved})")
    
    print(f"\nPreprocessing Summary:")
    print(f"  Total files processed: {processed_files}/{len(all_files)}")
    print(f"  Total segments generated: {total_segments}")
    print(f"  Failed files: {failed_files}")

def consolidate_dataset(output_path):
    data_files = sorted(glob.glob(os.path.join(output_path, 'data_batch_*.npz')))
    if not data_files:
        print('No batch files to consolidate.')
        return
        
    Xf, Xdwt_f, Ys, Xs, Xdwt = [], [], [], [], []
    for f in data_files:
        d = np.load(f)
        Xf.append(d['feat'])
        if 'dwt_feat' in d:
            Xdwt_f.append(d['dwt_feat'])  # Load DWT energy features
        Ys.append(d['labels'])
        
        suf = os.path.basename(f).replace('data_', '').replace('.npz', '')
        sf = os.path.join(output_path, f"X_spec_{suf}.npy")
        if os.path.exists(sf):
            Xs.append(np.load(sf))
        
        dwt_f = os.path.join(output_path, f"X_dwt_{suf}.npy")
        if os.path.exists(dwt_f):
            Xdwt.append(np.load(dwt_f))  # Load DWT scalograms

    np.save(os.path.join(output_path, 'X_train_feat.npy'), np.concatenate(Xf,0))
    if Xdwt_f:
        np.save(os.path.join(output_path, 'X_train_dwt_feat.npy'), np.concatenate(Xdwt_f,0))
    np.save(os.path.join(output_path, 'y_train.npy'), np.concatenate(Ys,0))
    if Xs:
        np.save(os.path.join(output_path, 'X_train_spec.npy'), np.concatenate(Xs,0))
    if Xdwt:
        np.save(os.path.join(output_path, 'X_train_dwt.npy'), np.concatenate(Xdwt,0))

    for f in data_files:
        os.remove(f)
    for npy_file in glob.glob(os.path.join(output_path, 'X_spec_batch_*.npy')):
        os.remove(npy_file)
    for npy_file in glob.glob(os.path.join(output_path, 'X_dwt_batch_*.npy')):
        os.remove(npy_file)  # Clean up DWT batch files
    for rng_file in glob.glob(os.path.join(output_path, 'X_spec_range_batch_*.npy')):
        os.remove(rng_file)
    print("Consolidation complete.")

'''
preprocess_and_save_incrementally(DATASET_PATH, OUTPUT_FOLDER, batch_size=200)
consolidate_dataset(OUTPUT_FOLDER)
'''


'''
Visualisation of Processed Data
'''

X_spec_path = os.path.join(OUTPUT_FOLDER, "X_train_spec.npy")
X_feat_path = os.path.join(OUTPUT_FOLDER, "X_train_feat.npy")
y_path = os.path.join(OUTPUT_FOLDER, "y_train.npy")

if os.path.exists(X_spec_path) and os.path.exists(y_path):
    X_spec = np.load(X_spec_path)
    X_feat = np.load(X_feat_path)
    y_all = np.load(y_path)
    print("Loaded dataset:", X_spec.shape, y_all.shape)
    print(f"Memory usage: {X_spec.nbytes / 1e9:.2f} GB (spec) + {X_feat.nbytes / 1e9:.2f} GB (feat)")
else:
    print("Dataset not found. Please run preprocessing first.")
    X_spec, X_feat, y_all = None, None, None

# 1) Comparative Spectrogram Grid 
if X_spec is not None:
    uniq_labels = sorted(np.unique(y_all))
    n_classes = len(uniq_labels)
    fig, axes = plt.subplots(n_classes, 1, figsize=(11, 4*n_classes))
    if n_classes == 1:
        axes = [axes]
    vmin, vmax = -80.0, 0.0
    for ax, lab in zip(axes, uniq_labels):
        idx = np.where(y_all == lab)[0]
        if len(idx) == 0:
            ax.axis('off')
            continue
        i0 = idx[0]
        spec = X_spec[i0]
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        title = inv_class_map.get(int(lab), f"Class {int(lab)}")
        ax.set_title(f"{title} Class", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Frequency Bins')
    
    fig.suptitle(f'Spectrograms per Class ({vmin:.0f} to {vmax:.0f} dB)', fontsize=14, fontweight='bold')
    plt.subplots_adjust(right=0.82, hspace=0.3)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02, label='Magnitude (dB)')
    plt.savefig('spectrograms_per_class.png', dpi=300)

if X_spec is not None:
    extractor_viz = FeatureExtractor(sr=TARGET_SR, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH, win_length=STFT_WIN_LENGTH)
    uniq_labels = sorted(np.unique(y_all))
    n_classes = len(uniq_labels)
    fig_cwt, axes_cwt = plt.subplots(n_classes, 1, figsize=(11, 4*n_classes))
    if n_classes == 1:
        axes_cwt = [axes_cwt]
    vmin_cwt, vmax_cwt = -80.0, 0.0
    for ax, lab in zip(axes_cwt, uniq_labels):
        idx = np.where(y_all == lab)[0]
        if len(idx) == 0:
            ax.axis('off')
            continue
        i0 = idx[0]
        class_name = inv_class_map.get(int(lab), f"Class {int(lab)}")
        class_dir = os.path.join(DATASET_PATH, class_name)
        
        if os.path.isdir(class_dir):
            sample_files = sorted(glob.glob(os.path.join(class_dir, "**", "*.wav"), recursive=True) + 
                                 glob.glob(os.path.join(class_dir, "**", "*.mp3"), recursive=True))
            if sample_files:
                cleaner_temp = AudioCleaner(sr=TARGET_SR, low_cut=LOW_CUT, high_cut=HIGH_CUT)
                raw = cleaner_temp.resample_audio(sample_files[0], target_sr=TARGET_SR)
                min_len = int(TARGET_SR * SEGMENT_DURATION)
                if len(raw) < min_len:
                    raw = np.pad(raw, (0, min_len - len(raw)))
                chunks = cleaner_temp.segment_audio(raw, duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP, class_name=class_name)
                if chunks:
                    filtered = cleaner_temp.apply_bandpass_filter(chunks[0])
                    normalized = cleaner_temp.normalize(filtered)
                    cwt_scalogram = extractor_viz.compute_cwt_scalogram_viz_only(normalized, wavelet='morl')
                    im_cwt = ax.imshow(cwt_scalogram, aspect='auto', origin='lower', cmap='magma', vmin=vmin_cwt, vmax=vmax_cwt)
                    ax.set_title(f"{class_name} Class", fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time Frames')
                    ax.set_ylabel('Scales (Frequency)')
    
    fig_cwt.suptitle(f'CWT Scalograms per Class - Morlet Wavelet ({vmin_cwt:.0f} to {vmax_cwt:.0f} dB)', 
                     fontsize=14, fontweight='bold')
    plt.subplots_adjust(right=0.82, hspace=0.3)
    cbar_cwt = fig_cwt.colorbar(im_cwt, ax=axes_cwt, orientation='vertical', fraction=0.05, pad=0.02, label='Magnitude (dB)')
    plt.savefig('cwt_scalograms_per_class.png', dpi=300)

if X_spec is not None:    
    uniq_labels = sorted(np.unique(y_all))
    n_classes = len(uniq_labels)
    fig_dwt, axes_dwt = plt.subplots(n_classes, 1, figsize=(11, 4*n_classes))
    if n_classes == 1:
        axes_dwt = [axes_dwt]
    
    vmin_dwt, vmax_dwt = -80.0, 0.0
    for ax, lab in zip(axes_dwt, uniq_labels):
        idx = np.where(y_all == lab)[0]
        if len(idx) == 0:
            ax.axis('off')
            continue
        i0 = idx[0]
        class_name = inv_class_map.get(int(lab), f"Class {int(lab)}")
        class_dir = os.path.join(DATASET_PATH, class_name)
        
        if os.path.isdir(class_dir):
            sample_files = sorted(glob.glob(os.path.join(class_dir, "**", "*.wav"), recursive=True) + 
                                 glob.glob(os.path.join(class_dir, "**", "*.mp3"), recursive=True))
            if sample_files:
                cleaner_temp = AudioCleaner(sr=TARGET_SR, low_cut=LOW_CUT, high_cut=HIGH_CUT)
                raw = cleaner_temp.resample_audio(sample_files[0], target_sr=TARGET_SR)
                min_len = int(TARGET_SR * SEGMENT_DURATION)
                if len(raw) < min_len:
                    raw = np.pad(raw, (0, min_len - len(raw)))
                chunks = cleaner_temp.segment_audio(raw, duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP, class_name=class_name)
                if chunks:
                    filtered = cleaner_temp.apply_bandpass_filter(chunks[0])
                    normalized = cleaner_temp.normalize(filtered)
                    
                    # Compute DWT scalogram
                    dwt_scalogram = extractor_viz.compute_dwt_scalogram(normalized, wavelet='db4', level=6)
                    
                    im_dwt = ax.imshow(dwt_scalogram, aspect='auto', origin='lower', cmap='plasma', vmin=vmin_dwt, vmax=vmax_dwt)
                    ax.set_title(f"{class_name} Class", fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time Frames')
                    ax.set_ylabel('Decomposition Levels')
    
    fig_dwt.suptitle(f'Discrete DWT Scalograms per Class - db4 Wavelet ({vmin_dwt:.0f} to {vmax_dwt:.0f} dB)', 
                     fontsize=14, fontweight='bold')
    plt.subplots_adjust(right=0.82, hspace=0.3)
    cbar_dwt = fig_dwt.colorbar(im_dwt, ax=axes_dwt, orientation='vertical', fraction=0.05, pad=0.02, label='Magnitude (dB)')
    plt.savefig('dwt_scalograms_per_class.png', dpi=300)


# 2) Mean Spectrogram Magnitude per Class
if X_spec is not None:
    colors = plt.cm.Set2.colors if len(np.unique(y_all)) <= 8 else plt.cm.tab20.colors
    freq_lim = 2000 
    fig, ax = plt.subplots(figsize=(12, 6))
    unique_labels = sorted(set(np.unique(y_all)))
    plotted_classes = set()
    
    for i, lab in enumerate(unique_labels):
        class_name = inv_class_map.get(int(lab), str(lab))
        if class_name in plotted_classes:
            continue
        plotted_classes.add(class_name)
        
        sel_idx = np.where(y_all == lab)[0]
        if len(sel_idx) == 0: 
            continue
        take = sel_idx[: min(200, len(sel_idx))]
        specs = X_spec[take]
        spec_mean = np.mean(specs, axis=(0, 2)) 
        freqs = np.linspace(0, TARGET_SR//2, spec_mean.shape[0])
        
        mask = freqs <= freq_lim
        ax.plot(freqs[mask], spec_mean[mask], label=class_name, color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, alpha=0.8)

    ax.set_title('Mean Spectrogram Magnitude per Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.savefig('Mean_spectrogram_magnitude_per_class.png', dpi=300)

# 3) Sequential Preprocessing Pipeline Visualization -  Sick-Varroa Sample 

preferred = None
for cname in ["Sick-Varroa"]:
    cdir = os.path.join(DATASET_PATH, cname)
    if os.path.isdir(cdir):
        cand = sorted(glob.glob(os.path.join(cdir, "**", "*.wav"), recursive=True) + 
                     glob.glob(os.path.join(cdir, "**", "*.mp3"), recursive=True))
        if cand:
            preferred = cand[0]
            selected_class = cname
            break

if preferred:
    cleaner = AudioCleaner(sr=TARGET_SR, low_cut=LOW_CUT, high_cut=HIGH_CUT)
    
    try:
        import soundfile as sf
        step0_original, original_sr = sf.read(preferred)
        if step0_original.ndim > 1:
            step0_original = np.mean(step0_original, axis=1)
    except:
        step0_original, original_sr = librosa.load(preferred, sr=None, mono=True)
    if original_sr != TARGET_SR:
        step1_resampled = librosa.resample(step0_original, orig_sr=original_sr, target_sr=TARGET_SR)
    else:
        step1_resampled = step0_original.copy()

    min_len = int(TARGET_SR * SEGMENT_DURATION)
    if len(step1_resampled) < min_len:
        step2_raw = np.pad(step1_resampled, (0, min_len - len(step1_resampled)))    
    else:
        step2_raw = step1_resampled.copy()  

    chunks = cleaner.segment_audio(step2_raw, duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP, class_name=selected_class)
    step3_segment = chunks[0]
    step4_filtered = cleaner.apply_bandpass_filter(step3_segment)
    step5_normalized = cleaner.normalize(step4_filtered)
    rms_before = np.sqrt(np.mean(step4_filtered**2))
    rms_after = np.sqrt(np.mean(step5_normalized**2))
    display_len = min_len
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))

    t_full = np.arange(min(len(step2_raw), int(TARGET_SR * 10))) / TARGET_SR  
    y_full = step2_raw[:len(t_full)]
    axes[0, 0].plot(t_full, y_full, color='navy', linewidth=0.6, alpha=0.8)
    axes[0, 0].set_title(f'BEFORE Segmentation', fontweight='bold', fontsize=10, color='navy', pad=10)
    axes[0, 0].set_ylabel('Amplitude', fontsize=9)
    axes[0, 0].set_xlabel('Time (s)', fontsize=9)

    t_segment = np.arange(len(step3_segment)) / TARGET_SR
    axes[0, 1].plot(t_segment, step3_segment, color='purple', linewidth=0.8)
    axes[0, 1].set_title(f'AFTER Segmentation', fontweight='bold', fontsize=10, color='purple', pad=10)
    axes[0, 1].set_ylabel('Amplitude', fontsize=9)
    axes[0, 1].set_xlabel('Time (s)', fontsize=9)
    

    axes[1, 0].plot(t_segment, step3_segment, color='purple', linewidth=0.8)
    axes[1, 0].set_title(f'BEFORE Bandpass Filter', fontweight='bold', fontsize=10, color='purple', pad=10)
    axes[1, 0].set_ylabel('Amplitude', fontsize=9)
    axes[1, 0].set_xlabel('Time (s)', fontsize=9)
    axes[1, 1].plot(t_segment, step4_filtered, color='blue', linewidth=0.8)
    axes[1, 1].set_title(f'AFTER Bandpass Filter ({LOW_CUT}-{HIGH_CUT} Hz)', fontweight='bold', fontsize=10, color='blue', pad=10)
    axes[1, 1].set_ylabel('Amplitude', fontsize=9)
    axes[1, 1].set_xlabel('Time (s)', fontsize=9)
    
    axes[2, 0].plot(t_segment, step4_filtered, color='blue', linewidth=0.8)
    axes[2, 0].set_title(f'BEFORE RMS Normalisation', fontweight='bold', fontsize=10, color='blue', pad=10)
    axes[2, 0].set_ylabel('Amplitude', fontsize=9)
    axes[2, 0].set_xlabel('Time (s)', fontsize=9)
    axes[2, 1].plot(t_segment, step5_normalized, color='green', linewidth=0.8)
    axes[2, 1].set_title(f'AFTER RMS Normalisation', fontweight='bold', fontsize=10, color='green', pad=10)
    axes[2, 1].set_ylabel('Amplitude', fontsize=9)
    axes[2, 1].set_xlabel('Time (s)', fontsize=9)

    freqs_seg = np.fft.rfftfreq(len(step3_segment), 1/TARGET_SR)
    fft_before = np.abs(np.fft.rfft(step3_segment))
    axes[3, 0].plot(freqs_seg, fft_before, color='purple', linewidth=1.2)
    axes[3, 0].set_title('Frequency Spectrum\nBefore Filtering (Raw Segment)', fontweight='bold', fontsize=10, color='purple', pad=10)
    axes[3, 0].set_xlabel('Frequency (Hz)', fontsize=9)
    axes[3, 0].set_ylabel('Magnitude', fontsize=9)
    axes[3, 0].set_xlim(0, 2000)
    fft_after = np.abs(np.fft.rfft(step5_normalized))
    axes[3, 1].plot(freqs_seg, fft_after, color='green', linewidth=1.2)
    axes[3, 1].set_title('Frequency Spectrum\nAfter Filtering', fontweight='bold', fontsize=10, color='green', pad=10)
    axes[3, 1].set_xlabel('Frequency (Hz)', fontsize=9)
    axes[3, 1].set_ylabel('Magnitude', fontsize=9)
    axes[3, 1].set_xlim(0, 2000)
    axes[3, 1].axvspan(LOW_CUT, HIGH_CUT, alpha=0.2, color='yellow', 
                      label=f'{LOW_CUT}-{HIGH_CUT} Hz')
    axes[3, 1].legend(fontsize=8, loc='upper right')
    
    for i in range(3):
        for j in range(2):
            axes[i, j].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig('Complete_preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    
    extractor_compare = FeatureExtractor(sr=TARGET_SR, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH, win_length=STFT_WIN_LENGTH)
    
    stft_spec = extractor_compare.compute_stft_spectrogram(step5_normalized)
    dwt_scalogram_db4 = extractor_compare.compute_dwt_scalogram(step5_normalized, wavelet='db4', level=6)
    dwt_scalogram_morl = extractor_compare.compute_cwt_scalogram_viz_only(step5_normalized, wavelet='morl')
    fig_compare, axes_compare = plt.subplots(2, 2, figsize=(16, 10))
    im1 = axes_compare[0, 0].imshow(stft_spec, aspect='auto', origin='lower', cmap='viridis')
    axes_compare[0, 0].set_title('STFT Spectrogram',fontweight='bold', fontsize=12, color='darkblue', pad=10)
    axes_compare[0, 0].set_ylabel('Frequency Bins (128)', fontsize=10)
    axes_compare[0, 0].set_xlabel('Time Frames (100)', fontsize=10)
    plt.colorbar(im1, ax=axes_compare[0, 0], fraction=0.046, pad=0.04, label='Magnitude (dB)')
    im2 = axes_compare[0, 1].imshow(dwt_scalogram_morl, aspect='auto', origin='lower', cmap='magma')

    axes_compare[0, 1].set_title('CWT Scalogram - Morlet', fontweight='bold', fontsize=12, color='darkred', pad=10)
    axes_compare[0, 1].set_ylabel('Scales (Low freq → High freq)', fontsize=10)
    axes_compare[0, 1].set_xlabel('Time Frames (100)', fontsize=10)
    plt.colorbar(im2, ax=axes_compare[0, 1], fraction=0.046, pad=0.04, label='Magnitude (dB)')
    im3 = axes_compare[1, 0].imshow(dwt_scalogram_db4, aspect='auto', origin='lower', cmap='plasma')
    
    axes_compare[1, 0].set_title('Discrete DWT Scalogram - db4',fontweight='bold', fontsize=12, color='purple', pad=10)
    axes_compare[1, 0].set_ylabel('Decomposition Levels', fontsize=10)
    axes_compare[1, 0].set_xlabel('Time Frames (100)', fontsize=10)
    plt.colorbar(im3, ax=axes_compare[1, 0], fraction=0.046, pad=0.04, label='Magnitude (dB)')    
    dwt_energies = extractor_compare.compute_discrete_dwt_features(step5_normalized, wavelet='db4', level=6)
    level_names = ['A6 (Low)', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1 (High)']
    colors_dwt = plt.cm.viridis(np.linspace(0, 1, len(dwt_energies)))
    
    axes_compare[1, 1].bar(range(len(dwt_energies)), dwt_energies, color=colors_dwt, edgecolor='black', linewidth=1.2)
    axes_compare[1, 1].set_title('Discrete DWT Energy Distribution (db4)', fontweight='bold', fontsize=12, color='darkgreen', pad=10)
    axes_compare[1, 1].set_ylabel('Normalised Energy', fontsize=10)
    axes_compare[1, 1].set_xlabel('Decomposition Level', fontsize=10)
    axes_compare[1, 1].set_xticks(range(len(dwt_energies)))
    axes_compare[1, 1].set_xticklabels(level_names, rotation=45, ha='right')
    axes_compare[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.95, bottom=0.12)

    plt.savefig('STFT_vs_DWT_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Discrete DWT energies: {dwt_energies}")
    print(f"  Dominant frequency band: {level_names[np.argmax(dwt_energies)]}")


# 4) Class Distribution: Pre vs Post Processing
def count_segments_for_file(wav_path, sr=TARGET_SR, duration=SEGMENT_DURATION, overlap=0.25):
    try:
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
    except: return 0
    n = int(duration * sr)
    if len(y) < n: return 0
    step = int(n * (1 - overlap))
    return 1 + max(0, (len(y) - n) // step)

pre_counts, post_counts = {}, {}
for cname, cid in class_map.items():
    files = sorted(glob.glob(os.path.join(DATASET_PATH, cname, "**", "*.*"), recursive=True))
    if not files: continue
    pre_counts[cname] = sum(count_segments_for_file(fp, overlap=0.25) for fp in files)
    dyn_ov = 0.0 if cname == 'Active' else 0.90
    post_counts[cname] = sum(count_segments_for_file(fp, overlap=dyn_ov) for fp in files)

cats = list(pre_counts.keys())
pre = [pre_counts.get(c,0) for c in cats]
post = [post_counts.get(c,0) for c in cats]
x = np.arange(len(cats))
width = 0.35

if cats:
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, pre, width, label='Initial')
    plt.bar(x + width/2, post, width, label='Final')
    plt.xticks(x, cats)
    plt.title('Class Distribution')
    plt.legend()
    plt.savefig('Class_distribution_pre_post_processing.png', dpi=300)

# 5) Spectral Feature Boxplots
if X_feat is not None:
    import pandas as pd
    lab_names = [inv_class_map.get(int(l), str(int(l))) for l in y_all]
    df = pd.DataFrame({
        'centroid': X_feat[:, 0], 'ber': X_feat[:, 1], 'zcr': X_feat[:, 2], 'label': lab_names
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 13))
    sns.boxplot(data=df, x='label', y='centroid', ax=axes[0])
    axes[0].set_title('Centroid')
    
    sns.boxplot(data=df, x='label', y='ber', ax=axes[1])
    axes[1].set_title('Band Energy Ratio')

    sns.boxplot(data=df, x='label', y='zcr', ax=axes[2])
    axes[2].set_title('ZCR')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('Spectral_feature_boxplots.png', dpi=300)



