import os
import sys
import numpy as np
from shap.plots import bar
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')
from fusion_model_modular import (
    extract_handcrafted_features,
    CLASS_NAMES
)
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.config.set_visible_devices([], 'GPU')
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Model loading will be disabled.")
    TF_AVAILABLE = False
import librosa
import pywt
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom


# Audio file path
DEFAULT_AUDIO_FILE = "/home/sachcith/Documents/Sem 4/IOT/Project/MissingQueen.wav"
TARGET_SR = 16000          
LOW_CUT = 100              
HIGH_CUT = 1000            
SEGMENT_DURATION = 2.0     
STFT_N_FFT = 1024          
STFT_HOP_LENGTH = 256      
STFT_WIN_LENGTH = 1024     
SPEC_SHAPE = (128, 100)    
DWT_WAVELET = 'db4'        
DWT_LEVEL = 6              

def resample_audio(file_path, target_sr=TARGET_SR):
    audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
    return audio

def apply_bandpass_filter(data, sr=TARGET_SR, low_cut=LOW_CUT, high_cut=HIGH_CUT, order=5):
    nyq = 0.5 * sr
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_audio(data):
    rms = np.sqrt(np.mean(data**2))
    return data / rms if rms > 0 else data

def preprocess_audio(file_path, target_sr=TARGET_SR):
    audio = resample_audio(file_path, target_sr)
    min_len = int(target_sr * SEGMENT_DURATION)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    audio = audio[:min_len]
    audio_filtered = apply_bandpass_filter(audio, target_sr)
    audio_normalized = normalize_audio(audio_filtered)
    return audio_normalized


def compute_stft_spectrogram(audio_segment, sr=TARGET_SR, n_fft=STFT_N_FFT, 
                            hop_length=STFT_HOP_LENGTH, win_length=STFT_WIN_LENGTH,
                            target_shape=SPEC_SHAPE):
    stft_result = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length, 
                               win_length=win_length)
    magnitude = np.abs(stft_result)
    log_spectrogram = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
    freq_factor = target_shape[0] / log_spectrogram.shape[0]
    time_factor = target_shape[1] / log_spectrogram.shape[1]
    reshaped_spec = zoom(log_spectrogram, (freq_factor, time_factor), order=1)
    reshaped_spec = reshaped_spec[:target_shape[0], :target_shape[1]]
    if reshaped_spec.shape != target_shape:
        padded = np.zeros(target_shape, dtype=np.float32)
        padded[:reshaped_spec.shape[0], :reshaped_spec.shape[1]] = reshaped_spec
        reshaped_spec = padded
    spec_mean = np.mean(reshaped_spec)
    spec_std = np.std(reshaped_spec) + 1e-8
    reshaped_spec = (reshaped_spec - spec_mean) / spec_std
    
    return reshaped_spec.astype(np.float32)

def compute_dwt_scalogram(audio_segment, wavelet=DWT_WAVELET, level=DWT_LEVEL, 
                         shape=SPEC_SHAPE):
    coeffs = pywt.wavedec(audio_segment, wavelet, level=level)
    scalogram_list = []
    for coeff in coeffs:
        scalogram_list.append(coeff)
    
    max_len = max(len(c) for c in scalogram_list)
    scalogram_raw = np.zeros((len(scalogram_list), max_len), dtype=np.float32)
    for i, coeff in enumerate(scalogram_list):
        scalogram_raw[i, :len(coeff)] = coeff[:]
    
    # Convert to dB scale
    magnitude = np.abs(scalogram_raw)
    scalogram_db = librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)
    
    # Resize to target shape
    if scalogram_db.shape != shape:
        zoom_factor = (shape[0] / scalogram_db.shape[0], shape[1] / scalogram_db.shape[1])
        scalogram_db = zoom(scalogram_db, zoom_factor, order=1)
    
    return scalogram_db[:shape[0], :shape[1]].astype(np.float32)


def extract_70_features(stft_spec, dwt_spec):
    X_stft = stft_spec[np.newaxis, :, :]  
    X_dwt = dwt_spec[np.newaxis, :, :]    
    features_batch = extract_handcrafted_features(X_stft, X_dwt)
    return features_batch[0]

def load_student_mlp(model_path):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to load the model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model: {e}")

def load_scaler(scaler_mean_path, scaler_std_path):
    if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_std_path):
        raise FileNotFoundError(f"Scaler files not found")
    mean = np.load(scaler_mean_path)
    std = np.load(scaler_std_path)
    return mean, std

def scale_features(features, mean, std):
    return (features - mean) / (std + 1e-8)

def predict(model, features, scaler_mean=None, scaler_std=None):
    if scaler_mean is not None and scaler_std is not None:
        features_scaled = scale_features(features, scaler_mean, scaler_std)
    else:
        features_scaled = features
    features_input = features_scaled.reshape(1, -1)
    predictions = model.predict(features_input, verbose=0)
    return predictions[0]

def display_results(predictions, features):
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100
    print(f"Confidence: {confidence:.2f}%")
    print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    print("Model Prediction:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        print(f"  [{i}] {class_name:20s}: {prob*100:6.2f}%")

def fuse_dense_batchnorm(model):
    """
    Fold BatchNormalization into preceding Dense layers.
    
    BN computes: y = gamma * (x - moving_mean) / sqrt(moving_variance + eps) + beta
    Dense computes: x = W * input + b
    
    Fused:
        scale = gamma / sqrt(moving_variance + eps)
        W_fused = W * scale
        b_fused = (b - moving_mean) * scale + beta
    """
    fused_layers = []
    layers = model.layers
    i = 0
    
    while i < len(layers):
        layer = layers[i]
        weights = layer.get_weights()
        
        # Check if this Dense layer is followed by BatchNormalization
        if 'dense' in layer.name or 'hidden' in layer.name or 'output' in layer.name:
            if len(weights) >= 2:
                W = weights[0]  # (in, out)
                b = weights[1]  # (out,)
                
                # Look ahead for BatchNorm
                if i + 1 < len(layers) and 'batch_normalization' in layers[i + 1].name:
                    bn = layers[i + 1]
                    bn_weights = bn.get_weights()
                    gamma = bn_weights[0]          # scale
                    beta = bn_weights[1]           # offset
                    moving_mean = bn_weights[2]    # moving mean
                    moving_var = bn_weights[3]     # moving variance
                    eps = bn.epsilon if hasattr(bn, 'epsilon') else 1e-3
                    
                    # Fuse: scale = gamma / sqrt(var + eps)
                    scale = gamma / np.sqrt(moving_var + eps)
                    
                    # W_fused = W * scale (broadcast across output dim)
                    W_fused = W * scale[np.newaxis, :]
                    
                    # b_fused = (b - moving_mean) * scale + beta
                    b_fused = (b - moving_mean) * scale + beta
                    
                    fused_layers.append({
                        'name': layer.name,
                        'weights': W_fused.astype(np.float32),
                        'bias': b_fused.astype(np.float32),
                        'fused_with': bn.name
                    })
                    i += 2  # Skip the BN layer
                    continue
                else:
                    # Dense without BN after it
                    fused_layers.append({
                        'name': layer.name,
                        'weights': W.astype(np.float32),
                        'bias': b.astype(np.float32),
                        'fused_with': None
                    })
        i += 1
    
    return fused_layers

def export_weights_as_cc(model, output_path):
    """
    Export model weights as a C++ file (.cc) for ESP32 deployment.
    BatchNormalization is folded into Dense layers so only weights + bias remain.
    """
    # Fuse BN into Dense layers
    fused_layers = fuse_dense_batchnorm(model)
    
    cc_content = """// Auto-generated MLP model weights (BatchNorm folded into Dense)
// Distilled Student Model - ESP32 deployment
// Only weights and bias per layer (BN fused)

#ifndef MLP_MODEL_WEIGHTS_H
#define MLP_MODEL_WEIGHTS_H

#include <cstddef>

namespace mlp_weights {

"""
    
    export_stats = []
    
    for layer_info in fused_layers:
        layer_name = layer_info['name'].replace('/', '_').replace(' ', '_').replace('-', '_')
        W = layer_info['weights']
        b = layer_info['bias']
        fused_with = layer_info['fused_with']
        
        fused_tag = f" (fused with {fused_with})" if fused_with else ""
        
        # Write weights
        w_flat = W.flatten()
        cc_content += f"// {layer_name}{fused_tag}\n"
        cc_content += f"// Weights shape: {W.shape}\n"
        cc_content += f"static constexpr float {layer_name}_weights[] = {{\n"
        for i in range(0, len(w_flat), 8):
            chunk = w_flat[i:i+8]
            cc_content += "    " + ", ".join([f"{v:.8f}f" for v in chunk]) + ",\n"
        cc_content += "};\n"
        cc_content += f"static constexpr size_t {layer_name}_weights_rows = {W.shape[0]};\n"
        cc_content += f"static constexpr size_t {layer_name}_weights_cols = {W.shape[1]};\n\n"
        
        # Write bias
        cc_content += f"// {layer_name} bias\n"
        cc_content += f"static constexpr float {layer_name}_bias[] = {{\n"
        cc_content += "    " + ", ".join([f"{v:.8f}f" for v in b]) + "\n"
        cc_content += "};\n"
        cc_content += f"static constexpr size_t {layer_name}_bias_size = {len(b)};\n\n"
        
        export_stats.append({
            'name': layer_name,
            'fused_with': fused_with,
            'w_shape': W.shape,
            'b_shape': b.shape,
            'total': W.size + b.size
        })
    
    # Add metadata
    num_layers = len(fused_layers)
    cc_content += f"""// Model Metadata
static constexpr size_t NUM_LAYERS = {num_layers};
static constexpr size_t INPUT_SIZE = {fused_layers[0]['weights'].shape[0]};
static constexpr size_t OUTPUT_SIZE = {fused_layers[-1]['weights'].shape[1]};
static constexpr const char* CLASS_NAMES[] = {{
    "{CLASS_NAMES[0]}",
    "{CLASS_NAMES[1]}",
    "{CLASS_NAMES[2]}",
    "{CLASS_NAMES[3]}",
    "{CLASS_NAMES[4]}",
    "{CLASS_NAMES[5]}"
}};

}} // namespace mlp_weights

#endif // MLP_MODEL_WEIGHTS_H
"""
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(cc_content)
    
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"✓ Model weights exported to C++: {output_path}")
    print(f"  File size: {file_size_kb:.2f} KB")
    print(f"  Layers (BN folded into Dense): {num_layers}")
    
    print(f"\n✓ Fused layer summary:")
    total_params = 0
    for s in export_stats:
        fused_str = f" ← fused with {s['fused_with']}" if s['fused_with'] else ""
        print(f"  {s['name']:20s}: W{s['w_shape']} + b{s['b_shape']} = {s['total']:6d} params{fused_str}")
        total_params += s['total']
    print(f"\n✓ Total fused parameters: {total_params:,}")
    print(f"  (Original model had {model.count_params():,} params including BN)")
    
def main():
    audio_file = DEFAULT_AUDIO_FILE
    model_path = "../results/student_ds-cnn_best.keras"
    scaler_mean = "../results/scaler_mean.npy"
    scaler_std = "../results/scaler_std.npy"
    print(f"\nAudio file: {audio_file}")
    output_dir = os.path.join("../results/esp", os.path.splitext(os.path.basename(audio_file))[0])
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    audio = preprocess_audio(audio_file)
    stft_spec = compute_stft_spectrogram(audio)
    dwt_spec = compute_dwt_scalogram(audio)
    features = extract_70_features(stft_spec, dwt_spec)
    audio_output_path = os.path.join(output_dir, f"{base_name}_preprocessed.wav")
    sf.write(audio_output_path, audio, TARGET_SR)
    audio_original_path = os.path.join(output_dir, f"{base_name}_original.wav")
    original_audio = resample_audio(DEFAULT_AUDIO_FILE, TARGET_SR)
    min_len = int(TARGET_SR * SEGMENT_DURATION)
    if len(original_audio) < min_len:
        original_audio = np.pad(original_audio, (0, min_len - len(original_audio)))
    original_audio = original_audio[:min_len]
    sf.write(audio_original_path, original_audio, TARGET_SR)
    predictions = None
    predicted_class = None
    confidence = None
    
    if TF_AVAILABLE:
        try:
            model = load_student_mlp(model_path)
            scaler_mean_val, scaler_std_val = None, None
            if os.path.exists(scaler_mean) and os.path.exists(scaler_std):
                scaler_mean_val, scaler_std_val = load_scaler(scaler_mean, scaler_std)
            predictions = predict(model, features, scaler_mean_val, scaler_std_val)
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            display_results(predictions, features)
            '''
            model_output_path = os.path.join(output_dir, "student_mlp_model.keras")
            model.save(model_output_path)
            model_json_path = os.path.join(output_dir, "student_mlp_architecture.json")
            model_json = model.to_json()
            with open(model_json_path, 'w') as f:
                f.write(model_json)
            weights_dir = os.path.join(output_dir, "model_weights")
            os.makedirs(weights_dir, exist_ok=True)
            weights_info = []
            for layer in model.layers:
                layer_weights = layer.get_weights()
                if len(layer_weights) > 0:
                    layer_name = layer.name.replace('/', '_').replace(' ', '_')
                    if len(layer_weights) > 0:
                        weights_path = os.path.join(weights_dir, f"{layer_name}_weights.npy")
                        np.save(weights_path, layer_weights[0].astype(np.float32))
                        weights_info.append(f"  {layer_name}: {layer_weights[0].shape}")
                    if len(layer_weights) > 1:
                        bias_path = os.path.join(weights_dir, f"{layer_name}_bias.npy")
                        np.save(bias_path, layer_weights[1].astype(np.float32))
                        weights_info.append(f"  {layer_name}_bias: {layer_weights[1].shape}")


            
            # Export weights as C++ file for ESP32
            cc_weights_path = os.path.join(output_dir, "mlp_model_weights.cc")
            export_weights_as_cc(model, cc_weights_path)
            
            if scaler_mean_val is not None and scaler_std_val is not None:
                scaler_mean_path = os.path.join(output_dir, "scaler_mean.npy")
                scaler_std_path = os.path.join(output_dir, "scaler_std.npy")
                np.save(scaler_mean_path, scaler_mean_val)
                np.save(scaler_std_path, scaler_std_val)
            '''
            # Save prediction results to ESP folder
            prediction_path = os.path.join(output_dir, f"{base_name}_prediction.txt")
            with open(prediction_path, 'w') as f:
                f.write(f"Predicted Class: {CLASS_NAMES[predicted_class]}\n")
                f.write(f"Confidence: {confidence:.2f}%\n\n")
                f.write(f"Class Probabilities:\n")
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                    f.write(f"  [{i}] {class_name:20s}: {prob*100:6.2f}%\n")
                f.write(f"\nModel Information:\n")
                f.write(f"  Total Parameters: {model.count_params():,}\n")
                f.write(f"  Input Shape: (70,)\n")
                f.write(f"  Output Classes: 6\n")
                f.write(f"\nLayer Details:\n")
                for layer in model.layers:
                    if hasattr(layer, 'output_shape'):
                        f.write(f"  - {layer.name}: {layer.output_shape}\n")
                    else:
                        f.write(f"  - {layer.name}: InputLayer\n")
    
                print(" Completed !!!")
            
        except Exception as e:
            print(f"\nError during prediction: {e}")
    else:
        print("\nTensorFlow not available.")

if __name__ == "__main__":
    main()
