import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from fusion_model_modular import CLASS_NAMES
from extract_features_and_predict import (
    preprocess_audio,
    compute_stft_spectrogram,
    compute_dwt_scalogram,
    extract_70_features,
    load_student_mlp,
    load_scaler,
    scale_features,
)
RESULTS_PATH = "../results"
MODEL_PATH   = os.path.join(RESULTS_PATH, "student_ds-cnn_best.keras")
SCALER_MEAN  = os.path.join(RESULTS_PATH, "scaler_mean.npy")
SCALER_STD   = os.path.join(RESULTS_PATH, "scaler_std.npy")

MISSING_QUEEN_IDX = 1
ACTIVE_IDX = 0
BINARY_LABELS = ["QUEEN PRESENT", "NO QUEEN"]
ACTIVE_THRESHOLD = 0.55
AUDIO_FILE = "CrossData/Queenless/Hive1 31_05_2018_NO_QueenBee____00_00_00_chunk25.wav"

def predict_single_file(model, scaler_mean, scaler_std, audio_path):
    audio = preprocess_audio(audio_path)
    stft_spec = compute_stft_spectrogram(audio)
    dwt_spec  = compute_dwt_scalogram(audio)
    features  = extract_70_features(stft_spec, dwt_spec)
    features_scaled = scale_features(features, scaler_mean, scaler_std)
    features_input  = features_scaled.reshape(1, -1)
    probs = model.predict(features_input, verbose=0)[0]
    pred_6 = int(np.argmax(probs))
    active_prob = float(probs[ACTIVE_IDX])
    binary = 0 if active_prob >= ACTIVE_THRESHOLD and pred_6 != MISSING_QUEEN_IDX else 1

    return {
        "class_6":       pred_6,
        "class_6_name":  CLASS_NAMES[pred_6],
        "probabilities": probs,
        "binary_label":  BINARY_LABELS[binary],
        "confidence":    float(probs[pred_6]) * 100,
        "active_prob":   active_prob * 100,
    }

def test_single_file(audio_path):
    print(f"  Audio : {audio_path}")
    model = load_student_mlp(MODEL_PATH)
    scaler_mean, scaler_std = load_scaler(SCALER_MEAN, SCALER_STD)
    result = predict_single_file(model, scaler_mean, scaler_std, audio_path)
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, result["probabilities"])):
        print(f"    [{i}] {name:20s}: {prob*100:6.2f}%")
    print(f"\n  Predicted 6-class : {result['class_6_name']}  "
          f"({result['confidence']:.1f}% confidence)")
    print(f"  P(Active) = {result['active_prob']:.1f}%")
    print(f"  Queen/Queenless? : {result['binary_label']}")









def main():
    if not os.path.isfile(AUDIO_FILE):
        print(f"No such file: {AUDIO_FILE}")
        sys.exit(1)
    test_single_file(AUDIO_FILE)


if __name__ == "__main__":
    main()
