import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XGBOOST_USE_CUDA"] = "0"

DATA_PATH = "../dataset_processed"
RESULTS_PATH = "../results"
os.makedirs(RESULTS_PATH, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
USE_SAVED_BALANCED_DATA = True 
OVERSAMPLE_STRATEGY = 'SMOTE'  


def load_data():    
    X_spec = np.load(os.path.join(DATA_PATH, "X_train_spec.npy"))  # (n_samples, 128, 100)
    y = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    unique_labels = sorted(np.unique(y))
    class_map = {f"Class_{i}": i for i in unique_labels}
    
    inv_class_map = {v: k for k, v in class_map.items()}
    print(f"Data shape: {X_spec.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:")
    for label in sorted(np.unique(y)):
        count = np.sum(y == label)
        class_name = inv_class_map.get(label, f"Unknown_{label}")
        print(f"  {class_name}: {count} samples ({100*count/len(y):.2f}%)")
    
    return X_spec, y, class_map, inv_class_map


def balance_dataset(X, y):
    print("Original class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({100*count/len(y):.2f}%)")
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    usable_counts = [count for count in class_counts.values() if count >= 300]
    min_class_size = min(usable_counts) if usable_counts else 300
    samples_per_class = min(min_class_size, 3000)
    
    balanced_indices = []
    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        n_samples = min(len(class_indices), samples_per_class) 
        if len(class_indices) >= samples_per_class:
            selected = np.random.choice(class_indices, size=samples_per_class, replace=False)
        else:
            selected = np.random.choice(class_indices, size=samples_per_class, replace=True)
        
        balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print("\nBalanced class distribution:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({100*count/len(y_balanced):.2f}%)")
    
    print(f"\nDataset size: {len(y)} → {len(y_balanced)} samples")
    return X_balanced, y_balanced


def compute_class_weights_dict(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    
    print("Computed class weights:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    return class_weight_dict

def extract_statistical_features_from_stft(X_spec):
    """
    Features extracted:
        - Spectral energy statistics (mean, std, max, min per frequency bin)
        - Spectral centroid (center of mass of spectrum)
        - Spectral bandwidth (spread of spectrum)
        - Spectral entropy (uncertainty in frequency distribution)
        - Temporal variability (energy variation over time)
        - Frequency band energies (low, mid, high)
    """
    n_samples = X_spec.shape[0]
    n_freq_bins = X_spec.shape[1]
    n_time_frames = X_spec.shape[2]
    
    features_list = []
    
    for i in range(n_samples):
        spec = X_spec[i] 
        spec_linear = 10 ** (spec / 20.0)
        freq_energies = np.sum(spec_linear ** 2, axis=1) 
        energy_mean = np.mean(freq_energies)
        energy_std = np.std(freq_energies)
        energy_max = np.max(freq_energies)
        energy_min = np.min(freq_energies)
        energy_median = np.median(freq_energies)
        energy_q25 = np.percentile(freq_energies, 25)
        energy_q75 = np.percentile(freq_energies, 75)
        freq_bins = np.arange(n_freq_bins)
        spectral_centroid = np.sum(freq_bins[:, np.newaxis] * spec_linear, axis=0) / (np.sum(spec_linear, axis=0) + 1e-10)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)
        bandwidth = np.sqrt(np.sum(((freq_bins[:, np.newaxis] - spectral_centroid) ** 2) * spec_linear, axis=0) / (np.sum(spec_linear, axis=0) + 1e-10))
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        spec_prob = spec_linear / (np.sum(spec_linear, axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(spec_prob * np.log2(spec_prob + 1e-10), axis=0)
        entropy_mean = np.mean(entropy)
        entropy_std = np.std(entropy)
        time_energies = np.sum(spec_linear ** 2, axis=0) 
        temporal_mean = np.mean(time_energies)
        temporal_std = np.std(time_energies)
        temporal_range = np.max(time_energies) - np.min(time_energies)
        temporal_diff = np.diff(time_energies)
        temporal_diff_mean = np.mean(np.abs(temporal_diff))
        temporal_diff_std = np.std(temporal_diff)
        low_band = spec_linear[:n_freq_bins//3, :]
        mid_band = spec_linear[n_freq_bins//3:2*n_freq_bins//3, :]
        high_band = spec_linear[2*n_freq_bins//3:, :]      
        low_energy = np.mean(np.sum(low_band ** 2, axis=0))
        mid_energy = np.mean(np.sum(mid_band ** 2, axis=0))
        high_energy = np.mean(np.sum(high_band ** 2, axis=0)) 
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        cumsum_energy = np.cumsum(freq_energies)
        rolloff_threshold = 0.85 * cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= rolloff_threshold)[0]
        rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else n_freq_bins - 1
        geometric_mean = np.exp(np.mean(np.log(spec_linear + 1e-10), axis=0))
        arithmetic_mean = np.mean(spec_linear, axis=0)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        flatness_mean = np.mean(flatness)
        flatness_std = np.std(flatness)
        feature_vector = [
            energy_mean, energy_std, energy_max, energy_min, energy_median, energy_q25, energy_q75,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            entropy_mean, entropy_std,
            temporal_mean, temporal_std, temporal_range, temporal_diff_mean, temporal_diff_std,
            low_energy, mid_energy, high_energy, low_ratio, mid_ratio, high_ratio,
            rolloff, flatness_mean, flatness_std
        ]
        
        features_list.append(feature_vector)
        if (i+1) % 500 == 0 or (i+1) == n_samples:
            print(f"  Processed {i+1}/{n_samples} samples", end='\r')
    
    features = np.array(features_list, dtype=np.float32)
    print(f"Extracted features shape: {features.shape}")
    print(f"Total features per sample: {features.shape[1]}")
    
    return features


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_deep_models(X_train, X_test, y_train, y_test, num_classes, class_names, class_weights_dict=None):
    print(f"Original range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    X_train_linear = 10 ** (X_train / 20.0)
    X_test_linear = 10 ** (X_test / 20.0)
    train_min, train_max = X_train_linear.min(), X_train_linear.max()
    X_train_norm = (X_train_linear - train_min) / (train_max - train_min + 1e-8)
    X_test_norm = (X_test_linear - train_min) / (train_max - train_min + 1e-8)
    
    X_train_cnn = np.expand_dims(X_train_norm, axis=-1)  # (n, 128, 100, 1)
    X_test_cnn = np.expand_dims(X_test_norm, axis=-1)
    
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass weights for CNN:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls} ({class_names[cls]}): {weight:.2f}")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False) 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    results = {}
    cnn_model = build_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]), num_classes)
    cnn_checkpoint = ModelCheckpoint(os.path.join(RESULTS_PATH, 'cnn_best.h5'), 
                                     save_best_only=True, monitor='val_accuracy')
    
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train_cat,
        validation_split=0.2,  
        epochs=50,  
        batch_size=32,  
        callbacks=[early_stop, reduce_lr, cnn_checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn, batch_size=64, verbose=0), axis=1)
    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    cnn_f1 = f1_score(y_test, y_pred_cnn, average='weighted')
    

    pred_unique, pred_counts = np.unique(y_pred_cnn, return_counts=True)
    for label, count in zip(pred_unique, pred_counts):
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        print(f"  Predicted {class_name}: {count} samples ({100*count/len(y_pred_cnn):.2f}%)") 
    
    print("\nActual test set distribution:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    for label, count in zip(test_unique, test_counts):
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        print(f"  Actual {class_name}: {count} samples ({100*count/len(y_test):.2f}%)") 

    print(f"\nCNN Test Accuracy: {cnn_acc:.4f}")
    print(f"CNN Test F1-Score: {cnn_f1:.4f}")
    print("\nCNN Classification Report:")
    print(classification_report(y_test, y_pred_cnn, target_names=class_names))
    
    results['CNN'] = {
        'model': cnn_model,
        'history': cnn_history,
        'accuracy': cnn_acc,
        'f1_score': cnn_f1,
        'predictions': y_pred_cnn
    }
    return results

def train_classical_models(X_train_feat, X_test_feat, y_train, y_test, class_names, class_weights_dict=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    sample_weights = None
    if class_weights_dict:
        sample_weights = np.array([class_weights_dict[label] for label in y_train])
    
    results = {}

    # SVM
    svm_model = SVC(kernel='rbf', C=100, gamma='auto', random_state=RANDOM_STATE, 
                    class_weight='balanced' if class_weights_dict else None,
                    decision_function_shape='ovr', cache_size=500)
    if sample_weights is not None:
        svm_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')
    print(f"SVM Test Accuracy: {svm_acc:.4f}")
    print(f"SVM Test F1-Score: {svm_f1:.4f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=class_names))
    
    results['SVM'] = {
        'model': svm_model,
        'scaler': scaler,
        'accuracy': svm_acc,
        'f1_score': svm_f1,
        'predictions': y_pred_svm
    }
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=3,
                                      min_samples_leaf=1, max_features='sqrt',
                                      random_state=RANDOM_STATE, n_jobs=-1, 
                                      class_weight='balanced' if class_weights_dict else None,
                                      bootstrap=True, oob_score=True)
    if sample_weights is not None:
        rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    print(f"RF Test Accuracy: {rf_acc:.4f}")
    print(f"RF Test F1-Score: {rf_f1:.4f}")
    print("\nRF Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    
    results['RandomForest'] = {
        'model': rf_model,
        'scaler': scaler,
        'accuracy': rf_acc,
        'f1_score': rf_f1,
        'predictions': y_pred_rf
    }
    
    # XGBoost
    xgb_params = dict(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
        predictor="cpu_predictor",
        eval_metric="mlogloss"
    )

    if class_weights_dict:
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train)
    
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb, average='weighted')
    
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    print(f"XGBoost Test F1-Score: {xgb_f1:.4f}")
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=class_names))
    results['XGBoost'] = {
        'model': xgb_model,
        'scaler': scaler,
        'accuracy': xgb_acc,
        'f1_score': xgb_f1,
        'predictions': y_pred_xgb
    }
    
    return results


def plot_results(deep_results, classical_results, y_test, class_names):
    models = list(deep_results.keys()) + list(classical_results.keys())
    accuracies = [deep_results[m]['accuracy'] for m in deep_results.keys()] + \
                 [classical_results[m]['accuracy'] for m in classical_results.keys()]
    f1_scores = [deep_results[m]['f1_score'] for m in deep_results.keys()] + \
                [classical_results[m]['f1_score'] for m in classical_results.keys()]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - STFT Features Only', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'model_comparison.png'), dpi=300)
    plt.close()

    # Confusion Matrices
    all_results = {**deep_results, **classical_results}
    n_models = len(all_results)
    
    n_rows = (n_models + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, result) in enumerate(all_results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=class_names, yticklabels=class_names, cbar=True)
        axes[idx].set_title(f'{model_name}\nAcc: {result["accuracy"]:.3f}, F1: {result["f1_score"]:.3f}',
                           fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # Deep Learning Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, result in deep_results.items():
        history = result['history']
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label=f'{model_name} Train')
        axes[0].plot(history.history['val_accuracy'], label=f'{model_name} Val', linestyle='--')
        
        # Loss
        axes[1].plot(history.history['loss'], label=f'{model_name} Train')
        axes[1].plot(history.history['val_loss'], label=f'{model_name} Val', linestyle='--')
    
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'deep_learning_training_history.png'), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {RESULTS_PATH}/")


def main():
    X_spec, y, class_map, inv_class_map = load_data()
    num_classes = len(np.unique(y))
    class_names = [inv_class_map[i] for i in range(num_classes)]
    balanced_spec_path = os.path.join(DATA_PATH, 'X_train_spec_balanced.npy')
    balanced_y_path = os.path.join(DATA_PATH, 'y_train_balanced.npy')
    test_spec_path = os.path.join(DATA_PATH, 'X_test_spec.npy')
    test_y_path = os.path.join(DATA_PATH, 'y_test.npy')
    
    if USE_SAVED_BALANCED_DATA and os.path.exists(balanced_spec_path) and os.path.exists(test_spec_path):
        X_train_balanced = np.load(balanced_spec_path)
        y_train_balanced = np.load(balanced_y_path)
        X_test = np.load(test_spec_path)
        y_test = np.load(test_y_path)
        
        print(f"Loaded train set: {X_train_balanced.shape[0]} samples (balanced)")
        print(f"Loaded test set: {X_test.shape[0]} samples")
        print("\nBalanced class distribution:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label} ({class_names[label]}): {count} samples ({100*count/len(y_train_balanced):.2f}%)")
    else:
        print("Balanced data not found.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_spec, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\nOriginal train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
        np.save(balanced_spec_path, X_train_balanced.astype(np.float16))
        np.save(balanced_y_path, y_train_balanced)
        np.save(test_spec_path, X_test.astype(np.float16))
        np.save(test_y_path, y_test)
        print("Balanced data saved successfully!")
    
    print(f"\nFinal train set: {X_train_balanced.shape[0]} samples\n")
    print(f"Original test set: {X_test.shape[0]} samples")
    X_test_balanced, y_test_balanced = balance_dataset(X_test, y_test)
    print(f"Test set final: {X_test_balanced.shape[0]} samples\n")

    class_weights_dict = None
    
    # Extract statistical features
    X_train_feat = extract_statistical_features_from_stft(X_train_balanced)
    X_test_feat = extract_statistical_features_from_stft(X_test_balanced)
    
    # Train models
    deep_results = train_deep_models(X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, 
                                     num_classes, class_names, class_weights_dict)
    classical_results = train_classical_models(X_train_feat, X_test_feat, y_train_balanced, y_test_balanced, 
                                               class_names, class_weights_dict)
    plot_results(deep_results, classical_results, y_test_balanced, class_names)
    
    all_results = {**deep_results, **classical_results}
    for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_name:<15} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f}")



if __name__ == "__main__":
    main()
