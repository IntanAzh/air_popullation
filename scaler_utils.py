import os
import json
import numpy as np
import joblib

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"

def load_scaler_and_meta(tensor_dir=TENSOR_DIR):
    """
    Loads fitted MinMaxScaler object and metadata for inverse transformations.
    """
    scaler_path = os.path.join(tensor_dir, "scaler.joblib")
    meta_path = os.path.join(tensor_dir, "tensor_metadata.json")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    return scaler, meta

def inverse_transform_targets(y_scaled, scaler, target_indices):
    """
    Inverse-transforms model predictions or true target values from [0, 1] scale back to original physical units (ug/m3).
    
    Parameters:
    - y_scaled: numpy array of shape (samples, horizon, n_targets) or (samples, n_targets)
    - scaler: Fitted MinMaxScaler instance
    - target_indices: List of column indices in feature space corresponding to target variables (e.g. PM2.5 & PM10)
    
    Returns:
    - y_unscaled: numpy array in original physical scale with same shape as y_scaled.
    """
    y_scaled = np.array(y_scaled)
    orig_shape = y_scaled.shape

    # Reshape 3D (samples, horizon, n_targets) into 2D (samples * horizon, n_targets)
    if len(orig_shape) == 3:
        n_samples, horizon, n_targets = orig_shape
        y_flat = y_scaled.reshape(-1, n_targets)
    elif len(orig_shape) == 2:
        n_samples, n_targets = orig_shape
        y_flat = y_scaled
        horizon = 1
    else:
        raise ValueError(f"Unsupported shape {orig_shape}. Expected 2D or 3D tensor.")

    # Create dummy matrix matching scaler's full feature count (59 features)
    dummy = np.zeros((len(y_flat), len(scaler.data_min_)))
    
    # Place target predictions into corresponding feature column locations
    for i, idx in enumerate(target_indices):
        dummy[:, idx] = y_flat[:, i]
        
    # Apply inverse transform
    unscaled_full = scaler.inverse_transform(dummy)
    
    # Extract unscaled target columns
    unscaled_targets = unscaled_full[:, target_indices]
    
    # Reshape back to original tensor dimensions
    if len(orig_shape) == 3:
        return unscaled_targets.reshape(n_samples, horizon, n_targets)
    else:
        return unscaled_targets

if __name__ == "__main__":
    print("Testing Scaler Inverse Transformation...")
    scaler, meta = load_scaler_and_meta()
    
    # Load scaled y_test sample
    y_test_scaled = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))
    
    # Perform inverse transform
    y_test_unscaled = inverse_transform_targets(y_test_scaled, scaler, meta['target_indices'])
    
    print("Target Variables:", meta['target_cols'])
    print(f"y_test scaled range:   min={y_test_scaled.min():.4f}, max={y_test_scaled.max():.4f}")
    print(f"y_test unscaled range: min={y_test_unscaled.min():.2f} ug/m3, max={y_test_unscaled.max():.2f} ug/m3")
    print("\nSample Unscaled PM2.5 and PM10 values (First 5 prediction steps):")
    for step in range(5):
        pm25, pm10 = y_test_unscaled[0, step, :]
        print(f"  Step t+{step+1}: PM2.5 = {pm25:.2f} ug/m3 | PM10 = {pm10:.2f} ug/m3")
