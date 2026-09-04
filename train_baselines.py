import os
import json
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scaler_utils import load_scaler_and_meta, inverse_transform_targets

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

def compute_mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0

def train_baselines():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train = np.load(os.path.join(TENSOR_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(TENSOR_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    # Flatten input sequences for 2D tabular sklearn models: (n_samples, seq_len * n_features)
    n_tr, seq_len, n_feat = X_train.shape
    n_te = X_test.shape[0]

    X_tr_flat = X_train.reshape(n_tr, seq_len * n_feat)
    X_te_flat = X_test.reshape(n_te, seq_len * n_feat)

    # Reshape target to (n_samples, horizon * n_targets)
    horizon, n_targets = y_train.shape[1], y_train.shape[2]
    y_tr_flat = y_train.reshape(n_tr, horizon * n_targets)
    y_te_flat = y_test.reshape(n_te, horizon * n_targets)

    scaler, meta = load_scaler_and_meta()
    target_names = meta['target_cols']

    baseline_results = {}

    print("=" * 65)
    print("TRAINING BASELINE BENCHMARK MODELS FOR JOURNAL COMPARISON")
    print("=" * 65)

    # 1. Ridge Regression Baseline
    print("\n1. Training Ridge Regression Baseline...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_flat, y_tr_flat)
    y_pred_ridge_scaled = ridge.predict(X_te_flat).reshape(n_te, horizon, n_targets)
    
    y_pred_ridge = inverse_transform_targets(y_pred_ridge_scaled, scaler, meta['target_indices'])
    y_true = inverse_transform_targets(y_test, scaler, meta['target_indices'])

    baseline_results['Ridge'] = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, :, i].flatten()
        yp = y_pred_ridge[:, :, i].flatten()
        baseline_results['Ridge'][name] = {
            'MAE': round(float(mean_absolute_error(yt, yp)), 4),
            'RMSE': round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            'MAPE': round(float(compute_mape(yt, yp)), 4),
            'R2': round(float(r2_score(yt, yp)), 4)
        }
    print("   [V] Ridge Regression completed.")

    # 2. Random Forest Baseline
    print("\n2. Training Random Forest Regressor Baseline...")
    rf = RandomForestRegressor(n_estimators=10, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X_tr_flat, y_tr_flat)
    y_pred_rf_scaled = rf.predict(X_te_flat).reshape(n_te, horizon, n_targets)

    y_pred_rf = inverse_transform_targets(y_pred_rf_scaled, scaler, meta['target_indices'])

    baseline_results['RandomForest'] = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, :, i].flatten()
        yp = y_pred_rf[:, :, i].flatten()
        baseline_results['RandomForest'][name] = {
            'MAE': round(float(mean_absolute_error(yt, yp)), 4),
            'RMSE': round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            'MAPE': round(float(compute_mape(yt, yp)), 4),
            'R2': round(float(r2_score(yt, yp)), 4)
        }
    print("   [V] Random Forest Regressor completed.")

    # Save Baseline Results
    save_path = os.path.join(MODEL_DIR, "baseline_metrics.json")
    with open(save_path, 'w') as f:
        json.dump(baseline_results, f, indent=4)

    print("\n" + "=" * 65)
    print("SUMMARY COMPARATIVE TABLE (JOURNAL BENCHMARK)")
    print("=" * 65)
    for model_name, metrics in baseline_results.items():
        print(f"\n[{model_name}]")
        for target, val in metrics.items():
            print(f"  {target:<6} -> MAE: {val['MAE']:<7} | RMSE: {val['RMSE']:<7} | MAPE: {val['MAPE']:<6}% | R2: {val['R2']}")

    print(f"\nSaved Baseline Metrics Report: {save_path}")

if __name__ == "__main__":
    train_baselines()
