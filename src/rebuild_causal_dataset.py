import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath("."))
from sklearn.preprocessing import StandardScaler
from src.etl_pipeline import load_and_clean_csv_pollutants, load_and_clean_excel_meteo, merge_and_align_datasets
from src.feature_engineering import engineer_all_features

PROCESSED_DIR = r"c:\Users\INTAN\airpollutan\data\processed"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"
TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
REPORTS_DIR = r"c:\Users\INTAN\airpollutan\reports"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_sequences(df, feature_cols, lookback=48, horizons=[1, 2]):
    """
    Generates 3D Input Sequences (samples, timesteps, features) and 2D Targets (samples, horizons)
    with strict timestamp continuity and finite value checks.
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    
    timestamps = df['datetime'].values
    X_mat = df[feature_cols].values
    
    y_scaled_30m = df['target_PM2.5_30m_scaled'].values
    y_scaled_60m = df['target_PM2.5_60m_scaled'].values
    
    y_unscaled_30m = df['target_PM2.5_30m'].values
    y_unscaled_60m = df['target_PM2.5_60m'].values
    
    n_records = len(df)
    
    X_sequences = []
    y_scaled_sequences = []
    y_unscaled_sequences = []
    
    origin_times = []
    target_times_t1 = []
    target_times_t2 = []
    
    rejected_gap = 0
    rejected_target_nan = 0
    rejected_feature_nan = 0
    
    expected_delta = np.timedelta64(30, 'm')
    
    for i in range(lookback - 1, n_records - max(horizons)):
        win_start = i - lookback + 1
        win_end = i + 1
        
        # 1. Timestamp Continuity for Lookback Window
        win_times = timestamps[win_start:win_end]
        time_diffs = np.diff(win_times)
        if not np.all(time_diffs == expected_delta):
            rejected_gap += 1
            continue
            
        # 2. Timestamp Continuity for Forecast Horizons
        target_time_1 = timestamps[i + 1]
        target_time_2 = timestamps[i + 2]
        if (target_time_1 - win_times[-1] != expected_delta) or (target_time_2 - target_time_1 != expected_delta):
            rejected_gap += 1
            continue
            
        # 3. Target Validity (Must be finite observed values; no synthetic targets)
        y_s1 = y_scaled_30m[i]
        y_s2 = y_scaled_60m[i]
        y_u1 = y_unscaled_30m[i]
        y_u2 = y_unscaled_60m[i]
        
        if not (np.isfinite(y_s1) and np.isfinite(y_s2) and np.isfinite(y_u1) and np.isfinite(y_u2)):
            rejected_target_nan += 1
            continue
            
        # 4. Feature Window Validity (All 48 steps must be finite; rejects initial lag NaNs and long gap NaNs)
        X_win = X_mat[win_start:win_end, :]
        if not np.isfinite(X_win).all():
            rejected_feature_nan += 1
            continue
            
        # Sequence Accepted
        X_sequences.append(X_win)
        y_scaled_sequences.append([y_s1, y_s2])
        y_unscaled_sequences.append([y_u1, y_u2])
        
        origin_times.append(str(win_times[-1]))
        target_times_t1.append(str(target_time_1))
        target_times_t2.append(str(target_time_2))
        
    X_arr = np.array(X_sequences, dtype=np.float32) if X_sequences else np.empty((0, lookback, len(feature_cols)))
    y_scaled_arr = np.array(y_scaled_sequences, dtype=np.float32) if y_scaled_sequences else np.empty((0, 2))
    y_unscaled_arr = np.array(y_unscaled_sequences, dtype=np.float32) if y_unscaled_sequences else np.empty((0, 2))
    
    stats = {
        "accepted_sequences": len(X_sequences),
        "rejected_timestamp_gaps": rejected_gap,
        "rejected_target_nans": rejected_target_nan,
        "rejected_feature_nans": rejected_feature_nan,
        "tensor_X_shape": list(X_arr.shape),
        "tensor_y_shape": list(y_scaled_arr.shape)
    }
    
    return X_arr, y_scaled_arr, y_unscaled_arr, origin_times, target_times_t1, target_times_t2, stats

def rebuild_all():
    print("=" * 80)
    print("LANGKAH 3: MEMBANGUN ULANG DATASET DAN TENSOR DENGAN PREPROCESSING KAUSAL")
    print("=" * 80)
    
    # 1. ETL & Temporal Alignment
    print("\n[1/5] Memproses ETL dan Penyelarasan Temporal Kausal...")
    pol_df = load_and_clean_csv_pollutants()
    met_df = load_and_clean_excel_meteo()
    master = merge_and_align_datasets(pol_df, met_df)
    
    parquet_master = os.path.join(PROCESSED_DIR, "aligned_master_dataset.parquet")
    csv_master = os.path.join(PROCESSED_DIR, "aligned_master_dataset.csv")
    master.to_parquet(parquet_master, index=False)
    master.to_csv(csv_master, index=False)
    print(f"Master Dataset tersimpan: {master.shape} baris.")
    
    # 2. Causal Feature Engineering
    print("\n[2/5] Rekayasa Fitur Kausal (94 Fitur Model + 2 Target Ground Truth)...")
    feat_df = engineer_all_features(master)
    
    parquet_feat = os.path.join(PROCESSED_DIR, "feature_dataset.parquet")
    csv_feat = os.path.join(PROCESSED_DIR, "feature_dataset.csv")
    feat_df.to_parquet(parquet_feat, index=False)
    feat_df.to_csv(csv_feat, index=False)
    print(f"Feature Dataset tersimpan: {feat_df.shape} (Fitur: {len(feat_df.columns)-3}).")
    
    # 3. Chronological Split (70% Train, 15% Val, 15% Test)
    print("\n[3/5] Pemisahan Kronologis (70% Train / 15% Val / 15% Test)...")
    total_len = len(feat_df)
    train_end_idx = int(total_len * 0.70)
    val_end_idx = int(total_len * 0.85)
    
    df_train = feat_df.iloc[:train_end_idx].copy().reset_index(drop=True)
    df_val = feat_df.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
    df_test = feat_df.iloc[val_end_idx:].copy().reset_index(drop=True)
    
    print(f"Train : {len(df_train)} baris | {df_train['datetime'].min()} s/d {df_train['datetime'].max()}")
    print(f"Val   : {len(df_val)} baris | {df_val['datetime'].min()} s/d {df_val['datetime'].max()}")
    print(f"Test  : {len(df_test)} baris | {df_test['datetime'].min()} s/d {df_test['datetime'].max()}")
    
    df_train.to_parquet(os.path.join(PROCESSED_DIR, "train_dataset.parquet"), index=False)
    df_val.to_parquet(os.path.join(PROCESSED_DIR, "val_dataset.parquet"), index=False)
    df_test.to_parquet(os.path.join(PROCESSED_DIR, "test_dataset.parquet"), index=False)
    
    # 4. Standard Scaling (Fit strictly on Train set only)
    print("\n[4/5] Standardisasi Fitur & Target (Fit KHUSUS pada Training Set)...")
    exclude_cols = ['datetime', 'target_PM2.5_30m', 'target_PM2.5_60m']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    target_cols = ['target_PM2.5_30m', 'target_PM2.5_60m']
    
    x_scaler = StandardScaler()
    X_train_raw = df_train[feature_cols].values
    X_val_raw = df_val[feature_cols].values
    X_test_raw = df_test[feature_cols].values
    
    # Fit X_scaler strictly on train
    x_scaler.fit(X_train_raw)
    X_train_scaled = x_scaler.transform(X_train_raw)
    X_val_scaled = x_scaler.transform(X_val_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)
    
    # Fit Y_scaler strictly on train targets (ignoring NaNs)
    y_scaler = StandardScaler()
    y_train_raw = df_train[target_cols].values
    y_val_raw = df_val[target_cols].values
    y_test_raw = df_test[target_cols].values
    
    y_scaler.fit(y_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)
    
    # Save Scalers
    joblib.dump(x_scaler, os.path.join(MODEL_DIR, "x_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "y_scaler.joblib"))
    
    # Build Scaled DataFrames
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    df_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    df_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    for df_sc, df_orig, y_sc in [(df_train_scaled, df_train, y_train_scaled),
                                 (df_val_scaled, df_val, y_val_scaled),
                                 (df_test_scaled, df_test, y_test_scaled)]:
        df_sc['datetime'] = df_orig['datetime'].values
        for idx, tcol in enumerate(target_cols):
            df_sc[tcol] = df_orig[tcol].values
            df_sc[f'{tcol}_scaled'] = y_sc[:, idx]
            
    df_train_scaled.to_parquet(os.path.join(PROCESSED_DIR, "train_scaled.parquet"), index=False)
    df_val_scaled.to_parquet(os.path.join(PROCESSED_DIR, "val_scaled.parquet"), index=False)
    df_test_scaled.to_parquet(os.path.join(PROCESSED_DIR, "test_scaled.parquet"), index=False)
    
    # 5. 3D Sliding Window Sequence Generation
    print("\n[5/5] Membentuk Tensor Temporal 3D Sliding Window (Lookback=48 langkah / 24 jam)...")
    X_train, y_train_s, y_train_u, t_orig_tr, t1_tr, t2_tr, stats_tr = generate_sequences(df_train_scaled, feature_cols)
    X_val, y_val_s, y_val_u, t_orig_va, t1_va, t2_va, stats_va = generate_sequences(df_val_scaled, feature_cols)
    X_test, y_test_s, y_test_u, t_orig_te, t1_te, t2_te, stats_te = generate_sequences(df_test_scaled, feature_cols)
    
    print(f"Train Tensors : X={X_train.shape}, y={y_train_s.shape} (Diterima: {stats_tr['accepted_sequences']}, Ditolak target NaN: {stats_tr['rejected_target_nans']}, Ditolak feature NaN: {stats_tr['rejected_feature_nans']})")
    print(f"Val Tensors   : X={X_val.shape}, y={y_val_s.shape} (Diterima: {stats_va['accepted_sequences']}, Ditolak target NaN: {stats_va['rejected_target_nans']}, Ditolak feature NaN: {stats_va['rejected_feature_nans']})")
    print(f"Test Tensors  : X={X_test.shape}, y={y_test_s.shape} (Diterima: {stats_te['accepted_sequences']}, Ditolak target NaN: {stats_te['rejected_target_nans']}, Ditolak feature NaN: {stats_te['rejected_feature_nans']})")
    
    np.savez_compressed(
        os.path.join(TENSOR_DIR, "train_tensors.npz"),
        X=X_train, y_scaled=y_train_s, y_unscaled=y_train_u,
        origin_times=np.array(t_orig_tr), target_times_t1=np.array(t1_tr), target_times_t2=np.array(t2_tr)
    )
    np.savez_compressed(
        os.path.join(TENSOR_DIR, "val_tensors.npz"),
        X=X_val, y_scaled=y_val_s, y_unscaled=y_val_u,
        origin_times=np.array(t_orig_va), target_times_t1=np.array(t1_va), target_times_t2=np.array(t2_va)
    )
    np.savez_compressed(
        os.path.join(TENSOR_DIR, "test_tensors.npz"),
        X=X_test, y_scaled=y_test_s, y_unscaled=y_test_u,
        origin_times=np.array(t_orig_te), target_times_t1=np.array(t1_te), target_times_t2=np.array(t2_te)
    )
    
    summary_report = {
        "status": "SUCCESS",
        "causal_preprocessing": {
            "target_synthetic_imputation": False,
            "target_ground_truth_source": "Raw sensor measurements only (PM2.5_raw)",
            "input_feature_imputation": "Causal forward-only (limit 12 steps / 6h, no bfill)",
            "sliding_window_outage_handling": "Windows crossing unobserved outages >12 steps or missing targets strictly rejected"
        },
        "chronological_split": {
            "train": {"rows": len(df_train), "start": str(df_train['datetime'].min()), "end": str(df_train['datetime'].max())},
            "val": {"rows": len(df_val), "start": str(df_val['datetime'].min()), "end": str(df_val['datetime'].max())},
            "test": {"rows": len(df_test), "start": str(df_test['datetime'].min()), "end": str(df_test['datetime'].max())}
        },
        "tensor_statistics": {
            "train": stats_tr,
            "val": stats_va,
            "test": stats_te
        }
    }
    
    report_file = os.path.join(REPORTS_DIR, "causal_dataset_rebuild_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)
        
    print(f"\n[OK] Dataset dan Tensor Causal Berhasil Dibangun Ulang! Laporan disimpan di: {report_file}")

if __name__ == "__main__":
    rebuild_all()
