import os
import json
import numpy as np
import pandas as pd
import joblib

from etl_pipeline import load_and_clean_csv_pollutants, load_and_clean_excel_meteo, merge_and_align_datasets
from feature_engineering import engineer_all_features
from create_dataset_tensors import create_sliding_windows_chronological

OUTPUT_DIR = r"c:\Users\INTAN\airpollutan\data\processed"
TENSOR_DIR = os.path.join(OUTPUT_DIR, "tensors")

def export_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TENSOR_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Executing ETL Pipeline for Raw Data...")
    print("=" * 60)
    pollutants = load_and_clean_csv_pollutants()
    meteo = load_and_clean_excel_meteo()
    clean_master = merge_and_align_datasets(pollutants, meteo)
    print(f"ETL completed. Master shape: {clean_master.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Executing Feature Engineering...")
    print("=" * 60)
    feat_df = engineer_all_features(clean_master)
    print(f"Feature engineering completed. Shape: {feat_df.shape}")

    print("\n" + "=" * 60)
    print("STEP 3: Creating Hourly Resampled Dataset...")
    print("=" * 60)
    hourly_df = feat_df.set_index('datetime').resample('1h').mean().reset_index()
    hourly_df = hourly_df.bfill().ffill()
    print(f"Hourly dataset created. Shape: {hourly_df.shape}")

    print("\n" + "=" * 60)
    print("STEP 4: Saving Master CSV, Parquet, and Quality Reports...")
    print("=" * 60)
    
    csv_30min_path = os.path.join(OUTPUT_DIR, "cilacap_air_quality_30min.csv")
    csv_hourly_path = os.path.join(OUTPUT_DIR, "cilacap_air_quality_hourly.csv")
    parquet_path = os.path.join(OUTPUT_DIR, "cilacap_air_quality_master.parquet")
    report_path = os.path.join(OUTPUT_DIR, "data_quality_report.json")

    feat_df.to_csv(csv_30min_path, index=False)
    hourly_df.to_csv(csv_hourly_path, index=False)

    try:
        feat_df.to_parquet(parquet_path, index=False)
        print(f"Saved: {parquet_path}")
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    quality_report = {
        'total_30min_records': len(feat_df),
        'total_hourly_records': len(hourly_df),
        'total_features': len(feat_df.columns) - 1,
        'start_date': str(feat_df['datetime'].min()),
        'end_date': str(feat_df['datetime'].max()),
        'null_count_after_imputation': int(feat_df.isnull().sum().sum()),
        'features': [c for c in feat_df.columns if c != 'datetime']
    }

    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=4, default=str)

    print(f"Saved: {csv_30min_path}")
    print(f"Saved: {csv_hourly_path}")
    print(f"Saved: {report_path}")

    print("\n" + "=" * 60)
    print("STEP 5: Generating 3D Tensor Arrays (Strict Chronological 80% Train / 20% Test Split)...")
    print("=" * 60)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), meta = create_sliding_windows_chronological(
        feat_df, target_cols=['PM2.5', 'PM10'], window_size=48, horizon=2, train_ratio=0.80, val_ratio=0.0
    )

    np.save(os.path.join(TENSOR_DIR, "X_train.npy"), X_tr)
    np.save(os.path.join(TENSOR_DIR, "y_train.npy"), y_tr)
    if X_va is not None:
        np.save(os.path.join(TENSOR_DIR, "X_val.npy"), X_va)
        np.save(os.path.join(TENSOR_DIR, "y_val.npy"), y_va)
    np.save(os.path.join(TENSOR_DIR, "X_test.npy"), X_te)
    np.save(os.path.join(TENSOR_DIR, "y_test.npy"), y_te)

    joblib.dump(meta['scaler'], os.path.join(TENSOR_DIR, "scaler.joblib"))

    meta_save = {
        'feature_cols': meta['feature_cols'],
        'target_cols': meta['target_cols'],
        'target_indices': meta['target_indices'],
        'window_size': meta['window_size'],
        'horizon': meta['horizon'],
        'X_train_shape': list(X_tr.shape),
        'y_train_shape': list(y_tr.shape),
        'X_test_shape': list(X_te.shape),
        'y_test_shape': list(y_te.shape),
        'train_period': meta['train_period'],
        'test_period': meta['test_period'],
        'shuffle': meta['shuffle']
    }

    with open(os.path.join(TENSOR_DIR, "tensor_metadata.json"), 'w') as f:
        json.dump(meta_save, f, indent=4)

    print(f"Tensor arrays saved in: {TENSOR_DIR}")
    print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
    if X_va is not None:
        print(f"X_val:   {X_va.shape} | y_val:   {y_va.shape}")
    print(f"X_test:  {X_te.shape} | y_test:  {y_te.shape}")
    print(f"Train Period: {meta['train_period']}")
    print(f"Test Period:  {meta['test_period']}")
    print("\nSUCCESSFULLY COMPLETED BIG DATA PREPROCESSING PIPELINE!")

if __name__ == "__main__":
    export_all()
