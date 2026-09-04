import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sliding_windows_chronological(df, target_cols=['PM2.5', 'PM10'], window_size=48, horizon=2, train_ratio=0.80, val_ratio=0.0):
    """
    Formulates 3D Temporal Sliding Window Sequences for Time-Series Models (CNN-BiLSTM + Attention).
    
    STRICT TIME-SERIES REQUIREMENT:
    - NO RANDOM SHUFFLING (shuffle=False).
    - Chronological split based strictly on datetime sequence to prevent data leakage.
    - MinMaxScaler is fitted strictly on the Training set (first 80% of time series).
    
    Parameters:
    - df: Feature-engineered DataFrame containing 'datetime' and numerical features.
    - target_cols: Target variables to predict (PM2.5 and PM10).
    - window_size: Lookback length (default 48 steps = 24 hours).
    - horizon: Forecast horizon (default 2 steps = 1 hour ahead).
    - train_ratio: Ratio for chronological train split (default 0.80 = 80%).
    - val_ratio: Ratio for validation split if desired (default 0.0 = 20% pure test set).
    """
    df = df.copy()
    # Sort strictly by datetime to guarantee chronological order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c != 'datetime']
    data_values = df[feature_cols].values
    target_indices = [feature_cols.index(c) for c in target_cols]

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    if val_ratio > 0.0:
        n_val = int(n_total * val_ratio)
        train_data = data_values[:n_train]
        val_data = data_values[n_train - window_size : n_train + n_val]
        test_data = data_values[n_train + n_val - window_size :]
    else:
        # Pure 80% Train / 20% Test Chronological Split
        train_data = data_values[:n_train]
        test_data = data_values[n_train - window_size :]
        val_data = None

    # Fit MinMaxScaler strictly on Training Data (prevents data leakage)
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    val_scaled = scaler.transform(val_data) if val_data is not None else None

    def generate_3d_sequences(scaled_matrix):
        if scaled_matrix is None:
            return None, None
        X, y = [], []
        for i in range(len(scaled_matrix) - window_size - horizon + 1):
            X_win = scaled_matrix[i : i + window_size]
            y_win = scaled_matrix[i + window_size : i + window_size + horizon, target_indices]
            X.append(X_win)
            y.append(y_win)
        return np.array(X), np.array(y)

    X_train, y_train = generate_3d_sequences(train_scaled)
    X_test, y_test = generate_3d_sequences(test_scaled)
    X_val, y_val = generate_3d_sequences(val_scaled)

    metadata = {
        'total_rows': n_total,
        'train_rows': n_train,
        'test_rows': n_total - n_train,
        'train_period': f"{df['datetime'].iloc[0]} to {df['datetime'].iloc[n_train-1]}",
        'test_period': f"{df['datetime'].iloc[n_train]} to {df['datetime'].iloc[-1]}",
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'target_indices': target_indices,
        'window_size': window_size,
        'horizon': horizon,
        'shuffle': False,
        'scaler': scaler
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata

if __name__ == "__main__":
    from etl_pipeline import load_and_clean_csv_pollutants, load_and_clean_excel_meteo, merge_and_align_datasets
    from feature_engineering import engineer_all_features

    raw = merge_and_align_datasets(load_and_clean_csv_pollutants(), load_and_clean_excel_meteo())
    feat_df = engineer_all_features(raw)

    (X_tr, y_tr), _, (X_te, y_te), meta = create_sliding_windows_chronological(feat_df, train_ratio=0.80, val_ratio=0.0)

    print("=" * 70)
    print("PEMISAHAN DATA KRONOLOGIS (CHRONOLOGICAL TIME-SERIES SPLIT, SHUFFLE = FALSE)")
    print("=" * 70)
    print(f"Data Latih (Train 80%): {meta['train_rows']} baris | Periode: {meta['train_period']}")
    print(f"Data Uji   (Test 20%) : {meta['test_rows']} baris  | Periode: {meta['test_period']}")
    print("-" * 70)
    print(f"X_train (3D Tensor): {X_tr.shape}  | y_train (3D Tensor): {y_tr.shape}")
    print(f"X_test  (3D Tensor): {X_te.shape}  | y_test  (3D Tensor): {y_te.shape}")
