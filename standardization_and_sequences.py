import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

CSV_PATH = r"c:\Users\INTAN\airpollutan\data\processed\cilacap_air_quality_30min.csv"
TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"

def process_standardization_and_sequences(
    df_path=CSV_PATH, 
    window_size=48, 
    horizon=2, 
    train_ratio=0.80, 
    target_cols=['PM2.5', 'PM10'],
    use_standard_scaler=True
):
    print("=" * 80)
    print("STANDARDISASI MEAN-CENTERING & UNIT VARIANSI (STANDARD SCALER)")
    print("SERTA PEMBENTUKAN SEKUENS TEMPORAL SLIDING WINDOW")
    print("=" * 80)

    df = pd.read_csv(df_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Feature columns (exclude datetime)
    feature_cols = [c for c in df.columns if c != 'datetime']
    target_indices = [feature_cols.index(col) for col in target_cols]

    data_values = df[feature_cols].values
    n_total = len(data_values)
    n_train = int(n_total * train_ratio)

    train_data = data_values[:n_train]
    test_data = data_values[n_train:]

    # 1. Standardisasi Mean-Centering (mean=0, variance=1) strictly on Training Data
    if use_standard_scaler:
        scaler = StandardScaler()
        print("1. Mengaplikasikan StandardScaler (Mean-Centering mu=0, Variansi Unit sigma=1):")
    else:
        scaler = MinMaxScaler()
        print("1. Mengaplikasikan MinMaxScaler (Rentang [0, 1]):")

    scaler.fit(train_data)  # Strictly fit on Training set (Data Leakage Prevention)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Print Mean & Std verification
    print(f"   [V] Scaler di-fit secara khusus pada 80% Training Set ({n_train} baris).")
    print(f"   - Mean Rata-rata Train Set (Target mu=0) : {train_scaled[:, target_indices[0]].mean():.6f}")
    print(f"   - Std Standar Deviasi Train Set (Target sigma=1): {train_scaled[:, target_indices[0]].std():.6f}")

    # 2. Pembentukan Sekuens Temporal (Sliding Window 2D -> 3D)
    print("\n2. Pembentukan Sekuens Temporal Sliding Window:")
    
    def build_3d_sequences(scaled_array, window_size, horizon, target_indices):
        X, Y = [], []
        for i in range(len(scaled_array) - window_size - horizon + 1):
            x_seq = scaled_array[i : i + window_size]
            y_seq = scaled_array[i + window_size : i + window_size + horizon, target_indices]
            X.append(x_seq)
            Y.append(y_seq)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    X_tr, y_tr = build_3d_sequences(train_scaled, window_size, horizon, target_indices)
    X_te, y_te = build_3d_sequences(test_scaled, window_size, horizon, target_indices)

    print(f"   - Tensor Input 3D X_train : {X_tr.shape} -> (Sampel, Lookback={window_size}, Fitur={X_tr.shape[2]})")
    print(f"   - Tensor Target 3D y_train: {y_tr.shape}  -> (Sampel, Horizon={horizon}, Target={y_tr.shape[2]})")
    print(f"   - Tensor Input 3D X_test  : {X_te.shape} -> (Sampel, Lookback={window_size}, Fitur={X_te.shape[2]})")
    print(f"   - Tensor Target 3D y_test : {y_te.shape}  -> (Sampel, Horizon={horizon}, Target={y_te.shape[2]})")

    # Save StandardScaler
    scaler_path = os.path.join(TENSOR_DIR, "standard_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved StandardScaler ke: {scaler_path}")

    print("\n3. Sampel Uji Sebelum vs Sesudah Standardisasi Mean-Centering:")
    print(f"{'Nama Variabel Fitur':<25} | {'Mean Asli':<12} | {'Std Asli':<12} | {'Mean Stdz':<12} | {'Std Stdz'}")
    print("-" * 80)
    for col_name in ['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed', 'pressure']:
        if col_name in feature_cols:
            idx = feature_cols.index(col_name)
            mean_orig = scaler.mean_[idx]
            std_orig = np.sqrt(scaler.var_[idx])
            mean_stdz = train_scaled[:, idx].mean()
            std_stdz = train_scaled[:, idx].std()
            print(f"{col_name:<25} | {mean_orig:<12.2f} | {std_orig:<12.2f} | {mean_stdz:<12.4f} | {std_stdz:<12.4f}")

    print("=" * 80)
    print("STATUS: STANDARDISASI MEAN-CENTERING & SLIDING WINDOW SELESAI 100%!")
    print("=" * 80)

    return (X_tr, y_tr), (X_te, y_te), scaler

if __name__ == "__main__":
    process_standardization_and_sequences()
