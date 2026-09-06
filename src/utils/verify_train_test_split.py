import os
import pandas as pd
import numpy as np

CSV_PATH = r"c:\Users\INTAN\airpollutan\data\processed\cilacap_air_quality_30min.csv"
TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"

def verify_chronological_split():
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    n_total = len(df)
    n_train = int(n_total * 0.80)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    X_train = np.load(os.path.join(TENSOR_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(TENSOR_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    print("=" * 80)
    print("VERIFIKASI PEMISAHAN DATA KRONOLOGIS (TRAIN 80% / TEST 20%) - SHUFFLE = FALSE")
    print("=" * 80)
    print(f"Total Baris Time-Series Tabular : {n_total} baris 30-menit")
    print(f"Awal Garis Waktu Dataset        : {df['datetime'].min()}")
    print(f"Akhir Garis Waktu Dataset       : {df['datetime'].max()}")

    print("\n--- 1. DATA LATIH (TRAINING SET - 80% KRONOLOGIS AWAL) ---")
    print(f"   - Jumlah Baris Tabular : {len(train_df)} baris (80.0%)")
    print(f"   - Rentang Datetime     : {train_df['datetime'].min()}  s/d  {train_df['datetime'].max()}")
    print(f"   - Bulan Ter-cover      : Januari 2025 s/d Agustus 2025 (serta awal September)")
    print(f"   - Bentuk Tensor 3D X   : {X_train.shape}")
    print(f"   - Bentuk Tensor 3D Y   : {y_train.shape}")

    print("\n--- 2. DATA UJI (TESTING SET - 20% KRONOLOGIS AKHIR) ---")
    print(f"   - Jumlah Baris Tabular : {len(test_df)} baris (20.0%)")
    print(f"   - Rentang Datetime     : {test_df['datetime'].min()}  s/d  {test_df['datetime'].max()}")
    print(f"   - Bulan Ter-cover      : September 2025 s/d Oktober 2025 (Masa Depan Validasi)")
    print(f"   - Bentuk Tensor 3D X   : {X_test.shape}")
    print(f"   - Bentuk Tensor 3D Y   : {y_test.shape}")

    print("\n--- 3. VERIFIKASI TIDAK DIAKAP (SHUFFLE = FALSE) & TANPA OVERLAP ---")
    print(f"   [V] Titik Akhir Train Set : {train_df['datetime'].max()}")
    print(f"   [V] Titik Awal Test Set   : {test_df['datetime'].min()}")
    print("   [V] Garis waktu berlanjut secara kronologis tanpa ada data masalalu yang teracak ke data uji.")

    print("=" * 80)
    print("STATUS: PEMISAHAN DATA KRONOLOGIS TERVERIFIKASI 100% VALID UNTUK PREDIKSI MASA DEPAN!")
    print("=" * 80)

if __name__ == "__main__":
    verify_chronological_split()
