import os
import numpy as np
import pandas as pd
from create_dataset_tensors import create_sliding_windows_chronological

CSV_PATH = r"c:\Users\INTAN\airpollutan\data\processed\cilacap_air_quality_30min.csv"

def demonstrate_sliding_window_transformation():
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print("=" * 75)
    print("DEMONSTRASI TEKNIK SLIDING WINDOW: TABULAR 2D -> TENSOR 3D (X, Y)")
    print("=" * 75)
    print(f"1. Bentuk Data Tabular 2D Asli (Total Baris x Total Fitur): {df.shape[0]} baris x {df.shape[1]-1} fitur")
    print(f"   Rentang Datetime: {df['datetime'].min()} s/d {df['datetime'].max()}")

    # Demo Lookback Options (24 Jam = 48 steps, 48 Jam = 96 steps)
    for hours, lookback_steps in [(24, 48), (48, 96)]:
        (X_tr, y_tr), _, (X_te, y_te), meta = create_sliding_windows_chronological(
            df=df,
            target_cols=['PM2.5', 'PM10'],
            window_size=lookback_steps,
            horizon=2,
            train_ratio=0.80
        )
        print(f"\n--- Konfigurasi Lookback {hours} Jam ({lookback_steps} Step Waktu 30-menit) ---")
        print(f"   - Matriks Input 3D X_train: {X_tr.shape}  -> (Sampel, Lookback={lookback_steps}, Fitur=59)")
        print(f"   - Matriks Target 3D y_train: {y_tr.shape}   -> (Sampel, Horizon=2, Target=2)")
        print(f"   - Matriks Input 3D X_test : {X_te.shape}  -> (Sampel, Lookback={lookback_steps}, Fitur=59)")
        print(f"   - Matriks Target 3D y_test : {y_te.shape}   -> (Sampel, Horizon=2, Target=2)")

    print("\n" + "=" * 75)
    print("PEMETAAN CONTOH SAMPELE PERTAMA (SLIDING WINDOW STEP 1):")
    print("=" * 75)
    sample_X = X_tr[0]
    sample_y = y_tr[0]
    print(f"Sample X[0] shape : {sample_X.shape} (Merangkum 48 step historis fitur cuaca & polutan)")
    print(f"Sample y[0] shape : {sample_y.shape} (Dipetakan ke 2 step waktu peramalan berikutnya)")
    print("=" * 75)

if __name__ == "__main__":
    demonstrate_sliding_window_transformation()
