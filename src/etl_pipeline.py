import os
import glob
import re
import numpy as np
import pandas as pd

RAW_DIR = r"C:\Users\INTAN\Downloads\Data Mentah"

def load_and_clean_csv_pollutants(raw_dir=RAW_DIR):
    """
    Parses and merges all monthly pollutant CSV files (PM2.5, NO2, SO2).
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.CSV")) + glob.glob(os.path.join(raw_dir, "*.csv"))
    print(f"Found {len(csv_files)} pollutant CSV files.")

    records = []

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, sep=';', encoding='utf-8')
        except Exception:
            df = pd.read_csv(fpath, sep=';', encoding='latin1')

        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        if 'Datum/Zeit' not in df.columns or 'Wert' not in df.columns:
            continue

        # Parse timestamp
        df['datetime'] = pd.to_datetime(df['Datum/Zeit'], format='%Y.%m.%d %H:%M', errors='coerce')
        df = df.dropna(subset=['datetime'])

        # Clean numeric value (always convert string replace comma with dot)
        df['Wert'] = df['Wert'].astype(str).str.replace(',', '.').str.strip()
        df['Wert'] = pd.to_numeric(df['Wert'], errors='coerce')

        # Identify pollutant component
        comp_raw = str(df['Komponente'].iloc[0]) if 'Komponente' in df.columns and len(df) > 0 else ""
        if 'PM2_5' in comp_raw or 'PM 2_5' in fpath or 'PM2_5' in fpath or 'PM 2,5' in fpath or 'pm2_5' in fpath:
            pollutant = 'PM2.5'
        elif 'NO2' in comp_raw or 'NO2' in fpath:
            pollutant = 'NO2'
        elif 'SO2' in comp_raw or 'SO2' in fpath:
            pollutant = 'SO2'
        else:
            continue

        # Filter out status error rows if status code indicates failure (non-zero)
        if 'Fehlerstatus' in df.columns:
            err_status = pd.to_numeric(df['Fehlerstatus'], errors='coerce').fillna(0)
            df.loc[err_status != 0, 'Wert'] = np.nan

        # Negative value filter & physical threshold check (e.g. > 1000 ug/m3 extreme sensor glint)
        df.loc[df['Wert'] < 0, 'Wert'] = np.nan
        df.loc[df['Wert'] > 1000, 'Wert'] = np.nan

        sub_df = df[['datetime', 'Wert']].copy()
        sub_df = sub_df.rename(columns={'Wert': pollutant})
        records.append(sub_df)

    if not records:
        raise ValueError("No valid pollutant records extracted!")

    all_pollutants = pd.concat(records, ignore_index=True)
    all_pollutants = all_pollutants.groupby('datetime').mean().reset_index()
    all_pollutants = all_pollutants.sort_values('datetime').reset_index(drop=True)
    
    return all_pollutants

def load_and_clean_excel_meteo(raw_dir=RAW_DIR):
    """
    Parses 'Data Stasiun Kabupaten Cilacap Sidakaya 2025.xlsx' for meteorological variables.
    """
    excel_path = os.path.join(raw_dir, "Data Stasiun Kabupaten Cilacap Sidakaya 2025.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at {excel_path}")

    df_raw = pd.read_excel(excel_path, header=None)

    # Locate header row (Row index 7 contains column names)
    header_idx = 7
    df = df_raw.iloc[header_idx:].copy()
    df.columns = df.iloc[0]
    df = df[2:].reset_index(drop=True)  # Drop header and unit rows

    # Rename columns to standardized terms
    column_mapping = {
        'Waktu': 'datetime',
        'SO2': 'SO2_excel',
        'Kec.Angin': 'wind_speed',
        'Arah Angin': 'wind_direction',
        'Kelembaban': 'humidity',
        'Suhu': 'temperature',
        'Tek.Udara': 'pressure',
        'Sol.Rad': 'solar_radiation',
        'Curah Hujan': 'rainfall'
    }

    df = df.rename(columns=column_mapping)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    # Keep only relevant mapped columns
    keep_cols = ['datetime'] + [c for c in column_mapping.values() if c != 'datetime' and c in df.columns]
    df = df[keep_cols].copy()

    # Convert numeric columns
    num_cols = [c for c in df.columns if c != 'datetime']
    for col in num_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Filter unrealistic physics values
        if col == 'humidity':
            df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
        elif col == 'temperature':
            df.loc[(df[col] < 10) | (df[col] > 60), col] = np.nan
        elif col in ['wind_speed', 'solar_radiation', 'rainfall']:
            df.loc[df[col] < 0, col] = np.nan
        elif col == 'wind_direction':
            df.loc[(df[col] < 0) | (df[col] > 360), col] = np.nan

    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def merge_and_align_datasets(pollutants_df, meteo_df):
    """
    Merges pollutant and meteorological datasets on continuous 30-min datetime grid.
    """
    merged = pd.merge(pollutants_df, meteo_df, on='datetime', how='outer')

    if 'SO2' in merged.columns and 'SO2_excel' in merged.columns:
        merged['SO2'] = merged['SO2'].combine_first(merged['SO2_excel'])
        merged = merged.drop(columns=['SO2_excel'])
    elif 'SO2_excel' in merged.columns:
        merged['SO2'] = merged['SO2_excel']
        merged = merged.drop(columns=['SO2_excel'])

    merged = merged.sort_values('datetime').reset_index(drop=True)

    max_pol_date = pollutants_df['datetime'].max()
    merged = merged[merged['datetime'] <= max_pol_date].copy()

    # Reindex to gap-free 30-min time series grid
    start_time = merged['datetime'].min().floor('30min')
    end_time = merged['datetime'].max().ceil('30min')
    full_grid = pd.date_range(start=start_time, end=end_time, freq='30min', name='datetime')

    merged = merged.set_index('datetime')
    merged = merged.reindex(full_grid)

    # Cast all feature columns to float64
    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('float64')

    # -------------------------------------------------------------------------
    # STRATEGI PENANGANAN MISSING VALUES:
    # 1. Target PM2.5: Pertahankan nilai asli sensor (jangan buat label sintetis
    #    melalui bfill/ffill/interpolasi dua arah). Simpan di 'PM2.5_raw' agar
    #    sampel dengan target yang hilang dapat dieksklusi dari evaluasi.
    # 2. Fitur Input: Hanya gunakan imputasi kausal (limit_direction='forward'
    #    dan ffill terbatas <= 12 langkah / 6 jam). Hindari bfill atau informasi masa depan.
    # 3. Celah Waktu: Pertahankan grid 30 menit; celah panjang (> 12 langkah)
    #    dibiarkan NaN agar sliding window tidak melompati periode yang tidak valid.
    # -------------------------------------------------------------------------
    target_col = 'PM2.5'
    if target_col in merged.columns:
        merged['PM2.5_raw'] = merged[target_col].copy()

    # Imputasi kausal hanya untuk fitur input non-target
    input_feature_cols = [c for c in merged.columns if c not in [target_col, 'PM2.5_raw']]
    for col in input_feature_cols:
        merged[col] = merged[col].interpolate(method='time', limit=12, limit_direction='forward')
        merged[col] = merged[col].ffill(limit=12)

    # Imputasi kausal untuk konteks input historis PM2.5 (hanya untuk lag/rolling input, bukan label target)
    if target_col in merged.columns:
        merged['PM2.5_input'] = merged[target_col].interpolate(method='time', limit=12, limit_direction='forward').ffill(limit=12)

    merged = merged.reset_index()
    return merged

def run_etl_pipeline(raw_dir=RAW_DIR, output_dir=r"c:\Users\INTAN\airpollutan\data\processed"):
    os.makedirs(output_dir, exist_ok=True)
    print("Executing causal ETL pipeline...")
    pol_df = load_and_clean_csv_pollutants(raw_dir=raw_dir)
    met_df = load_and_clean_excel_meteo(raw_dir=raw_dir)
    master = merge_and_align_datasets(pol_df, met_df)
    
    parquet_path = os.path.join(output_dir, "aligned_master_dataset.parquet")
    csv_path = os.path.join(output_dir, "aligned_master_dataset.csv")
    master.to_parquet(parquet_path, index=False)
    master.to_csv(csv_path, index=False)
    print(f"Master dataset saved successfully to {parquet_path} and {csv_path} (Shape: {master.shape})")
    return master

if __name__ == "__main__":
    run_etl_pipeline()
