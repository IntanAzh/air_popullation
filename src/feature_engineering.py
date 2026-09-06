import numpy as np
import pandas as pd

def add_derived_pm10(df):
    """
    Adds PM10 feature based on PM2.5 industrial ratio relationship: PM10 ~ PM2.5 * 1.45.
    """
    df = df.copy()
    if 'PM2.5' in df.columns:
        # Standard industrial station ratio estimation for Cilacap
        df['PM10'] = (df['PM2.5'] * 1.45 + 2.5).round(2)
    return df

def add_cyclic_time_features(df):
    """
    Encodes datetime into smooth cyclic sine and cosine signals for hour, month, day of week.
    """
    df = df.copy()
    dt = pd.to_datetime(df['datetime'])
    
    hour = dt.dt.hour + dt.dt.minute / 60.0
    month = dt.dt.month
    dayofweek = dt.dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    df['day_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)

    return df

def add_wind_vector_features(df):
    """
    Decomposes wind speed and direction into u_wind (East-West) and v_wind (North-South).
    """
    df = df.copy()
    if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
        ws = df['wind_speed']
        wd_rad = df['wind_direction'] * np.pi / 180.0
        
        df['u_wind'] = -ws * np.sin(wd_rad)
        df['v_wind'] = -ws * np.cos(wd_rad)
    return df

def add_lag_and_rolling_features(df, target_cols=['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed']):
    """
    Generates time lag features and rolling window statistics for temporal models.
    """
    df = df.copy()

    # Lags for 30-min steps: 1 step (30m), 2 (1h), 6 (3h), 12 (6h), 24 (12h), 48 (24h)
    lags = [1, 2, 6, 12, 24]
    for col in target_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # Rolling statistics (3-hour window = 6 steps, 24-hour window = 48 steps)
            df[f'{col}_roll_mean_3h'] = df[col].rolling(window=6, min_periods=1).mean()
            df[f'{col}_roll_std_3h'] = df[col].rolling(window=6, min_periods=1).std().fillna(0)
            df[f'{col}_roll_mean_24h'] = df[col].rolling(window=48, min_periods=1).mean()

    # Backfill lag NaNs produced at start of series
    df = df.bfill().ffill()
    return df

def engineer_all_features(df):
    """
    Full feature engineering execution pipeline.
    """
    df = add_derived_pm10(df)
    df = add_cyclic_time_features(df)
    df = add_wind_vector_features(df)
    df = add_lag_and_rolling_features(df)
    return df

if __name__ == "__main__":
    from etl_pipeline import load_and_clean_csv_pollutants, load_and_clean_excel_meteo, merge_and_align_datasets
    
    raw = merge_and_align_datasets(load_and_clean_csv_pollutants(), load_and_clean_excel_meteo())
    feat_df = engineer_all_features(raw)
    print("Engineered feature set shape:", feat_df.shape)
    print("Columns:", feat_df.columns.tolist())
