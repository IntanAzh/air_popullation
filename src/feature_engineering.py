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

def add_lag_and_rolling_features(df, lag_vars=['PM2.5', 'NO2', 'SO2', 'temperature', 'humidity', 'wind_speed'],
                                 roll_vars=['PM2.5', 'temperature', 'humidity', 'wind_speed']):
    """
    Generates time lag features and rolling window statistics for temporal models.
    Strictly causal (backward-looking only, no future leakage).
    """
    df = df.copy()

    # Lags for 30-min steps: 1 step (30m), 2 (1h), 3 (1.5h), 4 (2h), 6 (3h), 12 (6h), 24 (12h), 48 (24h)
    lag_steps = [1, 2, 3, 4, 6, 12, 24, 48]
    for var in lag_vars:
        if var in df.columns:
            for step in lag_steps:
                df[f'{var}_lag_{step}'] = df[var].shift(step)

    # Rolling statistics (3h = 6 steps, 6h = 12 steps, 24h = 48 steps)
    for var in roll_vars:
        if var in df.columns:
            # 3-hour rolling (6 steps)
            df[f'{var}_roll_mean_3h'] = df[var].rolling(window=6, min_periods=3).mean()
            df[f'{var}_roll_std_3h'] = df[var].rolling(window=6, min_periods=3).std()
            df[f'{var}_roll_max_3h'] = df[var].rolling(window=6, min_periods=3).max()

            # 6-hour rolling (12 steps)
            df[f'{var}_roll_mean_6h'] = df[var].rolling(window=12, min_periods=6).mean()
            df[f'{var}_roll_std_6h'] = df[var].rolling(window=12, min_periods=6).std()

            # 24-hour rolling (48 steps)
            df[f'{var}_roll_mean_24h'] = df[var].rolling(window=48, min_periods=24).mean()
            df[f'{var}_roll_std_24h'] = df[var].rolling(window=48, min_periods=24).std()

    # Keep missing lag values; handle incomplete windows downstream.
    return df

def engineer_all_features(df):
    """
    Full feature engineering execution pipeline producing standardized 94 model features
    plus ground-truth forecasting targets (+30m and +60m).
    """
    df = df.copy()
    
    # Target ground truth strictly from raw un-imputed measurements (no synthetic evaluation labels)
    raw_pm25 = df['PM2.5_raw'] if 'PM2.5_raw' in df.columns else df['PM2.5']
    df['target_PM2.5_30m'] = raw_pm25.shift(-1)
    df['target_PM2.5_60m'] = raw_pm25.shift(-2)

    # Use causally imputed PM2.5 for input feature history if available
    if 'PM2.5_input' in df.columns:
        df['PM2.5'] = df['PM2.5_input']

    # Temporal Cyclic Features
    df = add_cyclic_time_features(df)

    # Wind Vector Features
    df = add_wind_vector_features(df)

    # Lag & Rolling Features
    df = add_lag_and_rolling_features(df)

    # Clean intermediate columns not part of feature set
    drop_temp = [c for c in ['PM2.5_raw', 'PM2.5_input', 'PM10'] if c in df.columns]
    df = df.drop(columns=drop_temp)

    return df

if __name__ == "__main__":
    from etl_pipeline import load_and_clean_csv_pollutants, load_and_clean_excel_meteo, merge_and_align_datasets
    
    raw = merge_and_align_datasets(load_and_clean_csv_pollutants(), load_and_clean_excel_meteo())
    feat_df = engineer_all_features(raw)
    print("Engineered feature set shape:", feat_df.shape)
    print("Columns:", feat_df.columns.tolist())
