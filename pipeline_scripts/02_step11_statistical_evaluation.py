import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROCESSED_DIR = r"c:\Users\INTAN\airpollutan\data\processed"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"
TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
SCRATCH_DIR = r"C:\Users\INTAN\.gemini\antigravity\brain\5c3a30c5-8ff2-4f5c-bacf-cc114ea67f12\scratch"
FIGURE_DIR = os.path.join(SCRATCH_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

class FlexibleCNNBiLSTM(nn.Module):
    def __init__(self, input_dim, conv_filters=32, lstm_hidden=32, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bilstm = nn.LSTM(conv_filters, lstm_hidden, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        out, _ = self.bilstm(x)
        last_out = out[:, -1, :]
        x = self.dropout(last_out)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_out):
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class FlexibleCNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, conv_filters=32, lstm_hidden=32, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bilstm = nn.LSTM(conv_filters, lstm_hidden, bidirectional=True, batch_first=True)
        self.attention = TemporalAttention(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        out, _ = self.bilstm(x)
        context = self.attention(out)
        x = self.dropout(context)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def calculate_metrics(y_true, y_pred):
    mae_30 = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
    rmse_30 = float(np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)))
    ss_tot_30 = np.sum((y_true[:, 0] - np.mean(y_true[:, 0])) ** 2)
    ss_res_30 = np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2)
    r2_30 = float(1.0 - (ss_res_30 / ss_tot_30))
    
    mae_60 = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))
    rmse_60 = float(np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)))
    ss_tot_60 = np.sum((y_true[:, 1] - np.mean(y_true[:, 1])) ** 2)
    ss_res_60 = np.sum((y_true[:, 1] - y_pred[:, 1]) ** 2)
    r2_60 = float(1.0 - (ss_res_60 / ss_tot_60))
    
    return {
        "MAE_30": round(mae_30, 4), "RMSE_30": round(rmse_30, 4), "R2_30": round(r2_30, 4),
        "MAE_60": round(mae_60, 4), "RMSE_60": round(rmse_60, 4), "R2_60": round(r2_60, 4),
        "MAE_avg": round((mae_30 + mae_60) / 2.0, 4),
        "RMSE_avg": round((rmse_30 + rmse_60) / 2.0, 4),
        "R2_avg": round((r2_30 + r2_60) / 2.0, 4)
    }

def diebold_mariano_test(e1, e2, h=1, power=2, max_lags=None):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction
    and HAC (Newey-West / Bartlett kernel) variance estimation for serial autocorrelation.
    """
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1**2 - e2**2
        
    n = len(d)
    d_bar = float(np.mean(d))
    
    # HAC bandwidth selection: Bartlett kernel with Andrews/Newey-West rule
    if max_lags is None:
        max_lags = max(h - 1, int(np.ceil(4.0 * (n / 100.0)**(2.0 / 9.0))))
        
    gamma_0 = float(np.mean((d - d_bar)**2))
    weighted_cov_sum = 0.0
    for k in range(1, max_lags + 1):
        gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        weight_k = 1.0 - (k / (max_lags + 1.0))  # Bartlett kernel
        weighted_cov_sum += 2.0 * weight_k * gamma_k
        
    var_d = gamma_0 + weighted_cov_sum
    if var_d <= 0:
        var_d = 1e-8
        
    dm_stat = d_bar / np.sqrt(var_d / n)
    hln_corr = np.sqrt((n + 1 - 2*h + (h/n)*(h-1)) / n)
    dm_stat_corrected = dm_stat * hln_corr
    
    # Precise two-sided p-value using Student-t distribution (df = n - 1)
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1)))
    
    return {
        "DM_statistic": round(float(dm_stat_corrected), 4),
        "DM_statistic_raw": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "hac_bandwidth_lags": int(max_lags),
        "hac_kernel": "Bartlett (Newey-West)",
        "degrees_of_freedom": int(n - 1),
        "significant_alpha_0.01": bool(p_value < 0.01),
        "significant_alpha_0.05": bool(p_value < 0.05)
    }

def block_bootstrap_rmse_diff_comprehensive(y_true, pred_base, pred_model, block_sizes=[12, 24, 48, 96], n_bootstraps=1000, seed=42):
    """
    Paired Moving Block Bootstrap across multiple block lengths.
    - Preserves paired sampling between ground truth, baseline, and model.
    - Tests multiple block lengths (6h, 12h, 24h diurnal, 48h).
    - Reports 95% and 99% confidence intervals.
    - Properly reports centered two-sided hypothesis test p-value and directional empirical probability.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    err_base_sq_30 = (y_true[:, 0] - pred_base[:, 0])**2
    err_mod_sq_30 = (y_true[:, 0] - pred_model[:, 0])**2
    err_base_sq_60 = (y_true[:, 1] - pred_base[:, 1])**2
    err_mod_sq_60 = (y_true[:, 1] - pred_model[:, 1])**2
    
    obs_rmse_b30 = np.sqrt(np.mean(err_base_sq_30))
    obs_rmse_m30 = np.sqrt(np.mean(err_mod_sq_30))
    obs_diff_30 = float(obs_rmse_b30 - obs_rmse_m30)
    
    obs_rmse_b60 = np.sqrt(np.mean(err_base_sq_60))
    obs_rmse_m60 = np.sqrt(np.mean(err_mod_sq_60))
    obs_diff_60 = float(obs_rmse_b60 - obs_rmse_m60)
    
    results = {
        "observed_rmse_diff": {
            "+30m": round(obs_diff_30, 4),
            "+60m": round(obs_diff_60, 4)
        },
        "block_results": {}
    }
    
    for b_size in block_sizes:
        num_blocks = int(np.ceil(n / b_size))
        starts = rng.integers(0, n - b_size + 1, size=(n_bootstraps, num_blocks))
        
        diff_30 = np.zeros(n_bootstraps)
        diff_60 = np.zeros(n_bootstraps)
        
        for b in range(n_bootstraps):
            # Paired block resample indices
            idx = np.concatenate([np.arange(s, s + b_size) for s in starts[b]])[:n]
            
            rmse_b30 = np.sqrt(np.mean(err_base_sq_30[idx]))
            rmse_m30 = np.sqrt(np.mean(err_mod_sq_30[idx]))
            diff_30[b] = rmse_b30 - rmse_m30
            
            rmse_b60 = np.sqrt(np.mean(err_base_sq_60[idx]))
            rmse_m60 = np.sqrt(np.mean(err_mod_sq_60[idx]))
            diff_60[b] = rmse_b60 - rmse_m60
            
        ci_95_30 = [float(np.percentile(diff_30, 2.5)), float(np.percentile(diff_30, 97.5))]
        ci_99_30 = [float(np.percentile(diff_30, 0.5)), float(np.percentile(diff_30, 99.5))]
        ci_95_60 = [float(np.percentile(diff_60, 2.5)), float(np.percentile(diff_60, 97.5))]
        ci_99_60 = [float(np.percentile(diff_60, 0.5)), float(np.percentile(diff_60, 99.5))]
        
        # Centered two-sided hypothesis test p-value: H0: mean(diff) = 0
        p_val_two_sided_30 = float(np.mean(np.abs(diff_30 - np.mean(diff_30)) >= np.abs(obs_diff_30)))
        p_val_two_sided_60 = float(np.mean(np.abs(diff_60 - np.mean(diff_60)) >= np.abs(obs_diff_60)))
        
        # Empirical two-sided p-value
        p_val_emp_30 = float(min(1.0, 2.0 * min(np.mean(diff_30 <= 0), np.mean(diff_30 >= 0))))
        p_val_emp_60 = float(min(1.0, 2.0 * min(np.mean(diff_60 <= 0), np.mean(diff_60 >= 0))))
        
        # One-sided directional tail: P(diff <= 0)
        p_dir_30 = float(np.mean(diff_30 <= 0))
        p_dir_60 = float(np.mean(diff_60 <= 0))
        
        duration_hours = float(b_size * 0.5)  # 30-min per step
        results["block_results"][f"block_{b_size}"] = {
            "block_size_timesteps": int(b_size),
            "block_duration_hours": duration_hours,
            "+30m": {
                "mean_diff": round(float(np.mean(diff_30)), 4),
                "std_err": round(float(np.std(diff_30)), 4),
                "95_CI": [round(c, 4) for c in ci_95_30],
                "99_CI": [round(c, 4) for c in ci_99_30],
                "p_value_two_sided_centered": round(p_val_two_sided_30, 5),
                "p_value_two_sided_empirical": round(p_val_emp_30, 5),
                "prop_diff_le_0_one_sided": round(p_dir_30, 5),
                "significant_at_alpha_0.05": bool(ci_95_30[0] > 0 or ci_95_30[1] < 0)
            },
            "+60m": {
                "mean_diff": round(float(np.mean(diff_60)), 4),
                "std_err": round(float(np.std(diff_60)), 4),
                "95_CI": [round(c, 4) for c in ci_95_60],
                "99_CI": [round(c, 4) for c in ci_99_60],
                "p_value_two_sided_centered": round(p_val_two_sided_60, 5),
                "p_value_two_sided_empirical": round(p_val_emp_60, 5),
                "prop_diff_le_0_one_sided": round(p_dir_60, 5),
                "significant_at_alpha_0.05": bool(ci_95_60[0] > 0 or ci_95_60[1] < 0)
            }
        }
        
    primary_block_key = f"block_{block_sizes[2]}" if len(block_sizes) > 2 else f"block_{block_sizes[0]}"
    results["primary_reference_block"] = results["block_results"][primary_block_key]
    
    return results

def main():
    print("=== STARTING STEP 11: INSTANT EVALUATION ===")
    
    df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_dataset.parquet"))
    all_cols = [c for c in df_train.columns if c not in ['datetime', 'target_PM2.5_30m', 'target_PM2.5_60m']]
    f1_cols = [c for c in all_cols if 'PM2.5' in c]
    f1_indices = [all_cols.index(c) for c in f1_cols]
    
    test_npz = np.load(os.path.join(TENSOR_DIR, "test_tensors.npz"))
    X_test_full = test_npz['X']
    y_test_unscaled = test_npz['y_unscaled']
    
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.joblib"))
    df_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_dataset.parquet"))
    test_dates = df_test['datetime'].values[47:47 + len(y_test_unscaled)]
    
    # 1. OBTAIN PREDICTIONS
    pm25_idx = all_cols.index('PM2.5')
    last_pm25_scaled = X_test_full[:, -1, pm25_idx]
    p_persistence_scaled = np.column_stack([last_pm25_scaled, last_pm25_scaled])
    p_persistence = y_scaler.inverse_transform(p_persistence_scaled)
    
    # Ridge (B1)
    train_npz = np.load(os.path.join(TENSOR_DIR, "train_tensors.npz"))
    X_tr_2d = train_npz['X'].reshape(len(train_npz['X']), -1)
    y_tr_s = train_npz['y_scaled']
    X_te_2d = X_test_full.reshape(len(X_test_full), -1)
    
    ridge_path = os.path.join(MODEL_DIR, "B1_Ridge.joblib")
    if os.path.exists(ridge_path):
        ridge = joblib.load(ridge_path)
    else:
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=100.0)
        ridge.fit(X_tr_2d, y_tr_s)
    p_ridge = y_scaler.inverse_transform(ridge.predict(X_te_2d))
    
    # Random Forest (B2)
    rf_path = os.path.join(MODEL_DIR, "B2_RandomForest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
    else:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr_2d, y_tr_s)
    p_rf = y_scaler.inverse_transform(rf.predict(X_te_2d))
    
    # Champion Model: Optimal CNN-BiLSTM (F1)
    X_te_f1 = torch.tensor(X_test_full[:, :, f1_indices], dtype=torch.float32)
    m_bilstm_f1 = FlexibleCNNBiLSTM(input_dim=len(f1_cols), conv_filters=32, lstm_hidden=32, dropout_rate=0.2)
    m_bilstm_f1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Optimal_CNN_BiLSTM.pt"), map_location='cpu'))
    m_bilstm_f1.eval()
    with torch.no_grad():
        p_cnn_bilstm_f1 = y_scaler.inverse_transform(m_bilstm_f1(X_te_f1).numpy())
        
    # Model: Optimal CNN-BiLSTM-Attention (F1)
    m_attn_f1 = FlexibleCNNBiLSTMAttention(input_dim=len(f1_cols), conv_filters=32, lstm_hidden=32, dropout_rate=0.2)
    m_attn_f1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Optimal_CNN_BiLSTM_Attention.pt"), map_location='cpu'))
    m_attn_f1.eval()
    with torch.no_grad():
        p_cnn_bilstm_attn_f1 = y_scaler.inverse_transform(m_attn_f1(X_te_f1).numpy())
        
    # Unregularized Model: CNN-BiLSTM (Full 94 features)
    X_te_f4 = torch.tensor(X_test_full, dtype=torch.float32)
    m_bilstm_f4 = FlexibleCNNBiLSTM(input_dim=len(all_cols), conv_filters=64, lstm_hidden=64, dropout_rate=0.3)
    m_bilstm_f4.load_state_dict(torch.load(os.path.join(MODEL_DIR, "B6_CNN_BiLSTM.pt"), map_location='cpu'))
    m_bilstm_f4.eval()
    with torch.no_grad():
        p_cnn_bilstm_f4 = y_scaler.inverse_transform(m_bilstm_f4(X_te_f4).numpy())
        
    # Autoregressive Model: AR(24)
    ar_model_path = os.path.join(MODEL_DIR, "B_AR_model.joblib")
    if os.path.exists(ar_model_path):
        ar_data = joblib.load(ar_model_path)
        ar_model = ar_data["model"]
        p_ar = y_scaler.inverse_transform(ar_model.predict(X_test_full[:, -24:, pm25_idx]))
    else:
        from sklearn.linear_model import LinearRegression
        ar_model = LinearRegression().fit(train_npz['X'][:, -24:, pm25_idx], y_tr_s)
        p_ar = y_scaler.inverse_transform(ar_model.predict(X_test_full[:, -24:, pm25_idx]))
        
    # -------------------------------------------------------------
    # 11A — FINAL PERFORMANCE SUMMARY TABLE
    # -------------------------------------------------------------
    models_pred = {
        "Persistence (B0)": p_persistence,
        "AR(24)": p_ar,
        "Ridge Regression (B1)": p_ridge,
        "Random Forest (B2)": p_rf,
        "CNN-BiLSTM (Full 94 Fitur)": p_cnn_bilstm_f4,
        "CNN-BiLSTM-Attention (F1)": p_cnn_bilstm_attn_f1,
        "CNN-BiLSTM (Champion F1)": p_cnn_bilstm_f1
    }
    
    summary_metrics = {}
    for name, pred in models_pred.items():
        summary_metrics[name] = calculate_metrics(y_test_unscaled, pred)
        
    # -------------------------------------------------------------
    # 11B — IMPROVEMENT OVER BASELINE
    # -------------------------------------------------------------
    rmse_pers_30 = summary_metrics["Persistence (B0)"]["RMSE_30"]
    rmse_pers_60 = summary_metrics["Persistence (B0)"]["RMSE_60"]
    rmse_champ_30 = summary_metrics["CNN-BiLSTM (Champion F1)"]["RMSE_30"]
    rmse_champ_60 = summary_metrics["CNN-BiLSTM (Champion F1)"]["RMSE_60"]
    
    imp_30 = ((rmse_pers_30 - rmse_champ_30) / rmse_pers_30) * 100.0
    imp_60 = ((rmse_pers_60 - rmse_champ_60) / rmse_pers_60) * 100.0
    
    mae_pers_30 = summary_metrics["Persistence (B0)"]["MAE_30"]
    mae_pers_60 = summary_metrics["Persistence (B0)"]["MAE_60"]
    mae_champ_30 = summary_metrics["CNN-BiLSTM (Champion F1)"]["MAE_30"]
    mae_champ_60 = summary_metrics["CNN-BiLSTM (Champion F1)"]["MAE_60"]
    
    imp_mae_30 = ((mae_pers_30 - mae_champ_30) / mae_pers_30) * 100.0
    imp_mae_60 = ((mae_pers_60 - mae_champ_60) / mae_pers_60) * 100.0
    
    improvements = {
        "+30m": {"RMSE_Persistence": rmse_pers_30, "RMSE_Champion": rmse_champ_30, "RMSE_Improvement_Percent": round(imp_30, 2), "MAE_Improvement_Percent": round(imp_mae_30, 2)},
        "+60m": {"RMSE_Persistence": rmse_pers_60, "RMSE_Champion": rmse_champ_60, "RMSE_Improvement_Percent": round(imp_60, 2), "MAE_Improvement_Percent": round(imp_mae_60, 2)}
    }
    
    # -------------------------------------------------------------
    # 11C — STATISTICAL SIGNIFICANCE TESTS
    # -------------------------------------------------------------
    TEST_SEED = 42
    N_BOOTSTRAPS = 1000
    BLOCK_SIZES = [12, 24, 48, 96]
    
    e_pers = y_test_unscaled - p_persistence
    e_champ = y_test_unscaled - p_cnn_bilstm_f1
    e_ar = y_test_unscaled - p_ar
    e_ridge = y_test_unscaled - p_ridge
    e_rf = y_test_unscaled - p_rf
    e_full = y_test_unscaled - p_cnn_bilstm_f4
    
    # Diebold-Mariano tests with HAC variance (Bartlett/Newey-West) & HLN correction
    dm_res_30 = diebold_mariano_test(e_pers[:, 0], e_champ[:, 0], h=1, power=2)
    dm_res_60 = diebold_mariano_test(e_pers[:, 1], e_champ[:, 1], h=2, power=2)
    
    dm_vs_ar_30 = diebold_mariano_test(e_ar[:, 0], e_champ[:, 0], h=1, power=2)
    dm_vs_ar_60 = diebold_mariano_test(e_ar[:, 1], e_champ[:, 1], h=2, power=2)
    dm_vs_ridge_30 = diebold_mariano_test(e_ridge[:, 0], e_champ[:, 0], h=1, power=2)
    dm_vs_ridge_60 = diebold_mariano_test(e_ridge[:, 1], e_champ[:, 1], h=2, power=2)
    dm_vs_rf_30 = diebold_mariano_test(e_rf[:, 0], e_champ[:, 0], h=1, power=2)
    dm_vs_rf_60 = diebold_mariano_test(e_rf[:, 1], e_champ[:, 1], h=2, power=2)
    dm_vs_full_30 = diebold_mariano_test(e_full[:, 0], e_champ[:, 0], h=1, power=2)
    dm_vs_full_60 = diebold_mariano_test(e_full[:, 1], e_champ[:, 1], h=2, power=2)

    # Moving Block Bootstrap across multiple block lengths with paired sampling
    boot_results = block_bootstrap_rmse_diff_comprehensive(
        y_test_unscaled, p_persistence, p_cnn_bilstm_f1,
        block_sizes=BLOCK_SIZES, n_bootstraps=N_BOOTSTRAPS, seed=TEST_SEED
    )
    
    stat_tests = {
        "Diebold_Mariano_Test": {
            "vs_Persistence_B0": {
                "+30m": dm_res_30,
                "+60m": dm_res_60
            },
            "vs_AR_24": {
                "+30m": dm_vs_ar_30,
                "+60m": dm_vs_ar_60
            },
            "vs_Ridge_Regression_B1": {
                "+30m": dm_vs_ridge_30,
                "+60m": dm_vs_ridge_60
            },
            "vs_Random_Forest_B2": {
                "+30m": dm_vs_rf_30,
                "+60m": dm_vs_rf_60
            },
            "vs_CNN_BiLSTM_94_Features": {
                "+30m": dm_vs_full_30,
                "+60m": dm_vs_full_60
            },
            "+30m": dm_res_30,
            "+60m": dm_res_60
        },
        "Block_Bootstrap_Test": boot_results
    }
    
    # -------------------------------------------------------------
    # 11D — RESIDUAL & ERROR ANALYSIS
    # -------------------------------------------------------------
    res_30 = e_champ[:, 0]
    res_60 = e_champ[:, 1]
    
    residual_stats = {
        "+30m": {
            "mean_bias": round(float(np.mean(res_30)), 4),
            "std_dev": round(float(np.std(res_30)), 4),
            "median": round(float(np.median(res_30)), 4),
            "skewness": round(float(stats.skew(res_30)), 4),
            "kurtosis": round(float(stats.kurtosis(res_30)), 4),
            "max_positive_error": round(float(np.max(res_30)), 4),
            "max_negative_error": round(float(np.min(res_30)), 4)
        },
        "+60m": {
            "mean_bias": round(float(np.mean(res_60)), 4),
            "std_dev": round(float(np.std(res_60)), 4),
            "median": round(float(np.median(res_60)), 4),
            "skewness": round(float(stats.skew(res_60)), 4),
            "kurtosis": round(float(stats.kurtosis(res_60)), 4),
            "max_positive_error": round(float(np.max(res_60)), 4),
            "max_negative_error": round(float(np.min(res_60)), 4)
        }
    }
    
    cat_bins = [0, 15, 35, 500]
    cat_labels = ["Baik (<15)", "Sedang (15-35)", "Tidak Sehat (>35)"]
    y_true_30 = y_test_unscaled[:, 0]
    categories = pd.cut(y_true_30, bins=cat_bins, labels=cat_labels)
    
    cat_breakdown = {}
    for cat in cat_labels:
        mask = (categories == cat)
        if np.sum(mask) > 0:
            mae_cat = np.mean(np.abs(e_champ[mask, 0]))
            rmse_cat = np.sqrt(np.mean(e_champ[mask, 0]**2))
            count_cat = int(np.sum(mask))
            cat_breakdown[cat] = {"count": count_cat, "MAE_30": round(float(mae_cat), 4), "RMSE_30": round(float(rmse_cat), 4)}

    abs_errors_30 = np.abs(res_30)
    top5_idx = np.argsort(abs_errors_30)[-5:][::-1]
    top5_extreme_events = []
    for idx in top5_idx:
        top5_extreme_events.append({
            "index": int(idx),
            "date": str(test_dates[idx]),
            "actual_PM2.5_30m": round(float(y_test_unscaled[idx, 0]), 2),
            "predicted_PM2.5_30m": round(float(p_cnn_bilstm_f1[idx, 0]), 2),
            "absolute_error": round(float(abs_errors_30[idx]), 2)
        })

    # -------------------------------------------------------------
    # 11E — VISUALIZATIONS
    # -------------------------------------------------------------
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'figure.dpi': 300})
    
    # Figure 1: Time-Series Predictions vs Actual (7-Day Sample Period)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    sample_slice = slice(200, 536)
    dates_sub = pd.to_datetime(test_dates[sample_slice])
    
    ax[0].plot(dates_sub, y_test_unscaled[sample_slice, 0], color='#1f77b4', linewidth=1.8, label='Actual PM2.5')
    ax[0].plot(dates_sub, p_cnn_bilstm_f1[sample_slice, 0], color='#d62728', linestyle='--', linewidth=1.5, label='CNN-BiLSTM (F1)')
    ax[0].plot(dates_sub, p_persistence[sample_slice, 0], color='#7f7f7f', linestyle=':', linewidth=1.2, label='Persistence')
    ax[0].set_title("A. Short-Term Forecast Horizon (+30 Minutes)", fontsize=13, fontweight='bold', loc='left')
    ax[0].set_ylabel("PM2.5 (µg/m³)")
    ax[0].legend(loc='upper right', frameon=True)
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(dates_sub, y_test_unscaled[sample_slice, 1], color='#1f77b4', linewidth=1.8, label='Actual PM2.5')
    ax[1].plot(dates_sub, p_cnn_bilstm_f1[sample_slice, 1], color='#2ca02c', linestyle='--', linewidth=1.5, label='CNN-BiLSTM (F1)')
    ax[1].plot(dates_sub, p_persistence[sample_slice, 1], color='#7f7f7f', linestyle=':', linewidth=1.2, label='Persistence')
    ax[1].set_title("B. Short-Term Forecast Horizon (+60 Minutes)", fontsize=13, fontweight='bold', loc='left')
    ax[1].set_ylabel("PM2.5 (µg/m³)")
    ax[1].set_xlabel("Datetime (UTC+7)")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax[1].legend(loc='upper right', frameon=True)
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(FIGURE_DIR, "fig1_timeseries_forecast.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Figure 2: Scatter Plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].scatter(y_test_unscaled[:, 0], p_cnn_bilstm_f1[:, 0], alpha=0.3, color='#1f77b4', edgecolors='none', s=20)
    max_val_30 = max(np.max(y_test_unscaled[:, 0]), np.max(p_cnn_bilstm_f1[:, 0])) + 5
    ax[0].plot([0, max_val_30], [0, max_val_30], 'r--', label='1:1 Ideal Line')
    ax[0].set_title(f"A. Horizon +30 Min ($R^2 = {summary_metrics['CNN-BiLSTM (Champion F1)']['R2_30']:.4f}$)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Actual PM2.5 (µg/m³)")
    ax[0].set_ylabel("Predicted PM2.5 (µg/m³)")
    ax[0].legend(loc='upper left')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].scatter(y_test_unscaled[:, 1], p_cnn_bilstm_f1[:, 1], alpha=0.3, color='#2ca02c', edgecolors='none', s=20)
    max_val_60 = max(np.max(y_test_unscaled[:, 1]), np.max(p_cnn_bilstm_f1[:, 1])) + 5
    ax[1].plot([0, max_val_60], [0, max_val_60], 'r--', label='1:1 Ideal Line')
    ax[1].set_title(f"B. Horizon +60 Min ($R^2 = {summary_metrics['CNN-BiLSTM (Champion F1)']['R2_60']:.4f}$)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Actual PM2.5 (µg/m³)")
    ax[1].set_ylabel("Predicted PM2.5 (µg/m³)")
    ax[1].legend(loc='upper left')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(FIGURE_DIR, "fig2_scatter_actual_vs_pred.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # Figure 3: Residual Distribution
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(res_30, bins=50, density=True, alpha=0.6, color='#1f77b4', edgecolor='black', label='Residual Error')
    kde_x_30 = np.linspace(np.min(res_30), np.max(res_30), 200)
    kde_30 = stats.gaussian_kde(res_30)(kde_x_30)
    ax[0].plot(kde_x_30, kde_30, 'r-', linewidth=2, label='KDE Density')
    ax[0].axvline(0, color='black', linestyle='--', linewidth=1)
    ax[0].set_title("A. Residual Error (+30 Min)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Error: $y_{true} - y_{pred}$ (µg/m³)")
    ax[0].set_ylabel("Density")
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].hist(res_60, bins=50, density=True, alpha=0.6, color='#2ca02c', edgecolor='black', label='Residual Error')
    kde_x_60 = np.linspace(np.min(res_60), np.max(res_60), 200)
    kde_60 = stats.gaussian_kde(res_60)(kde_x_60)
    ax[1].plot(kde_x_60, kde_60, 'r-', linewidth=2, label='KDE Density')
    ax[1].axvline(0, color='black', linestyle='--', linewidth=1)
    ax[1].set_title("B. Residual Error (+60 Min)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Error: $y_{true} - y_{pred}$ (µg/m³)")
    ax[1].set_ylabel("Density")
    ax[1].legend(loc='upper right')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = os.path.join(FIGURE_DIR, "fig3_residual_distribution.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # Figure 4: Model Comparison Bar Chart
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    model_names_short = ["Persistence", "AR(24)", "Ridge", "Random Forest", "CNN-BiLSTM (Full)", "CNN-BiLSTM-Attn", "CNN-BiLSTM (F1)"]
    mae_30_vals = [summary_metrics[m]["MAE_30"] for m in summary_metrics]
    rmse_30_vals = [summary_metrics[m]["RMSE_30"] for m in summary_metrics]
    r2_30_vals = [summary_metrics[m]["R2_30"] for m in summary_metrics]
    colors = ['#7f7f7f', '#17becf', '#aec7e8', '#ffbb78', '#ff7f0e', '#d62728', '#2ca02c']
    
    b0 = ax[0].barh(model_names_short, mae_30_vals, color=colors)
    ax[0].set_title("MAE (+30 Min)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("µg/m³")
    ax[0].grid(True, alpha=0.3)
    ax[0].bar_label(b0, fmt='%.3f', padding=3, fontsize=9)
    ax[0].set_xlim(0, max(mae_30_vals) * 1.15)
    
    b1 = ax[1].barh(model_names_short, rmse_30_vals, color=colors)
    ax[1].set_title("RMSE (+30 Min)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("µg/m³")
    ax[1].grid(True, alpha=0.3)
    ax[1].bar_label(b1, fmt='%.4f', padding=4, fontsize=9)
    ax[1].set_xlim(0, max(rmse_30_vals) * 1.18)
    
    b2 = ax[2].barh(model_names_short, r2_30_vals, color=colors)
    ax[2].set_title("R² Score (+30 Min)", fontsize=12, fontweight='bold')
    ax[2].set_xlabel("R² Score")
    min_r2 = min(r2_30_vals)
    ax[2].set_xlim(min(min_r2 * 1.25, -0.25), 0.88)
    ax[2].axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.7)
    ax[2].grid(True, alpha=0.3)
    ax[2].tick_params(axis='y', pad=8)
    ax[2].bar_label(b2, fmt='%.3f', padding=5, fontsize=9)
    
    plt.tight_layout()
    fig4_path = os.path.join(FIGURE_DIR, "fig4_model_comparison_bar.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    reproducibility_params = {
        "random_seed": TEST_SEED,
        "n_bootstraps": N_BOOTSTRAPS,
        "bootstrap_block_sizes_steps": BLOCK_SIZES,
        "bootstrap_block_durations_hours": [b * 0.5 for b in BLOCK_SIZES],
        "bootstrap_primary_block": "block_48 (24.0 hours / 1 diurnal cycle)",
        "paired_sampling": True,
        "dm_loss_power": 2,
        "dm_hac_kernel": "Bartlett (Newey-West)",
        "dm_bandwidth_rule": "max(h-1, ceil(4*(n/100)^(2/9)))",
        "dm_finite_sample_correction": "Harvey-Leybourne-Newbold (1997)",
        "dm_distribution": "Student-t (df = n - 1)",
        "test_sample_count": int(len(y_test_unscaled)),
        "execution_timestamp": str(pd.Timestamp.now())
    }

    report = {
        "status": "SUCCESS",
        "step_11a_final_performance_summary": summary_metrics,
        "step_11b_relative_improvements_over_persistence": improvements,
        "step_11c_statistical_significance_tests": stat_tests,
        "step_11d_residual_and_error_analysis": {
            "residual_statistics": residual_stats,
            "error_by_concentration_category": cat_breakdown,
            "top5_extreme_error_events": top5_extreme_events
        },
        "step_11e_generated_figures": {
            "fig1_timeseries": fig1_path,
            "fig2_scatter": fig2_path,
            "fig3_residuals": fig3_path,
            "fig4_comparison": fig4_path
        },
        "step_11f_reproducibility_parameters": reproducibility_params
    }
    
    out_path = os.path.join(SCRATCH_DIR, "step11_final_statistical_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    # Also save to project reports and figures directories
    project_report_path = r"c:\Users\INTAN\airpollutan\reports\step11_final_statistical_report.json"
    os.makedirs(os.path.dirname(project_report_path), exist_ok=True)
    with open(project_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    import shutil
    project_fig_dir = r"c:\Users\INTAN\airpollutan\figures"
    os.makedirs(project_fig_dir, exist_ok=True)
    for fig_f in [fig1_path, fig2_path, fig3_path, fig4_path]:
        shutil.copy2(fig_f, os.path.join(project_fig_dir, os.path.basename(fig_f)))

    print(f"\nStep 11 Instant Evaluation Completed Successfully! Saved report to {out_path} and {project_report_path}")

if __name__ == "__main__":
    main()
