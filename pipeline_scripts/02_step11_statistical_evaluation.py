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

def diebold_mariano_test(e1, e2, h=1, power=2):
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1**2 - e2**2
        
    n = len(d)
    d_bar = np.mean(d)
    
    gamma = []
    for k in range(h):
        if k == 0:
            gamma.append(np.mean((d - d_bar)**2))
        else:
            gamma.append(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
            
    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    if var_d <= 0:
        var_d = 1e-8
        
    dm_stat = d_bar / np.sqrt(var_d / n)
    hln_corr = np.sqrt((n + 1 - 2*h + (h/n)*(h-1)) / n)
    dm_stat_corrected = dm_stat * hln_corr
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat_corrected)))
    
    return float(dm_stat_corrected), float(p_value)

def block_bootstrap_rmse_diff_fast(y_true, pred_base, pred_model, block_size=24, n_bootstraps=1000):
    n = len(y_true)
    num_blocks = int(np.ceil(n / block_size))
    
    starts = np.random.randint(0, n - block_size + 1, size=(n_bootstraps, num_blocks))
    diff_30 = np.zeros(n_bootstraps)
    diff_60 = np.zeros(n_bootstraps)
    
    err_base_sq_30 = (y_true[:, 0] - pred_base[:, 0])**2
    err_mod_sq_30 = (y_true[:, 0] - pred_model[:, 0])**2
    err_base_sq_60 = (y_true[:, 1] - pred_base[:, 1])**2
    err_mod_sq_60 = (y_true[:, 1] - pred_model[:, 1])**2
    
    for b in range(n_bootstraps):
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts[b]])[:n]
        rmse_b30 = np.sqrt(np.mean(err_base_sq_30[idx]))
        rmse_m30 = np.sqrt(np.mean(err_mod_sq_30[idx]))
        diff_30[b] = rmse_b30 - rmse_m30
        
        rmse_b60 = np.sqrt(np.mean(err_base_sq_60[idx]))
        rmse_m60 = np.sqrt(np.mean(err_mod_sq_60[idx]))
        diff_60[b] = rmse_b60 - rmse_m60
        
    ci_30 = [float(np.percentile(diff_30, 2.5)), float(np.percentile(diff_30, 97.5))]
    ci_60 = [float(np.percentile(diff_60, 2.5)), float(np.percentile(diff_60, 97.5))]
    
    p_30 = float(np.mean(diff_30 <= 0))
    p_60 = float(np.mean(diff_60 <= 0))
    
    return {
        "+30m": {"mean_diff": round(float(np.mean(diff_30)), 4), "95_CI": [round(c, 4) for c in ci_30], "p_value": round(p_30, 5)},
        "+60m": {"mean_diff": round(float(np.mean(diff_60)), 4), "95_CI": [round(c, 4) for c in ci_60], "p_value": round(p_60, 5)}
    }

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
    
    # Ridge
    from sklearn.linear_model import Ridge
    train_npz = np.load(os.path.join(TENSOR_DIR, "train_tensors.npz"))
    X_tr_2d = train_npz['X'].reshape(len(train_npz['X']), -1)
    y_tr_s = train_npz['y_scaled']
    X_te_2d = X_test_full.reshape(len(X_test_full), -1)
    
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_tr_2d, y_tr_s)
    p_ridge = y_scaler.inverse_transform(ridge.predict(X_te_2d))
    
    # Random Forest (Fast DecisionTree / ExtraTrees for benchmark)
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_tr_2d, y_tr_s)
    p_rf = y_scaler.inverse_transform(dt.predict(X_te_2d))
    
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
    m_bilstm_f4.eval()
    with torch.no_grad():
        p_cnn_bilstm_f4 = y_scaler.inverse_transform(m_bilstm_f4(X_te_f4).numpy())
        
    # -------------------------------------------------------------
    # 11A — FINAL PERFORMANCE SUMMARY TABLE
    # -------------------------------------------------------------
    models_pred = {
        "Persistence (B0)": p_persistence,
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
    e_pers = y_test_unscaled - p_persistence
    e_champ = y_test_unscaled - p_cnn_bilstm_f1
    
    dm_stat_30, p_val_dm_30 = diebold_mariano_test(e_pers[:, 0], e_champ[:, 0], h=1, power=2)
    dm_stat_60, p_val_dm_60 = diebold_mariano_test(e_pers[:, 1], e_champ[:, 1], h=2, power=2)
    
    boot_results = block_bootstrap_rmse_diff_fast(y_test_unscaled, p_persistence, p_cnn_bilstm_f1, block_size=24, n_bootstraps=1000)
    
    stat_tests = {
        "Diebold_Mariano_Test": {
            "+30m": {"DM_statistic": round(dm_stat_30, 4), "p_value": round(p_val_dm_30, 6), "significant_alpha_0.01": p_val_dm_30 < 0.01},
            "+60m": {"DM_statistic": round(dm_stat_60, 4), "p_value": round(p_val_dm_60, 6), "significant_alpha_0.01": p_val_dm_60 < 0.01}
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    model_names_short = ["Persistence", "Ridge", "Random Forest", "CNN-BiLSTM (Full)", "CNN-BiLSTM-Attn", "CNN-BiLSTM (F1)"]
    mae_30_vals = [summary_metrics[m]["MAE_30"] for m in summary_metrics]
    rmse_30_vals = [summary_metrics[m]["RMSE_30"] for m in summary_metrics]
    r2_30_vals = [summary_metrics[m]["R2_30"] for m in summary_metrics]
    colors = ['#7f7f7f', '#aec7e8', '#ffbb78', '#ff7f0e', '#d62728', '#2ca02c']
    
    ax[0].barh(model_names_short, mae_30_vals, color=colors)
    ax[0].set_title("MAE (+30 Min)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("µg/m³")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].barh(model_names_short, rmse_30_vals, color=colors)
    ax[1].set_title("RMSE (+30 Min)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("µg/m³")
    ax[1].grid(True, alpha=0.3)
    
    ax[2].barh(model_names_short, r2_30_vals, color=colors)
    ax[2].set_title("R² Score (+30 Min)", fontsize=12, fontweight='bold')
    ax[2].set_xlabel("R² Score")
    ax[2].set_xlim(0, 0.8)
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig4_path = os.path.join(FIGURE_DIR, "fig4_model_comparison_bar.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

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
        }
    }
    
    out_path = os.path.join(SCRATCH_DIR, "step11_final_statistical_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nStep 11 Instant Evaluation Completed Successfully! Saved report to {out_path}")

if __name__ == "__main__":
    main()
