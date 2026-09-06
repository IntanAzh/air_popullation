import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = r"c:\Users\INTAN\airpollutan\data\processed"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"
TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
SCRATCH_DIR = r"C:\Users\INTAN\.gemini\antigravity\brain\5c3a30c5-8ff2-4f5c-bacf-cc114ea67f12\scratch"

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')
else:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device('cpu')

print(f"Using device: {device}")

def eval_metrics(y_true, y_pred):
    mae_30 = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    rmse_30 = float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])))
    r2_30 = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    
    mae_60 = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    rmse_60 = float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])))
    r2_60 = float(r2_score(y_true[:, 1], y_pred[:, 1]))
    
    mae_avg = float((mae_30 + mae_60) / 2.0)
    rmse_avg = float((rmse_30 + rmse_60) / 2.0)
    r2_avg = float((r2_30 + r2_60) / 2.0)
    
    return {
        "MAE_30": round(mae_30, 4), "RMSE_30": round(rmse_30, 4), "R2_30": round(r2_30, 4),
        "MAE_60": round(mae_60, 4), "RMSE_60": round(rmse_60, 4), "R2_60": round(r2_60, 4),
        "MAE_avg": round(mae_avg, 4), "RMSE_avg": round(rmse_avg, 4), "R2_avg": round(r2_avg, 4)
    }

class FlexibleCNNBiLSTM(nn.Module):
    def __init__(self, input_dim, conv_filters=32, lstm_hidden=32, dropout_rate=0.3):
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
    def __init__(self, input_dim, conv_filters=32, lstm_hidden=32, dropout_rate=0.3):
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

def train_with_checkpointing(model, train_loader, val_loader, epochs=40, patience=6):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * len(bx)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val_loss, best_epoch

def run_step10_fast():
    print("=== STARTING STEP 10 OPTIMIZATION PIPELINE (FAST) ===")
    
    df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_dataset.parquet"))
    all_cols = [c for c in df_train.columns if c not in ['datetime', 'target_PM2.5_30m', 'target_PM2.5_60m']]
    
    f1_cols = [c for c in all_cols if 'PM2.5' in c]
    meteo_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'solar_radiation', 'rainfall', 'u_wind', 'v_wind']
    f2_cols = [c for c in all_cols if any(m in c for m in ['PM2.5'] + meteo_names)]
    pol_names = ['PM2.5', 'NO2', 'SO2']
    f3_cols = [c for c in all_cols if any(p in c for p in pol_names)]
    f4_cols = all_cols
    
    feature_sets = {
        "F1_PM2.5_History_Only": f1_cols,
        "F2_PM2.5_Plus_Meteorology": f2_cols,
        "F3_PM2.5_Plus_Pollutants": f3_cols,
        "F4_Full_Model": f4_cols
    }
    
    train_npz = np.load(os.path.join(TENSOR_DIR, "train_tensors.npz"))
    val_npz = np.load(os.path.join(TENSOR_DIR, "val_tensors.npz"))
    test_npz = np.load(os.path.join(TENSOR_DIR, "test_tensors.npz"))
    
    X_tr_full, y_tr_s, y_tr_u = train_npz['X'], train_npz['y_scaled'], train_npz['y_unscaled']
    X_val_full, y_val_s, y_val_u = val_npz['X'], val_npz['y_scaled'], val_npz['y_unscaled']
    X_te_full, y_te_s, y_te_u = test_npz['X'], test_npz['y_scaled'], test_npz['y_unscaled']
    
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.joblib"))
    
    configs_to_test = [
        {"conv": 32, "lstm": 32, "drop": 0.2},
        {"conv": 32, "lstm": 32, "drop": 0.3},
        {"conv": 64, "lstm": 32, "drop": 0.3},
        {"conv": 64, "lstm": 64, "drop": 0.3},
        {"conv": 64, "lstm": 64, "drop": 0.5}
    ]
    
    print("\n--- STEP 10B & 10C: Validation Grid Search ---")
    val_search_results = []
    
    for fs_name, fcols in feature_sets.items():
        col_indices = [all_cols.index(c) for c in fcols]
        X_tr_f = X_tr_full[:, :, col_indices]
        X_val_f = X_val_full[:, :, col_indices]
        
        tr_ds = TensorDataset(torch.tensor(X_tr_f, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32))
        va_ds = TensorDataset(torch.tensor(X_val_f, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32))
        
        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
        
        for cfg in configs_to_test:
            model = FlexibleCNNBiLSTM(input_dim=len(fcols), conv_filters=cfg["conv"], lstm_hidden=cfg["lstm"], dropout_rate=cfg["drop"])
            model, best_val_loss, best_epoch = train_with_checkpointing(model, tr_loader, val_loader, epochs=35, patience=5)
            
            model.eval()
            with torch.no_grad():
                val_pred_s = model(torch.tensor(X_val_f, dtype=torch.float32).to(device)).cpu().numpy()
            val_pred_u = y_scaler.inverse_transform(val_pred_s)
            val_rmse = float(np.sqrt(mean_squared_error(y_val_u, val_pred_u)))
            
            val_search_results.append({
                "feature_set": fs_name,
                "feature_count": len(fcols),
                "conv_filters": cfg["conv"],
                "lstm_hidden": cfg["lstm"],
                "dropout_rate": cfg["drop"],
                "best_epoch": best_epoch,
                "val_loss_scaled": round(best_val_loss, 6),
                "val_rmse_unscaled": round(val_rmse, 4)
            })
            print(f"Validated {fs_name} (conv={cfg['conv']}, lstm={cfg['lstm']}, drop={cfg['drop']}) -> Val RMSE: {val_rmse:.4f}")
            
    val_search_results = sorted(val_search_results, key=lambda x: x["val_rmse_unscaled"])
    best_config = val_search_results[0]
    
    print("\n==========================================")
    print("STEP 10D: BEST CNN-BiLSTM LOCKED FROM VALIDATION SET:")
    print(f"Feature Set : {best_config['feature_set']} ({best_config['feature_count']} features)")
    print(f"Conv Filters: {best_config['conv_filters']}")
    print(f"LSTM Hidden : {best_config['lstm_hidden']}")
    print(f"Dropout     : {best_config['dropout_rate']}")
    print(f"Val RMSE    : {best_config['val_rmse_unscaled']} ug/m3")
    print("==========================================")
    
    # --- STEP 10E: FINAL EVALUATION ON LOCKED TEST SET (2,140 sequences) ---
    print("\n--- STEP 10E: Final Benchmark on Locked Test Set (CNN-BiLSTM vs CNN-BiLSTM-Attention) ---")
    
    best_fcols = feature_sets[best_config['feature_set']]
    best_col_idx = [all_cols.index(c) for c in best_fcols]
    
    X_tr_b = X_tr_full[:, :, best_col_idx]
    X_val_b = X_val_full[:, :, best_col_idx]
    X_te_b = X_te_full[:, :, best_col_idx]
    
    tr_ds = TensorDataset(torch.tensor(X_tr_b, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_val_b, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32))
    
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    
    X_te_tensor = torch.tensor(X_te_b, dtype=torch.float32).to(device)
    
    # 1. Train Best CNN-BiLSTM
    m_bilstm = FlexibleCNNBiLSTM(
        input_dim=len(best_fcols),
        conv_filters=best_config['conv_filters'],
        lstm_hidden=best_config['lstm_hidden'],
        dropout_rate=best_config['dropout_rate']
    )
    m_bilstm, _, _ = train_with_checkpointing(m_bilstm, tr_loader, val_loader, epochs=50, patience=7)
    m_bilstm.eval()
    with torch.no_grad():
        p_b6_s = m_bilstm(X_te_tensor).cpu().numpy()
    p_b6_u = y_scaler.inverse_transform(p_b6_s)
    res_cnn_bilstm = eval_metrics(y_te_u, p_b6_u)
    
    # 2. Add Attention: Train Best CNN-BiLSTM-Attention
    m_attn = FlexibleCNNBiLSTMAttention(
        input_dim=len(best_fcols),
        conv_filters=best_config['conv_filters'],
        lstm_hidden=best_config['lstm_hidden'],
        dropout_rate=best_config['dropout_rate']
    )
    m_attn, _, _ = train_with_checkpointing(m_attn, tr_loader, val_loader, epochs=50, patience=7)
    m_attn.eval()
    with torch.no_grad():
        p_b7_s = m_attn(X_te_tensor).cpu().numpy()
    p_b7_u = y_scaler.inverse_transform(p_b7_s)
    res_cnn_bilstm_attn = eval_metrics(y_te_u, p_b7_u)
    
    # Save Final Optimized Models
    torch.save(m_bilstm.state_dict(), os.path.join(MODEL_DIR, "Optimal_CNN_BiLSTM.pt"))
    torch.save(m_attn.state_dict(), os.path.join(MODEL_DIR, "Optimal_CNN_BiLSTM_Attention.pt"))
    
    report = {
        "status": "SUCCESS",
        "step_10a_early_stopping": "Patience 7 on val_loss with restore_best_checkpoint=True",
        "step_10bc_top_validation_search_results": val_search_results[:5],
        "step_10d_locked_best_architecture": best_config,
        "step_10e_final_test_benchmark": {
            "locked_test_sequences_count": len(y_te_u),
            "CNN_BiLSTM": res_cnn_bilstm,
            "CNN_BiLSTM_Attention": res_cnn_bilstm_attn,
            "net_gain_from_attention": {
                "R2_30_improvement": round(res_cnn_bilstm_attn["R2_30"] - res_cnn_bilstm["R2_30"], 4),
                "R2_60_improvement": round(res_cnn_bilstm_attn["R2_60"] - res_cnn_bilstm["R2_60"], 4),
                "R2_avg_improvement": round(res_cnn_bilstm_attn["R2_avg"] - res_cnn_bilstm["R2_avg"], 4),
                "RMSE_30_reduction": round(res_cnn_bilstm["RMSE_30"] - res_cnn_bilstm_attn["RMSE_30"], 4)
            }
        }
    }
    
    out_path = os.path.join(SCRATCH_DIR, "step10_optimization_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nStep 10 Optimization Pipeline completed successfully! Saved report to {out_path}")

if __name__ == "__main__":
    run_step10_fast()
