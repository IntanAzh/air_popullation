import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.abspath("."))

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')
else:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device('cpu')

print(f"Device for baseline training: {device}")

class FlexibleCNNBiLSTM(nn.Module):
    def __init__(self, input_dim, conv_filters=64, lstm_hidden=64, dropout_rate=0.3):
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

def retrain_baselines():
    print("=" * 80)
    print("MELATIH ULANG MODEL BASELINE (RIDGE, RANDOM FOREST, B6 CNN-BiLSTM 94 FITUR)")
    print("=" * 80)

    train_npz = np.load(os.path.join(TENSOR_DIR, "train_tensors.npz"))
    val_npz = np.load(os.path.join(TENSOR_DIR, "val_tensors.npz"))
    test_npz = np.load(os.path.join(TENSOR_DIR, "test_tensors.npz"))

    X_train, y_train_s, y_train_u = train_npz['X'], train_npz['y_scaled'], train_npz['y_unscaled']
    X_val, y_val_s, y_val_u = val_npz['X'], val_npz['y_scaled'], val_npz['y_unscaled']
    X_test, y_test_s, y_test_u = test_npz['X'], test_npz['y_scaled'], test_npz['y_unscaled']
    
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.joblib"))

    # Flatten for Tabular ML (samples, 48 * 94 = 4512 features)
    X_tr_2d = X_train.reshape(len(X_train), -1)
    X_va_2d = X_val.reshape(len(X_val), -1)
    X_te_2d = X_test.reshape(len(X_test), -1)

    # 1. Ridge Regression (B1) with validation tuning
    print("\n[1/3] Melatih Ridge Regression (B1)...")
    best_alpha = 100.0
    best_val_rmse = float('inf')
    for alpha in [1.0, 10.0, 100.0, 500.0, 1000.0]:
        r = Ridge(alpha=alpha, random_state=42)
        r.fit(X_tr_2d, y_train_s)
        val_pred_u = y_scaler.inverse_transform(r.predict(X_va_2d))
        rmse = float(np.sqrt(mean_squared_error(y_val_u, val_pred_u)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_alpha = alpha

    print(f"  -> Best Alpha: {best_alpha} (Val RMSE: {best_val_rmse:.4f})")
    ridge = Ridge(alpha=best_alpha, random_state=42)
    ridge.fit(X_tr_2d, y_train_s)
    joblib.dump(ridge, os.path.join(MODEL_DIR, "B1_Ridge.joblib"))
    print("  -> B1_Ridge.joblib tersimpan.")

    # 2. Random Forest Regressor (B2)
    print("\n[2/3] Melatih Random Forest (B2, n_estimators=100, max_depth=10)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_2d, y_train_s)
    joblib.dump(rf, os.path.join(MODEL_DIR, "B2_RandomForest.joblib"))
    print("  -> B2_RandomForest.joblib tersimpan.")

    # 3. B6: CNN-BiLSTM (Full 94 Features)
    print("\n[3/3] Melatih CNN-BiLSTM Full 94 Fitur (B6)...")
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_s, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32))
    
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

    m_b6 = FlexibleCNNBiLSTM(input_dim=X_train.shape[2], conv_filters=64, lstm_hidden=64, dropout_rate=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(m_b6.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    patience = 6
    patience_counter = 0

    for epoch in range(1, 40):
        m_b6.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = m_b6(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        m_b6.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = m_b6(bx)
                val_loss += criterion(pred, by).item() * len(bx)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in m_b6.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping pada epoch {epoch} (Best Val Loss: {best_val_loss:.6f})")
                break

    torch.save(best_state, os.path.join(MODEL_DIR, "B6_CNN_BiLSTM.pt"))
    print("  -> B6_CNN_BiLSTM.pt tersimpan.")
    print("\n[OK] Pelatihan baseline selesai!")

if __name__ == "__main__":
    retrain_baselines()
