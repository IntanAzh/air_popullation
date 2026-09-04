import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_cnn_bilstm_attention import CNN_BiLSTM_Attention
from scaler_utils import load_scaler_and_meta, inverse_transform_targets

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

def compute_mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0

def train_and_eval():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Load 3D tensor arrays
    X_train_full = np.load(os.path.join(TENSOR_DIR, "X_train.npy"))
    y_train_full = np.load(os.path.join(TENSOR_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    # Strictly Chronological Validation Split from end of Train set (10% of train)
    n_train_sub = int(len(X_train_full) * 0.90)
    X_train, y_train = X_train_full[:n_train_sub], y_train_full[:n_train_sub]
    X_val, y_val = X_train_full[n_train_sub:], y_train_full[n_train_sub:]

    print(f"X_train (80% Chronological): {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_val   (Chronological Val):  {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"X_test  (20% Chronological): {X_test.shape}  | y_test:  {y_test.shape}")

    # DataLoader without shuffling (shuffle=False to strictly respect time-series sequence)
    batch_size = 128
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    horizon = y_train.shape[1]
    n_targets = y_train.shape[2]

    model = CNN_BiLSTM_Attention(in_features=in_features, seq_len=seq_len, horizon=horizon, n_targets=n_targets).to(device)
    criterion = nn.HuberLoss(delta=0.1)  # Huber loss for robustness against sudden industrial spikes
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = 8
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODEL_DIR, "best_cnn_bilstm_attention.pth")

    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING (CNN-BiLSTM with Attention Mechanism)")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred, _ = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred, _ = model(X_b)
                loss = criterion(pred, y_b)
                val_loss += loss.item() * len(X_b)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.6f}")
    print(f"Saved Checkpoint: {best_model_path}")

    print("\n" + "=" * 60)
    print("EVALUATING ON 20% CHRONOLOGICAL TEST SET (ug/m3)")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device)
            pred, _ = model(X_b)
            test_preds.append(pred.cpu().numpy())
            
    y_pred_scaled = np.concatenate(test_preds, axis=0)

    scaler, meta = load_scaler_and_meta()
    y_pred_unscaled = inverse_transform_targets(y_pred_scaled, scaler, meta['target_indices'])
    y_test_unscaled = inverse_transform_targets(y_test, scaler, meta['target_indices'])

    metrics_results = {}
    target_names = meta['target_cols']  # ['PM2.5', 'PM10']

    for i, name in enumerate(target_names):
        y_t_true = y_test_unscaled[:, :, i].flatten()
        y_t_pred = y_pred_unscaled[:, :, i].flatten()

        mae = mean_absolute_error(y_t_true, y_t_pred)
        rmse = np.sqrt(mean_squared_error(y_t_true, y_t_pred))
        mape = compute_mape(y_t_true, y_t_pred)
        r2 = r2_score(y_t_true, y_t_pred)

        metrics_results[name] = {
            'MAE_ug_m3': round(float(mae), 4),
            'RMSE_ug_m3': round(float(rmse), 4),
            'MAPE_percent': round(float(mape), 4),
            'R2_Score': round(float(r2), 4)
        }

        print(f"\n--- EVALUATION METRICS FOR {name} (CHRONOLOGICAL TEST SET 20%) ---")
        print(f"  MAE  : {mae:.4f} ug/m3")
        print(f"  RMSE : {rmse:.4f} ug/m3")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  R2   : {r2:.4f}")

    metrics_path = os.path.join(MODEL_DIR, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)

    print(f"\nSaved Evaluation Metrics Report: {metrics_path}")

if __name__ == "__main__":
    train_and_eval()
