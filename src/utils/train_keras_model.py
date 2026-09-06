import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import numpy as np
import keras
from keras import callbacks, optimizers, losses
import joblib

from model_keras_cnn_bilstm_attention import build_keras_cnn_bilstm_attention
from scaler_utils import load_scaler_and_meta, inverse_transform_targets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

def compute_mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0

def train_and_validate_keras():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load 3D Tensors
    X_train_full = np.load(os.path.join(TENSOR_DIR, "X_train.npy"))
    y_train_full = np.load(os.path.join(TENSOR_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    # Strictly Chronological Validation Split from end of Train set (10% validation)
    n_train_sub = int(len(X_train_full) * 0.90)
    X_train, y_train = X_train_full[:n_train_sub], y_train_full[:n_train_sub]
    X_val, y_val = X_train_full[n_train_sub:], y_train_full[n_train_sub:]

    print("=" * 65)
    print("KERAS MODEL TRAINING & VALIDATION WITH EARLY STOPPING")
    print("=" * 65)
    print(f"X_train (3D): {X_train.shape} | y_train (3D): {y_train.shape}")
    print(f"X_val   (3D): {X_val.shape}   | y_val   (3D): {y_val.shape}")
    print(f"X_test  (3D): {X_test.shape}  | y_test  (3D): {y_test.shape}")

    seq_len = X_train.shape[1]
    in_features = X_train.shape[2]
    horizon = y_train.shape[1]
    n_targets = y_train.shape[2]

    # 2. Build Keras Hybrid Model (Conv1D -> MaxPool1D -> BiLSTM -> Attention -> Dropout -> Dense)
    model = build_keras_cnn_bilstm_attention(
        seq_len=seq_len, 
        in_features=in_features, 
        horizon=horizon, 
        n_targets=n_targets
    )

    # 3. Compile Model (Adam Optimizer and Mean Squared Error / MSE Loss)
    print("\n1. Compiling model with Adam Optimizer and MSE (Mean Squared Error) Loss...")
    adam_optimizer = optimizers.Adam(learning_rate=0.001)
    mse_loss = losses.MeanSquaredError()

    model.compile(
        optimizer=adam_optimizer,
        loss=mse_loss,
        metrics=['mae', 'mse']
    )

    # 4. Define Early Stopping & Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    model_save_path = os.path.join(MODEL_DIR, "best_keras_cnn_bilstm_attention.keras")
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 5. Execute Training (model.fit) with Early Stopping
    print("\n2. Executing model.fit() training with Early Stopping...")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    print(f"\nTraining completed! Saved best model to: {model_save_path}")

    # 6. Evaluation on Test Set
    print("\n" + "=" * 65)
    print("EVALUATING KERAS MODEL ON CHRONOLOGICAL TEST SET (ug/m3)")
    print("=" * 65)

    preds_scaled = model.predict(X_test, verbose=0)
    
    scaler, meta = load_scaler_and_meta()
    y_pred_unscaled = inverse_transform_targets(preds_scaled, scaler, meta['target_indices'])
    y_test_unscaled = inverse_transform_targets(y_test, scaler, meta['target_indices'])

    target_names = meta['target_cols']  # ['PM2.5', 'PM10']
    metrics_results = {}

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

        print(f"\n--- EVALUATION METRICS FOR {name} (TEST SET 20%) ---")
        print(f"  MAE  : {mae:.4f} ug/m3")
        print(f"  RMSE : {rmse:.4f} ug/m3")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  R2   : {r2:.4f}")

    metrics_path = os.path.join(MODEL_DIR, "keras_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)

    print(f"\nSaved Keras Evaluation Metrics Report: {metrics_path}")

if __name__ == "__main__":
    train_and_validate_keras()
