import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import numpy as np
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_keras_cnn_bilstm_attention import TemporalAttentionLayer
from scaler_utils import load_scaler_and_meta, inverse_transform_targets

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

def compute_mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0

def evaluate_model_on_test_data():
    print("=" * 85)
    print("EVALUASI PREDIKSI KEANDALAN MODEL PADA TEST DATA (ISOLATED 20% TEST SET)")
    print("=" * 85)

    # 1. Load Isolated Test Data Tensors
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    print(f"Bentuk Test Data Input Tensor (X_test): {X_test.shape}")
    print(f"Bentuk Test Data Target Tensor (y_test): {y_test.shape}")

    # 2. Load Trained Keras Model Checkpoint with Custom Objects
    model_path = os.path.join(MODEL_DIR, "best_keras_cnn_bilstm_attention.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    print(f"\nMemuat Model Terlatih dari: {model_path}")
    model = keras.models.load_model(
        model_path, 
        custom_objects={'TemporalAttentionLayer': TemporalAttentionLayer}
    )

    # 3. Predict on Test Data (Scaled Predictions)
    print("Melakukan Prediksi pada Isolated Test Data...")
    preds_scaled = model.predict(X_test, verbose=0)

    # 4. Inverse Transform Scaled Predictions & Targets back to Physical Units (ug/m3)
    scaler, meta = load_scaler_and_meta()
    y_pred_physical = inverse_transform_targets(preds_scaled, scaler, meta['target_indices'])
    y_true_physical = inverse_transform_targets(y_test, scaler, meta['target_indices'])

    target_names = meta['target_cols']  # ['PM2.5', 'PM10']
    evaluation_summary = {}

    print("\n" + "=" * 85)
    print("HASIL PERHITUNGAN METRIK STANDAR PERAMALAN KUALITAS UDARA (SATUAN ug/m3)")
    print("=" * 85)
    print(f"{'Parameter Polutan':<20} | {'MAE (ug/m3)':<15} | {'RMSE (ug/m3)':<15} | {'MAPE (%)':<12} | {'R2 Score':<12}")
    print("-" * 85)

    for i, name in enumerate(target_names):
        yt = y_true_physical[:, :, i].flatten()
        yp = y_pred_physical[:, :, i].flatten()

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mape = compute_mape(yt, yp)
        r2 = r2_score(yt, yp)

        evaluation_summary[name] = {
            'MAE_ug_m3': round(float(mae), 4),
            'RMSE_ug_m3': round(float(rmse), 4),
            'MAPE_percent': round(float(mape), 4),
            'R2_Score': round(float(r2), 4)
        }

        print(f"{name:<20} | {mae:<15.4f} | {rmse:<15.4f} | {mape:<12.2f} | {r2:<12.4f}")

    # Save Metrics Report JSON
    report_path = os.path.join(MODEL_DIR, "final_test_evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=4)

    print("-" * 85)
    print(f"Laporan Metrik Evaluasi Tersimpan di: {report_path}")
    print("=" * 85)
    print("KESIMPULAN: METRIK STANDAR PERAMALAN TERHITUNG KONTINU PADA SATUAN FISIK ASLI!")
    print("=" * 85)

if __name__ == "__main__":
    evaluate_model_on_test_data()
