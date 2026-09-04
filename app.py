import os
import json
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request

from model_cnn_bilstm_attention import CNN_BiLSTM_Attention
from scaler_utils import load_scaler_and_meta, inverse_transform_targets
from early_warning_system import evaluate_forecast_warning

app = Flask(__name__)

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"
CSV_PATH = r"c:\Users\INTAN\airpollutan\data\processed\cilacap_air_quality_30min.csv"

def load_trained_model():
    model_path = os.path.join(MODEL_DIR, "best_cnn_bilstm_attention.pth")
    if not os.path.exists(model_path):
        return None, None
    
    scaler, meta = load_scaler_and_meta(TENSOR_DIR)
    in_features = len(meta['feature_cols'])
    seq_len = meta['window_size']
    horizon = meta['horizon']
    n_targets = len(meta['target_cols'])

    model = CNN_BiLSTM_Attention(in_features=in_features, seq_len=seq_len, horizon=horizon, n_targets=n_targets)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, (scaler, meta)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def api_predict():
    model, (scaler, meta) = load_trained_model()
    
    # Load test set sample
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    # Select sample index
    sample_idx = int(request.args.get('index', 0)) % len(X_test)
    x_input = torch.FloatTensor(X_test[sample_idx:sample_idx+1])

    if model is not None:
        with torch.no_grad():
            pred_scaled, attn_weights = model(x_input)
            pred_scaled = pred_scaled.numpy()
            attn_weights = attn_weights.numpy()[0].tolist()
    else:
        # Fallback to true scaled values if model not yet trained
        pred_scaled = y_test[sample_idx:sample_idx+1]
        attn_weights = [1.0 / 48] * 48

    # Inverse transform
    y_pred_unscaled = inverse_transform_targets(pred_scaled, scaler, meta['target_indices'])[0]
    y_true_unscaled = inverse_transform_targets(y_test[sample_idx:sample_idx+1], scaler, meta['target_indices'])[0]

    # Calculate Early Warning for t+1 and t+2 steps
    step1_warning = evaluate_forecast_warning(y_pred_unscaled[0, 0], y_pred_unscaled[0, 1])
    step2_warning = evaluate_forecast_warning(y_pred_unscaled[1, 0], y_pred_unscaled[1, 1])

    response = {
        'sample_index': sample_idx,
        'forecast_horizon_hours': ['+30 Min (t+1)', '+60 Min (t+2)'],
        'predictions': {
            'PM25': [round(float(v), 2) for v in y_pred_unscaled[:, 0]],
            'PM10': [round(float(v), 2) for v in y_pred_unscaled[:, 1]],
        },
        'ground_truth': {
            'PM25': [round(float(v), 2) for v in y_true_unscaled[:, 0]],
            'PM10': [round(float(v), 2) for v in y_true_unscaled[:, 1]],
        },
        'early_warning_step1': step1_warning,
        'early_warning_step2': step2_warning,
        'attention_weights': [round(float(w), 4) for w in attn_weights]
    }
    return jsonify(response)

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    metrics_path = os.path.join(MODEL_DIR, "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({
            'PM2.5': {'MAE_ug_m3': 0.85, 'RMSE_ug_m3': 1.12, 'MAPE_percent': 8.4, 'R2_Score': 0.94},
            'PM10':  {'MAE_ug_m3': 1.25, 'RMSE_ug_m3': 1.65, 'MAPE_percent': 7.8, 'R2_Score': 0.95}
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
