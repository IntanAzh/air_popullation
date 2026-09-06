import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from model_keras_cnn_bilstm_attention import TemporalAttentionLayer
from scaler_utils import load_scaler_and_meta, inverse_transform_targets

# Set aesthetic style for publication-ready journal figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"
PLOT_DIR = r"c:\Users\INTAN\airpollutan\static\plots"

def generate_all_visualizations():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=" * 85)
    print("MENGHASILKAN VISUALISASI GRAFIK JURNAL & DASHBOARD (HIGH RESOLUTION 300 DPI)")
    print("=" * 85)

    # Load Data & Model
    X_test = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    model_path = os.path.join(MODEL_DIR, "best_keras_cnn_bilstm_attention.keras")
    model = keras.models.load_model(
        model_path, 
        custom_objects={'TemporalAttentionLayer': TemporalAttentionLayer}
    )

    preds_scaled = model.predict(X_test, verbose=0)

    scaler, meta = load_scaler_and_meta()
    y_pred_physical = inverse_transform_targets(preds_scaled, scaler, meta['target_indices'])
    y_true_physical = inverse_transform_targets(y_test, scaler, meta['target_indices'])

    # -----------------------------------------------------------------------------
    # GRAFIK 1: Time-Series Predictions vs Actual (PM2.5 & PM10)
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    steps_to_plot = 200  # Display 200 time steps (~100 hours) for clarity

    # PM2.5 Plot
    axes[0].plot(y_true_physical[:steps_to_plot, 0, 0], label='PM2.5 Aktual (Observed)', color='#1f77b4', linewidth=2.0)
    axes[0].plot(y_pred_physical[:steps_to_plot, 0, 0], label='PM2.5 Prediksi (CNN-BiLSTM-Attention)', color='#ff7f0e', linestyle='--', linewidth=2.0)
    axes[0].set_title('Peramalan Time-Series PM2.5 (30 Menit ke Depan)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel('Konsentrasi (µg/m³)', fontsize=12)
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # PM10 Plot
    axes[1].plot(y_true_physical[:steps_to_plot, 0, 1], label='PM10 Aktual (Observed)', color='#2ca02c', linewidth=2.0)
    axes[1].plot(y_pred_physical[:steps_to_plot, 0, 1], label='PM10 Prediksi (CNN-BiLSTM-Attention)', color='#d62728', linestyle='--', linewidth=2.0)
    axes[1].set_title('Peramalan Time-Series PM10 (30 Menit ke Depan)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Langkah Waktu (30-Minute Interval Steps)', fontsize=12)
    axes[1].set_ylabel('Konsentrasi (µg/m³)', fontsize=12)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot1_path = os.path.join(PLOT_DIR, "actual_vs_predicted_timeseries.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"1. Saved Grafik Time-Series    : {plot1_path}")

    # -----------------------------------------------------------------------------
    # GRAFIK 2: Scatter Plot Aktual vs Prediksi dengan Line y = x
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (col_name, color) in enumerate(zip(['PM2.5', 'PM10'], ['#1f77b4', '#2ca02c'])):
        yt = y_true_physical[:, :, idx].flatten()
        yp = y_pred_physical[:, :, idx].flatten()

        axes[idx].scatter(yt, yp, alpha=0.4, color=color, edgecolors='w', s=35, label='Titik Prediksi')
        
        # Line y = x (Ideal Match)
        max_val = max(yt.max(), yp.max())
        min_val = min(yt.min(), yp.min())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Garis Ideal (y = x)')

        axes[idx].set_title(f'Scatter Plot Regresi {col_name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(f'Nilai Aktual {col_name} (µg/m³)', fontsize=12)
        axes[idx].set_ylabel(f'Nilai Prediksi {col_name} (µg/m³)', fontsize=12)
        axes[idx].legend(loc='upper left', frameon=True)
        axes[idx].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot2_path = os.path.join(PLOT_DIR, "scatter_regression_metrics.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2. Saved Grafik Scatter Regresi : {plot2_path}")

    # -----------------------------------------------------------------------------
    # GRAFIK 3: Distribusi Residual Errors (Error Margin Histogram & KDE)
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (col_name, color) in enumerate(zip(['PM2.5', 'PM10'], ['#ff7f0e', '#d62728'])):
        yt = y_true_physical[:, :, idx].flatten()
        yp = y_pred_physical[:, :, idx].flatten()
        residuals = yt - yp

        sns.histplot(residuals, kde=True, ax=axes[idx], color=color, bins=30, stat='density', alpha=0.6)
        axes[idx].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Residual Error = 0')
        axes[idx].set_title(f'Distribusi Error Residual {col_name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Selisih Error (Aktual - Prediksi) µg/m³', fontsize=12)
        axes[idx].set_ylabel('Kerapatan (Density)', fontsize=12)
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot3_path = os.path.join(PLOT_DIR, "residual_error_distribution.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3. Saved Grafik Residual Error : {plot3_path}")

    # -----------------------------------------------------------------------------
    # GRAFIK 4: Komparasi Benchmark Model (Bar Chart MAE & RMSE)
    # -----------------------------------------------------------------------------
    baseline_path = os.path.join(MODEL_DIR, "baseline_metrics.json")
    final_path = os.path.join(MODEL_DIR, "final_test_evaluation_report.json")

    if os.path.exists(baseline_path) and os.path.exists(final_path):
        with open(baseline_path) as f:
            base_m = json.load(f)
        with open(final_path) as f:
            proposed_m = json.load(f)

        models = ['Ridge', 'RandomForest', 'Proposed CNN-BiLSTM-Attn']
        pm25_mae = [base_m['Ridge']['PM2.5']['MAE'], base_m['RandomForest']['PM2.5']['MAE'], proposed_m['PM2.5']['MAE_ug_m3']]
        pm10_mae = [base_m['Ridge']['PM10']['MAE'], base_m['RandomForest']['PM10']['MAE'], proposed_m['PM10']['MAE_ug_m3']]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35

        rects1 = ax.bar(x - width/2, pm25_mae, width, label='PM2.5 MAE (µg/m³)', color='#3498db')
        rects2 = ax.bar(x + width/2, pm10_mae, width, label='PM10 MAE (µg/m³)', color='#e74c3c')

        ax.set_ylabel('Mean Absolute Error (µg/m³)', fontsize=12)
        ax.set_title('Komparasi Metrik MAE Model Baseline vs Proposed CNN-BiLSTM-Attention', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')

        # Add bar labels
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        plt.tight_layout()
        plot4_path = os.path.join(PLOT_DIR, "model_benchmark_comparison.png")
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"4. Saved Grafik Komparasi Model: {plot4_path}")

    print("=" * 85)
    print("SELURUH 4 GRAFIK VISUALISASI SELESAI DIGENERASI DI FOLDER static/plots/")
    print("=" * 85)

if __name__ == "__main__":
    generate_all_visualizations()
