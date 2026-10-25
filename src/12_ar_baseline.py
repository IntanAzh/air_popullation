"""
================================================================================
SCRIPT: src/12_ar_baseline.py
PROJECT: Air Quality (PM2.5) Forecasting Pipeline
DESCRIPTION:
    Implements and evaluates the Autoregressive (AR) baseline model for
    multi-horizon PM2.5 forecasting (+30 min and +60 min).
    
    Ensures 100% fair and consistent benchmarking by evaluating on the exact same
    test partition (test_tensors.npz) and unscaled ground truth as CNN-BiLSTM
    and other baseline models.
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_project_paths():
    """Resolves project paths robustly whether executed from root or src/."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
    
    paths = {
        "root": project_root,
        "processed": os.path.join(project_root, "data", "processed"),
        "tensors": os.path.join(project_root, "data", "processed", "tensors"),
        "predictions": os.path.join(project_root, "data", "processed", "predictions"),
        "models": os.path.join(project_root, "models"),
        "reports": os.path.join(project_root, "reports"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 for +30 min, +60 min, and average."""
    mae_30 = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    rmse_30 = float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])))
    r2_30 = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    
    mae_60 = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    rmse_60 = float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])))
    r2_60 = float(r2_score(y_true[:, 1], y_pred[:, 1]))
    
    return {
        "MAE_30": round(mae_30, 4),
        "RMSE_30": round(rmse_30, 4),
        "R2_30": round(r2_30, 4),
        "MAE_60": round(mae_60, 4),
        "RMSE_60": round(rmse_60, 4),
        "R2_60": round(r2_60, 4),
        "MAE_avg": round((mae_30 + mae_60) / 2.0, 4),
        "RMSE_avg": round((rmse_30 + rmse_60) / 2.0, 4),
        "R2_avg": round((r2_30 + r2_60) / 2.0, 4)
    }


def main():
    paths = get_project_paths()
    print("=" * 80)
    print("  STEP 12: AUTOREGRESSIVE (AR) BASELINE MODEL BENCHMARK")
    print("=" * 80)
    
    # 1. Load Datasets & Tensors
    train_parquet_path = os.path.join(paths["processed"], "train_dataset.parquet")
    df_train = pd.read_parquet(train_parquet_path)
    all_cols = [c for c in df_train.columns if c not in ['datetime', 'target_PM2.5_30m', 'target_PM2.5_60m']]
    pm25_idx = all_cols.index('PM2.5')
    
    print(f"[*] Total input features per sequence: {len(all_cols)}")
    print(f"[*] PM2.5 target feature index: {pm25_idx} ('{all_cols[pm25_idx]}')")
    
    train_npz = np.load(os.path.join(paths["tensors"], "train_tensors.npz"))
    val_npz = np.load(os.path.join(paths["tensors"], "val_tensors.npz"))
    test_npz = np.load(os.path.join(paths["tensors"], "test_tensors.npz"))
    
    X_train = train_npz['X']
    y_train_s = train_npz['y_scaled']
    
    X_val = val_npz['X']
    y_val_s = val_npz['y_scaled']
    y_val_u = val_npz['y_unscaled']
    
    X_test = test_npz['X']
    y_test_s = test_npz['y_scaled']
    y_test_u = test_npz['y_unscaled']
    
    y_scaler = joblib.load(os.path.join(paths["models"], "y_scaler.joblib"))
    
    print(f"[*] Loaded Tensors: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]} sequences")
    print(f"[*] Sequence Window Length: {X_train.shape[1]} timesteps (24 hours lookback)")
    
    # 2. Lag Order Selection on Validation Set
    candidate_lags = [1, 2, 4, 12, 24, 48]
    val_sweep_results = {}
    best_lag = None
    best_val_rmse = float('inf')
    
    print("\n" + "-" * 80)
    print("  LAG ORDER HYPERPARAMETER SELECTION (ON VALIDATION SET)")
    print("-" * 80)
    print(f"{'Lag (p)':<10} {'Lookback Duration':<20} {'Val RMSE (+30m)':<18} {'Val RMSE (+60m)':<18} {'Val RMSE (Avg)':<15}")
    print("-" * 80)
    
    for p in candidate_lags:
        X_tr_p = X_train[:, -p:, pm25_idx]
        X_val_p = X_val[:, -p:, pm25_idx]
        
        ar_cand = LinearRegression()
        ar_cand.fit(X_tr_p, y_train_s)
        
        pred_val_s = ar_cand.predict(X_val_p)
        pred_val = y_scaler.inverse_transform(pred_val_s)
        
        metrics_val = calculate_metrics(y_val_u, pred_val)
        val_sweep_results[f"AR({p})"] = metrics_val
        
        duration = f"{p * 0.5:0.1f} hours"
        print(f"AR({p:<2d})     {duration:<20} {metrics_val['RMSE_30']:<18.4f} {metrics_val['RMSE_60']:<18.4f} {metrics_val['RMSE_avg']:<15.4f}")
        
        if metrics_val['RMSE_avg'] < best_val_rmse:
            best_val_rmse = metrics_val['RMSE_avg']
            best_lag = p

    print("-" * 80)
    print(f"[V] Optimal Lag Selected by Validation Set: AR({best_lag}) with Val RMSE Avg = {best_val_rmse:.4f}")
    
    # 3. Train Champion AR Model & Generate Predictions on Test Set
    X_train_opt = X_train[:, -best_lag:, pm25_idx]
    X_test_opt = X_test[:, -best_lag:, pm25_idx]
    
    ar_champion = LinearRegression()
    ar_champion.fit(X_train_opt, y_train_s)
    
    # Predict on Test Set
    pred_test_scaled = ar_champion.predict(X_test_opt)
    pred_test_unscaled = y_scaler.inverse_transform(pred_test_scaled)
    
    # Test Metrics
    test_metrics = calculate_metrics(y_test_u, pred_test_unscaled)
    
    # 4. Statistical Analysis via statsmodels (AIC, BIC, Lag Significance)
    X_train_opt_const = sm.add_constant(X_train_opt)
    sm_model_30 = sm.OLS(y_train_s[:, 0], X_train_opt_const).fit()
    sm_model_60 = sm.OLS(y_train_s[:, 1], X_train_opt_const).fit()
    
    sm_diagnostics = {
        "+30m": {
            "AIC": round(float(sm_model_30.aic), 2),
            "BIC": round(float(sm_model_30.bic), 2),
            "F_statistic": round(float(sm_model_30.fvalue), 2),
            "F_pvalue": float(sm_model_30.f_pvalue),
            "R2_uncentered": round(float(sm_model_30.rsquared), 4)
        },
        "+60m": {
            "AIC": round(float(sm_model_60.aic), 2),
            "BIC": round(float(sm_model_60.bic), 2),
            "F_statistic": round(float(sm_model_60.fvalue), 2),
            "F_pvalue": float(sm_model_60.f_pvalue),
            "R2_uncentered": round(float(sm_model_60.rsquared), 4)
        }
    }
    
    # Also evaluate all candidate lags on Test Set for complete reference
    test_sweep_results = {}
    for p in candidate_lags:
        X_tr_p = X_train[:, -p:, pm25_idx]
        X_te_p = X_test[:, -p:, pm25_idx]
        m = LinearRegression().fit(X_tr_p, y_train_s)
        p_te = y_scaler.inverse_transform(m.predict(X_te_p))
        test_sweep_results[f"AR({p})"] = calculate_metrics(y_test_u, p_te)
        
    # 5. Output Summary Table
    print("\n" + "=" * 80)
    print(f"  FINAL EVALUATION RESULTS: AUTOREGRESSIVE BASELINE AR({best_lag})")
    print("=" * 80)
    print(f"  Target Horizon       MAE (ug/m3)    RMSE (ug/m3)    R2 Score")
    print("-" * 80)
    print(f"  Horizon +30 Min      {test_metrics['MAE_30']:<14.4f} {test_metrics['RMSE_30']:<15.4f} {test_metrics['R2_30']:<10.4f}")
    print(f"  Horizon +60 Min      {test_metrics['MAE_60']:<14.4f} {test_metrics['RMSE_60']:<15.4f} {test_metrics['R2_60']:<10.4f}")
    print(f"  Average Multi-Step   {test_metrics['MAE_avg']:<14.4f} {test_metrics['RMSE_avg']:<15.4f} {test_metrics['R2_avg']:<10.4f}")
    print("=" * 80)
    
    # 6. Benchmark Comparison against Existing Project Models
    benchmarks_path = os.path.join(paths["reports"], "step11_final_statistical_report.json")
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path, 'r', encoding='utf-8') as f:
            b_data = json.load(f)
        existing_summary = b_data.get("step_11a_final_performance_summary", {})
        
        print("\n" + "=" * 80)
        print("  COMPARATIVE BENCHMARK TABLE (AR vs OTHER PROJECT MODELS)")
        print("=" * 80)
        print(f"{'Model Name':<28} {'MAE (+30m)':<12} {'RMSE (+30m)':<13} {'R2 (+30m)':<11} {'RMSE (+60m)':<13} {'R2 (+60m)':<10}")
        print("-" * 80)
        
        # Print AR Model
        print(f"{f'AR({best_lag}) Autoregressive':<28} {test_metrics['MAE_30']:<12.4f} {test_metrics['RMSE_30']:<13.4f} {test_metrics['R2_30']:<11.4f} {test_metrics['RMSE_60']:<13.4f} {test_metrics['R2_60']:<10.4f}")
        
        # Print Other Models
        for name, m in existing_summary.items():
            print(f"{name:<28} {m['MAE_30']:<12.4f} {m['RMSE_30']:<13.4f} {m['R2_30']:<11.4f} {m['RMSE_60']:<13.4f} {m['R2_60']:<10.4f}")
        print("=" * 80)

    # 7. Save Model and Predictions
    model_save_path = os.path.join(paths["models"], "B_AR_model.joblib")
    joblib.dump({
        "model": ar_champion,
        "best_lag": best_lag,
        "feature_name": "PM2.5",
        "feature_index": pm25_idx,
        "metrics": test_metrics
    }, model_save_path)
    print(f"\n[V] Model saved to: {model_save_path}")
    
    # Save predictions dataframe
    test_dates = test_npz.get('origin_times', np.arange(len(y_test_u)))
    df_preds = pd.DataFrame({
        "actual_30m": y_test_u[:, 0],
        "predicted_30m": pred_test_unscaled[:, 0],
        "actual_60m": y_test_u[:, 1],
        "predicted_60m": pred_test_unscaled[:, 1]
    })
    pred_parquet_path = os.path.join(paths["predictions"], f"B_AR_{best_lag}_predictions.parquet")
    pred_csv_path = os.path.join(paths["predictions"], f"B_AR_{best_lag}_predictions.csv")
    df_preds.to_parquet(pred_parquet_path)
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"[V] Predictions saved to: {pred_parquet_path}")
    
    # Save JSON Report
    report_output = {
        "status": "SUCCESS",
        "model_name": f"AR({best_lag}) Autoregressive Baseline",
        "best_lag_order": best_lag,
        "lookback_duration_hours": float(best_lag * 0.5),
        "test_performance": test_metrics,
        "validation_lag_sweep": val_sweep_results,
        "test_lag_sweep": test_sweep_results,
        "statsmodels_diagnostics": sm_diagnostics,
        "artifacts": {
            "model_path": model_save_path,
            "predictions_parquet": pred_parquet_path,
            "predictions_csv": pred_csv_path
        }
    }
    report_path = os.path.join(paths["reports"], "ar_baseline_results.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_output, f, indent=2)
    print(f"[V] Evaluation Report saved to: {report_path}")
    print("\nExecution of 12_ar_baseline.py completed successfully!\n")


if __name__ == "__main__":
    main()
