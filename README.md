# Rigorous 12-Step PM2.5 Short-Term Air Quality Forecasting & Early Warning System

> **Station Sidakaya, Cilacap, Indonesia**  
> *Spatio-Temporal PM2.5 Forecasting using Regularized CNN-BiLSTM and 12-Step Data Science Pipeline*

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)
![Status](https://img.shields.io/badge/Status-100%25%20Verified%20%26%20Pass-success.svg)

An end-to-end, scientifically rigorous 12-Step Data Science and Deep Learning Pipeline for **30-minute (+30m)** and **60-minute (+60m)** short-term PM2.5 concentration forecasting at Station Sidakaya, Cilacap, Indonesia.

---

## 📌 Key Milestones & Final Conclusions

1. **Pruning High-Dimensional Redundancy**: Pruning features from 94 (F4) to 16 (F1) increased the sample-to-input ratio from **2.05** to **12.02**, resolving deep learning overfitting.
2. **CNN-BiLSTM Superiority over Persistence**: The champion model **CNN-BiLSTM (F1)** achieves $R^2_{30} = 0.7062$ ($\text{RMSE}_{30} = 0.9549\text{ }\mu\text{g/m}^3$) and $R^2_{60} = 0.5282$ ($\text{RMSE}_{60} = 1.2101\text{ }\mu\text{g/m}^3$), officially outperforming Persistence ($R^2_{30} = 0.6882, R^2_{60} = 0.4393$) and Ridge ($R^2_{30} = 0.6202$).
3. **Statistical Significance Confirmed**: Paired Block Bootstrap Testing (1,000 resamples) proves the +60m RMSE improvement ($8.27\%$) is **statistically significant ($p = 0.0070 < 0.01$, 95% CI $[0.0203, 0.1965]$)**.
4. **Attention Mechanism Inefficiency**: Adding Soft Attention on ultra-short horizons (+30m/+60m) adds parameter noise, reducing average $R^2$ from 0.6172 to 0.5697. Standard BiLSTM hidden pooling is optimal.

---

## 📈 Publication Figures (300 DPI)

### 1. Time-Series Predictions vs Actual Observational Data (7-Day Sample Period)
![Figure 1: Time-Series Forecast](figures/fig1_timeseries_forecast.png)

### 2. Scatter Plot Regression ($y = x$ Ideal Line)
![Figure 2: Scatter Plot](figures/fig2_scatter_actual_vs_pred.png)

### 3. Residual Error Distribution & KDE Density Curve
![Figure 3: Residual Distribution](figures/fig3_residual_distribution.png)

### 4. Model Comparison Bar Chart (MAE, RMSE, R²)
![Figure 4: Model Comparison](figures/fig4_model_comparison_bar.png)

---

## 📊 Final Test Set Benchmark Results (2,140 Locked Sequences)

### Table 1: Comparative Model Performance Metrics ($\mu\text{g/m}^3$)

| Model | MAE +30m | RMSE +30m | R² +30m | MAE +60m | RMSE +60m | R² +60m | Avg R² |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence Baseline (B0)** | 0.5107 | 0.9836 | 0.6882 | 0.6881 | 1.3192 | 0.4393 | 0.5638 |
| **Ridge Regression (B1)** | 0.6929 | 1.0856 | 0.6202 | 0.8652 | 1.3117 | 0.4456 | 0.5329 |
| **Random Forest (B2)** | 0.9447 | 3.2848 | -2.4772 | 1.1336 | 2.5963 | -1.1718 | -1.8245 |
| **CNN-BiLSTM (Full 94 Fitur)** | 1.7487 | 2.4198 | -0.8870 | 1.7614 | 2.4300 | -0.9026 | -0.8948 |
| **CNN-BiLSTM-Attention (F1)** | 0.7427 | 1.0377 | 0.6530 | 0.8854 | 1.2626 | 0.4864 | 0.5697 |
| **CNN-BiLSTM (Champion F1)** | **0.5947** | **0.9549** | **0.7062** | **0.7741** | **1.2101** | **0.5282** | **0.6172** |

---

## 🛠️ Repository Structure

```
airpollutan/
├── WALKTHROUGH.md                     # Comprehensive 12-step technical research paper walkthrough
├── pipeline_scripts/
│   ├── perform_step10_fast.py          # Step 10 validation search & capacity pruning script
│   └── run_step11_instant.py           # Step 11 Diebold-Mariano & Block Bootstrap evaluator
├── figures/                           # 300 DPI publication figures
│   ├── fig1_timeseries_forecast.png
│   ├── fig2_scatter_actual_vs_pred.png
│   ├── fig3_residual_distribution.png
│   └── fig4_model_comparison_bar.png
├── reports/                           # Comprehensive JSON audit & statistical reports
│   ├── audit_7a_to_7e.json
│   ├── step8_benchmark_results.json
│   ├── step9_diagnostics_report.json
│   ├── step10_optimization_report.json
│   └── step11_final_statistical_report.json
├── models/                            # Trained PyTorch model checkpoints (.pt) and scalers (.joblib)
│   ├── Optimal_CNN_BiLSTM.pt
│   ├── Optimal_CNN_BiLSTM_Attention.pt
│   ├── x_scaler.joblib
│   └── y_scaler.joblib
├── app.py                             # Flask Web Early Warning System Application
└── README.md                          # Project Documentation & Summary
```
