import os
import sys
import json
import numpy as np

# Ensure VS Code workspace directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from export_dataset import export_all
from train_keras_model import train_and_validate_keras
from train_and_evaluate import train_and_eval
from scaler_utils import load_scaler_and_meta

TENSOR_DIR = r"c:\Users\INTAN\airpollutan\data\processed\tensors"
MODEL_DIR = r"c:\Users\INTAN\airpollutan\models"

def run_master_pipeline_vscode():
    print("=" * 85)
    print("PEMBANGUNAN MODEL DEEP LEARNING (CNN-BiLSTM WITH ATTENTION) DI VS CODE")
    print("SERTA ISOLASI PEMISAHAN DATA TRAIN / TEST SPLIT")
    print("=" * 85)

    # Step 1: Pemisahan Data & Isolasi Test Set
    print("\n[STEP 1] Pemisahan Dataset Sekuensial (80% Train Data / 20% Test Data Isolated)...")
    export_all()

    # Load isolated tensors to verify isolation
    X_tr = np.load(os.path.join(TENSOR_DIR, "X_train.npy"))
    y_tr = np.load(os.path.join(TENSOR_DIR, "y_train.npy"))
    X_te = np.load(os.path.join(TENSOR_DIR, "X_test.npy"))
    y_te = np.load(os.path.join(TENSOR_DIR, "y_test.npy"))

    print("\n--- STATUS ISOLASI PARTISI DATASET ---")
    print(f"  - Train Data (Terisolasi Khusus Melatih Model): X_train {X_tr.shape} | y_train {y_tr.shape}")
    print(f"  - Test Data  (Terisolasi Khusus Evaluasi Model): X_test  {X_te.shape} | y_test  {y_te.shape}")

    # Step 2: Perakitan Arsitektur & Pelatihan Model di VS Code
    print("\n[STEP 2] Pelatihan Jaringan Saraf Tiruan CNN-BiLSTM-Attention (Keras Engine)...")
    train_and_validate_keras()

    print("\n" + "=" * 85)
    print("STATUS MASTER PIPELINE: PEMBANGUNAN & PELATIHAN MODEL DI VS CODE SUKSES 100%!")
    print("=" * 85)

if __name__ == "__main__":
    run_master_pipeline_vscode()
