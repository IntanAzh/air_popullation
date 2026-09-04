import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers, ops
import numpy as np

# 1. Custom Keras Layer: Temporal Attention Mechanism
@keras.saving.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    """
    Lapisan Attention Mechanism Khusus:
    Memberikan bobot komputasi tertinggi pada momen transisi cuaca krusial
    dan lonjakan polusi ekstrem (spikes).
    """
    def __init__(self, units=128, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, activation='tanh', name="attn_transform")
        self.V = layers.Dense(1, name="attn_score")

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        score = self.W(inputs)                            # (batch_size, seq_len, units)
        attention_weights = ops.softmax(self.V(score), axis=1) # (batch_size, seq_len, 1)
        
        # Vektor konteks sebagai jumlah terbobot (weighted sum)
        context_vector = ops.sum(attention_weights * inputs, axis=1)  # (batch_size, features)
        return context_vector, ops.squeeze(attention_weights, axis=-1)

    def get_config(self):
        config = super(TemporalAttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

# 2. Konstruksi Arsitektur Algoritma Hibrida (Keras Functional API)
def create_hybrid_cnn_bilstm_attention_model(seq_len=48, in_features=59, horizon=2, n_targets=2):
    """
    Struktur Deep Learning Hibrida:
    1. Input Layer: (batch, seq_len=48, in_features=59)
    2. 1D-CNN (Conv1D): Menyaring representasi fitur lokal & spasial pergerakan cuaca
    3. BiLSTM (Bidirectional LSTM): Menganalisis konteks temporal dua arah (maju & mundur)
    4. Custom Attention Mechanism: Pembobotan transisi cuaca krusial & lonjakan ekstrem (spikes)
    5. Dropout Layer: Regularisasi
    6. Dense Linear Head: Regresi output peramalan
    """
    inputs = layers.Input(shape=(seq_len, in_features), name="input_meteorology_pollutant")

    # Step 1: Lapisan 1D-CNN (Penyaring Fitur Spasial/Lokal Cuaca)
    x_conv = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="1D_CNN_spatial_filter")(inputs)
    x_conv = layers.BatchNormalization(name="batch_norm")(x_conv)
    x_pool = layers.MaxPooling1D(pool_size=2, name="maxpool_downsampling")(x_conv)

    # Step 2: Lapisan BiLSTM (Konteks Temporal Dua Arah Maju & Mundur)
    x_bilstm = layers.Bidirectional(
        layers.LSTM(units=128, return_sequences=True), 
        name="BiLSTM_bidirectional_temporal"
    )(x_pool)

    # Step 3: Lapisan Attention Mechanism Khusus (Momen Transisi Krusial & Spikes)
    context_vector, attn_weights = TemporalAttentionLayer(units=128, name="custom_attention_mechanism")(x_bilstm)

    # Step 4: Dropout & Regresi Linear Dense Layer
    x_drop = layers.Dropout(rate=0.3, name="dropout_regularization")(context_vector)
    x_dense = layers.Dense(units=128, activation="relu", name="dense_representation")(x_drop)
    outputs_flat = layers.Dense(units=horizon * n_targets, activation="linear", name="dense_linear_regression")(x_dense)
    outputs = layers.Reshape(target_shape=(horizon, n_targets), name="forecast_output")(outputs_flat)

    # Model Keras Komplit
    model = keras.Model(inputs=inputs, outputs=[outputs, attn_weights], name="CNN_BiLSTM_Attention_Hybrid")
    return model

def verify_architecture():
    print("=" * 80)
    print("DEMONSTRASI VERIFIKASI ARSITEKTUR MODEL HIBRIDA (KERAS / TENSORFLOW)")
    print("=" * 80)
    
    model = create_hybrid_cnn_bilstm_attention_model()
    model.summary()

    # Test Forward Pass dengan Dummy Tensor Batch
    dummy_input = np.random.randn(32, 48, 59).astype(np.float32)
    preds, weights = model(dummy_input)

    print("\n" + "=" * 80)
    print("HASIL PENGUJIAN FORWARD INFERENCE MODEL:")
    print("=" * 80)
    print(f"1. Input Batch Shape      : {dummy_input.shape} -> (Batch=32, Lookback=48, Fitur=59)")
    print(f"2. Forecast Output Shape   : {preds.shape}        -> (Batch=32, Horizon=2, Target=2)")
    print(f"3. Attention Weights Map   : {weights.shape}      -> (Batch=32, Pooled_Time_Steps=24)")
    print("=" * 80)

if __name__ == "__main__":
    verify_architecture()
