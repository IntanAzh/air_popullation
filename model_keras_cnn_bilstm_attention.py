import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers, ops

@keras.saving.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    """
    Custom Keras Layer for Temporal Attention Mechanism.
    Assigns higher mathematical weights to extreme weather shifts and sudden pollution spikes.
    """
    def __init__(self, units=128, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = layers.Dense(self.units, activation='tanh', name="attn_transform")
        self.V = layers.Dense(1, name="attn_score")
        super(TemporalAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        score = self.W(inputs)                            # (batch_size, seq_len, units)
        attention_weights = ops.softmax(self.V(score), axis=1) # (batch_size, seq_len, 1)
        
        # Context vector as weighted sum
        context_vector = ops.sum(attention_weights * inputs, axis=1)  # (batch_size, features)
        return context_vector

    def get_config(self):
        config = super(TemporalAttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

def build_keras_cnn_bilstm_attention(seq_len=48, in_features=59, horizon=2, n_targets=2):
    """
    Rakit Algoritma Hibrida Lapis demi Lapis (Keras Functional API):
    1. Input Layer: (seq_len, in_features)
    2. Conv1D Layer: (filters=64, kernel_size=3) -> Ekstraksi fitur spasial/lokal meteorologi
    3. MaxPooling1D Layer: (pool_size=2) -> Reduksi dimensi & ekstraksi sinyal puncak
    4. Bidirectional LSTM Layer: (units=128, return_sequences=True) -> Membaca memori dua arah (maju & mundur)
    5. Custom Attention Layer: Memboboti lonjakan ekstrem cuaca/polusi
    6. Dropout Layer: (rate=0.3) -> Mencegah overfitting
    7. Dense Layer: Output linear regression (horizon * n_targets)
    """
    inputs = layers.Input(shape=(seq_len, in_features), name="meteorological_pollutant_input")

    # Layer 1: Conv1D (Ekstraksi korelasi lokal meteorologi)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv1d_spatial")(inputs)
    x = layers.BatchNormalization(name="batch_norm")(x)

    # Layer 2: MaxPooling1D (Reduksi dimensi temporal & retensi puncak)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool1d_reduction")(x)

    # Layer 3: Bidirectional LSTM (Dua arah waktu maju & mundur)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True), name="bilstm_bidirectional")(x)

    # Layer 4: Custom Temporal Attention Mechanism
    context = TemporalAttentionLayer(units=128, name="temporal_attention")(x)

    # Layer 5: Dropout Layer (Mencegah Overfitting)
    x = layers.Dropout(rate=0.3, name="dropout_regularization")(context)

    # Layer 6: Dense Layer (Linear Regression Output)
    x = layers.Dense(units=128, activation="relu", name="dense_hidden")(x)
    outputs_flat = layers.Dense(units=horizon * n_targets, activation="linear", name="dense_linear_regression")(x)

    # Reshape ke (batch_size, horizon, n_targets)
    outputs = layers.Reshape(target_shape=(horizon, n_targets), name="forecast_output")(outputs_flat)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Attention_Keras")
    return model

if __name__ == "__main__":
    model = build_keras_cnn_bilstm_attention()
    model.summary()
