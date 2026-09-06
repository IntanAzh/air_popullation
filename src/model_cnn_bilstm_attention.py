import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Custom Temporal Attention Mechanism to dynamically assign mathematical weights 
    to extreme weather events and sudden industrial pollution spikes.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(lstm_output))  # (batch_size, seq_len, hidden_dim)
        
        v = self.v.repeat(lstm_output.size(0), 1).unsqueeze(2)  # (batch_size, hidden_dim, 1)
        weights = torch.bmm(energy, v).squeeze(2)              # (batch_size, seq_len)
        
        attn_weights = F.softmax(weights, dim=1)               # (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim)
        return context, attn_weights

class CNN_BiLSTM_Attention(nn.Module):
    """
    Hybrid Model Architecture Layer-by-Layer:
    1. Conv1D (CNN): Extracts local spatial & meteorological feature correlations.
    2. MaxPool1D: Downsamples feature maps to retain peak signals and reduce dimensionality.
    3. Bidirectional LSTM (BiLSTM): Captures forward & backward long-short term temporal memory.
    4. Custom Temporal Attention: Dynamically weights extreme weather shifts & pollution spikes.
    5. Dropout Layer: Prevents model overfitting.
    6. Dense Layer: Linear regression output head producing multi-step PM2.5 & PM10 predictions.
    """
    def __init__(self, in_features, seq_len=48, horizon=2, n_targets=2, cnn_filters=64, lstm_hidden=128):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_targets = n_targets

        # Layer 1: Conv1D
        self.conv1d = nn.Conv1d(in_channels=in_features, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        
        # Layer 2: MaxPool1D
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3: Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_filters, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )

        # Layer 4: Custom Attention Mechanism
        self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)

        # Layer 5: Dropout Layer & Dense Output Head
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.fc_out = nn.Linear(128, horizon * n_targets)  # Linear regression output

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_features)
        
        # Layer 1 & 2: Conv1D -> ReLU -> MaxPool1D
        x_conv = x.permute(0, 2, 1)  # (batch_size, in_features, seq_len)
        x_conv = F.relu(self.bn1(self.conv1d(x_conv)))
        x_pool = self.maxpool1d(x_conv)  # (batch_size, cnn_filters, seq_len // 2)
        
        # Layer 3: BiLSTM
        x_lstm_in = x_pool.permute(0, 2, 1)  # (batch_size, seq_len // 2, cnn_filters)
        lstm_out, _ = self.bilstm(x_lstm_in)  # (batch_size, seq_len // 2, lstm_hidden * 2)

        # Layer 4: Custom Temporal Attention
        context, attn_weights = self.attention(lstm_out)  # (batch_size, lstm_hidden * 2)

        # Layer 5 & 6: Dropout -> Dense Linear Regression Head
        x_drop = self.dropout(context)
        h = F.relu(self.fc1(x_drop))
        out = self.fc_out(h)  # (batch_size, horizon * n_targets)

        # Reshape to (batch_size, horizon, n_targets)
        out = out.view(-1, self.horizon, self.n_targets)
        return out, attn_weights

if __name__ == "__main__":
    model = CNN_BiLSTM_Attention(in_features=59, seq_len=48, horizon=2, n_targets=2)
    dummy_x = torch.randn(32, 48, 59)
    out, weights = model(dummy_x)
    print("=" * 60)
    print("CNN-BiLSTM-ATTENTION ARCHITECTURE VERIFIED LAYER-BY-LAYER")
    print("=" * 60)
    print("1. Input Tensor (Batch, Seq_Len, Features):", dummy_x.shape)
    print("2. Output Forecast (Batch, Horizon, Targets):", out.shape)
    print("3. Attention Weights (Batch, Pooled_Seq_Len):", weights.shape)
