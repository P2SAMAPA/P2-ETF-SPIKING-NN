import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import snntorch as snn
from snntorch import spikegen

class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps, tau_mem=20.0, tau_syn=5.0, threshold=0.5):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(tau_mem=tau_mem, tau_syn=tau_syn, spike_grad=True, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(tau_mem=tau_mem, tau_syn=tau_syn, spike_grad=True, threshold=threshold)

    def forward(self, x):
        # x: (batch, num_steps, input_size)
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_spikes = []
        for step in range(self.num_steps):
            x_step = x[:, step, :]
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spikes.append(spk2)
        out = torch.stack(out_spikes, dim=1).mean(dim=1)
        return out

def spike_encode(returns_series, threshold_mult=1.5, vol_window=20):
    """Convert return series to spike trains based on rolling volatility threshold."""
    vol = returns_series.rolling(vol_window).std()
    threshold = vol * threshold_mult
    spikes = (returns_series > threshold).astype(int)
    return spikes.fillna(0).astype(int).values

def create_spike_dataset(returns_series, window, seq_len=10, threshold_mult=1.5, vol_window=20):
    """
    For a single ETF (pandas Series), create sliding windows of spike trains.
    Returns X (n_samples, seq_len, 1) and y (n_samples,).
    """
    if len(returns_series) < window + seq_len + 1:
        return None, None
    # Use last `window` days of returns
    returns_window = returns_series.iloc[-window:]
    # Compute spikes on the window
    spikes = spike_encode(returns_window, threshold_mult, vol_window)
    if len(spikes) < seq_len + 1:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(spikes)-1):
        X.append(spikes[i-seq_len:i])
        y.append(returns_window.iloc[i+1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if len(X) == 0:
        return None, None
    # Reshape to (batch, seq_len, 1) for SNN input (1 feature per time step)
    X = X.reshape(-1, seq_len, 1)
    return X, y

def train_snn(X_train, y_train, input_size=1, hidden_size=32, output_size=1,
              num_steps=10, tau_mem=20.0, tau_syn=5.0, threshold=0.5,
              lr=1e-3, epochs=50, batch_size=32, device='cpu'):
    model = SpikingNet(input_size, hidden_size, output_size, num_steps, tau_mem, tau_syn, threshold).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n = len(X_t)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            Xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            pred = model(Xb)
            loss = criterion(pred.squeeze(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def predict_snn(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred
