import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import snntorch as snn

class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps, tau_mem=20.0, tau_syn=5.0, threshold=0.5):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(tau_mem=tau_mem, tau_syn=tau_syn, spike_grad=True, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(tau_mem=tau_mem, tau_syn=tau_syn, spike_grad=True, threshold=threshold)

    def forward(self, x):
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

def spike_encode_percentile(returns_series, percentile=90, vol_window=20):
    """Spike when return exceeds rolling volatility threshold at given percentile."""
    # Use rolling standard deviation as volatility proxy
    vol = returns_series.rolling(vol_window).std()
    # Threshold at percentile of historical absolute returns (or use vol * mult)
    # Simpler: use fixed percentile of the series itself
    thresh = returns_series.abs().rolling(vol_window).quantile(percentile/100.0)
    thresh = thresh.fillna(returns_series.abs().quantile(percentile/100.0))
    spikes = (returns_series > thresh).astype(int)
    return spikes.fillna(0).astype(int).values

def create_spike_dataset(returns_series, window, seq_len=10, percentile=90, vol_window=20):
    if len(returns_series) < window + seq_len + 5:
        return None, None
    # Take the last `window` days
    returns_win = returns_series.iloc[-window:]
    # Compute spikes on the whole window
    spikes = spike_encode_percentile(returns_win, percentile, vol_window)
    if len(spikes) < seq_len + 2:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(spikes)-1):
        X.append(spikes[i-seq_len:i])
        y.append(returns_win.iloc[i+1])
    if len(X) == 0:
        return None, None
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
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
