import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        # x: (batch, time_steps, features) – spike trains over time
        batch_size = x.size(0)
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        # Accumulate outputs over time
        out_spikes = []
        for step in range(self.num_steps):
            x_step = x[:, step, :]
            # First layer
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            # Second layer
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spikes.append(spk2)
        # Average over time steps
        out = torch.stack(out_spikes, dim=1).mean(dim=1)
        return out

def spike_encode(returns, threshold_mult=1.5, vol_window=20):
    """
    Convert return series to spike trains.
    Spikes occur when return > threshold (rolling volatility * multiplier).
    Returns binary spike train (same length as returns).
    """
    # Compute rolling volatility
    vol = returns.rolling(vol_window).std()
    threshold = vol * threshold_mult
    spikes = (returns > threshold).astype(int)
    return spikes

def create_spike_dataset(returns_df, window, seq_len=10, threshold_mult=1.5, vol_window=20):
    """
    For a single ETF, create sliding windows of returns, encode as spikes,
    and prepare training data (X: spikes over time, y: next day return).
    Returns X (n_samples, seq_len, seq_len?) Actually each spike sequence is of length `seq_len`.
    """
    ret_series = returns_df.values.flatten()
    if len(ret_series) < window + seq_len + 1:
        return None, None
    # Use last `window` days of data
    ret_slice = ret_series[-window:]
    # Compute spikes for the whole slice
    # Need a DataFrame to compute rolling?
    # We'll compute using pandas on the original series
    # For simplicity, we'll precompute spikes on the entire series
    # Actually we need spikes for the training window only. We'll compute spikes on the window.
    # Create a dummy returns series
    import pandas as pd
    ret_series_pd = pd.Series(ret_slice)
    vol = ret_series_pd.rolling(vol_window).std()
    threshold = vol * threshold_mult
    spikes = (ret_series_pd > threshold).astype(int).values
    # Now create sliding windows of length seq_len
    X, y = [], []
    for i in range(seq_len, len(spikes)-1):
        X.append(spikes[i-seq_len:i])
        y.append(ret_slice[i+1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # Reshape X for SNN: (batch, time_steps, input_size)
    # Here input_size = seq_len? Actually each time step has a single feature (spike value).
    # We'll treat it as (batch, time_steps, 1) for CNN? No, the SNN expects input over time.
    # For LIF, each time step receives a vector of features. We'll set input_size = seq_len.
    # But we have seq_len features per time step? Actually we have a sequence of spikes over time.
    # We'll treat each time step as a single value (spike), so input_size = 1.
    # The network's forward expects (batch, time_steps, features). So we reshape:
    X = X.reshape(-1, seq_len, 1)
    return X, y

def train_snn(X_train, y_train, input_size=1, hidden_size=32, output_size=1,
              num_steps=20, tau_mem=20.0, tau_syn=5.0, threshold=0.5,
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
            # For SNN, we need to simulate over time; our forward already does that.
            # But we need to set num_steps (the number of time steps within the SNN).
            # The input Xb already has shape (batch, time_steps, features). This time_steps is the number of spike inputs.
            # We'll pass it directly.
            # However, the SpikingNet expects a fixed num_steps, but we can set it dynamically.
            # We'll pass Xb and forward will simulate over time.
            # But the forward method we wrote expects x of shape (batch, num_steps, features).
            # That matches.
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
