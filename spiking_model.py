import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import snntorch as snn
import math


def _beta_from_tau(tau: float, dt: float = 1.0) -> float:
    """Convert membrane time constant (days) to snntorch beta decay factor."""
    return math.exp(-dt / tau)


class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps,
                 tau_mem=20.0, tau_syn=5.0, threshold=0.5):
        super().__init__()
        self.num_steps = num_steps

        beta_mem = _beta_from_tau(tau_mem)   # ~0.951 for tau=20
        beta_syn = _beta_from_tau(tau_syn)   # ~0.819 for tau=5

        self.fc1  = nn.Linear(input_size, hidden_size)
        # FIX Bug 3: snn.Synaptic takes alpha (syn decay) + beta (mem decay)
        self.lif1 = snn.Synaptic(alpha=beta_syn, beta=beta_mem,
                                  spike_grad=snn.surrogate.fast_sigmoid(),
                                  threshold=threshold)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Synaptic(alpha=beta_syn, beta=beta_mem,
                                  spike_grad=snn.surrogate.fast_sigmoid(),
                                  threshold=threshold)

    def forward(self, x):
        # x: (batch, num_steps, input_size)
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        out_spikes = []
        for step in range(self.num_steps):
            x_step        = x[:, step, :]          # (batch, input_size)
            cur1          = self.fc1(x_step)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2          = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            out_spikes.append(spk2)

        # Average spike rate over time → scalar prediction per sample
        out = torch.stack(out_spikes, dim=1).mean(dim=1)   # (batch, output_size)
        return out


def spike_encode(returns_series, threshold=0.001):
    """Binary spike: 1 if |return| > threshold, else 0."""
    return (np.abs(returns_series) > threshold).astype(np.float32)


def create_spike_dataset(returns_series, window, seq_len=10, spike_threshold=0.001):
    """
    Create sliding-window spike datasets.
    BUG 1 FIX: ret_series is already sliced to `window` rows before calling
    this function, so the length check must NOT add `window` again.
    """
    series_clean = returns_series.dropna()

    # FIX Bug 1: was `window + seq_len + 1` — always failed because
    # series_clean already has at most `window` rows after iloc[-win:] slice.
    if len(series_clean) < seq_len + 2:
        return None, None

    spikes = spike_encode(series_clean.values, spike_threshold)

    X, y = [], []
    for i in range(seq_len, len(spikes) - 1):
        X.append(spikes[i - seq_len:i])
        y.append(series_clean.iloc[i + 1])

    if len(X) == 0:
        return None, None

    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_snn(X_train, y_train, input_size=1, hidden_size=32, output_size=1,
              num_steps=10, tau_mem=20.0, tau_syn=5.0, threshold=0.5,
              lr=1e-3, epochs=50, batch_size=32, device='cpu'):
    model = SpikingNet(input_size, hidden_size, output_size, num_steps,
                       tau_mem, tau_syn, threshold).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n   = len(X_t)

    for epoch in range(epochs):
        indices    = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx  = indices[i:i + batch_size]
            Xb   = X_t[idx]
            yb   = y_t[idx]
            pred = model(Xb)
            loss = criterion(pred.squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/n:.6f}")

    return model


def predict_snn(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred.flatten()
