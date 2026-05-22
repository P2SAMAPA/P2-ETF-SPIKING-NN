import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import snntorch as snn
import snntorch.surrogate as surrogate
import math


def _beta_from_tau(tau: float, dt: float = 1.0) -> float:
    """Convert membrane/synaptic time constant (days) to snntorch beta decay factor."""
    return float(math.exp(-dt / max(tau, 1e-6)))


class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps,
                 tau_mem=20.0, tau_syn=5.0, threshold=0.5):
        super().__init__()
        self.num_steps = num_steps

        beta_mem = _beta_from_tau(tau_mem)   # ~0.951 for tau=20
        beta_syn = _beta_from_tau(tau_syn)   # ~0.819 for tau=5
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1  = nn.Linear(input_size, hidden_size)
        # BUG FIX 1: snn.Leaky does NOT accept tau_mem/tau_syn kwargs — it takes
        # beta (scalar membrane decay). Wrong kwargs cause all weights to stay at
        # init and output 0. Use snn.Synaptic which accepts alpha (syn) + beta (mem).
        self.lif1 = snn.Synaptic(alpha=beta_syn, beta=beta_mem,
                                  spike_grad=spike_grad, threshold=threshold,
                                  learn_alpha=False, learn_beta=False,
                                  learn_threshold=False)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Synaptic(alpha=beta_syn, beta=beta_mem,
                                  spike_grad=spike_grad, threshold=threshold,
                                  learn_alpha=False, learn_beta=False,
                                  learn_threshold=False)

        # BUG FIX 2: The final layer must produce a continuous-valued output,
        # not binary spikes. A spike-rate head (mean of spikes) maps everything
        # to [0,1] which rounds to 0.00 when printed as a return. Replace with
        # a linear readout on the final membrane potential instead.
        self.readout = nn.Linear(output_size, 1)

    def forward(self, x):
        # x: (batch, num_steps, input_size)
        # BUG FIX 3: lif1.init_leaky() only returns mem (1-tensor).
        # snn.Synaptic.init_synaptic() returns (syn, mem) — must unpack both.
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        mem2_final = None
        for step in range(self.num_steps):
            x_step           = x[:, step, :]          # (batch, input_size)
            cur1             = self.fc1(x_step)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2             = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            mem2_final       = mem2                    # keep last membrane potential

        # BUG FIX 2 (continued): read out from membrane potential, not spike rate.
        # mem2_final is continuous-valued — passes through linear layer to give
        # unbounded predicted return (can be negative/positive, not just [0,1]).
        out = self.readout(mem2_final)                 # (batch, 1)
        return out


def spike_encode(returns_series: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """
    Encode returns as spike train.
    Uses SIGNED encoding: +1 for positive spike, -1 for negative spike, 0 otherwise.
    This preserves direction information — pure binary |r|>threshold loses the sign
    and makes it impossible for the network to distinguish up vs down moves.
    BUG FIX 4: original used unsigned binary (loses direction → predictions symmetric
    around 0 → net output ≈ 0 for balanced datasets).
    """
    spikes = np.zeros_like(returns_series, dtype=np.float32)
    spikes[returns_series >  threshold] =  1.0
    spikes[returns_series < -threshold] = -1.0
    return spikes


def create_spike_dataset(returns_series, window, seq_len=10, spike_threshold=0.001):
    """
    Create sliding-window spike datasets from a return series.

    Args:
        returns_series: pd.Series of log returns (already sliced to `window` rows
                        by the caller in trainer.py).
        window:         kept for API compatibility but NOT used in the length check
                        (see BUG FIX 5 below).
        seq_len:        number of time steps per input sequence (lookback).
        spike_threshold: threshold for spike encoding.

    Returns:
        X: np.ndarray of shape (n_samples, seq_len, 1)
        y: np.ndarray of shape (n_samples,) — raw returns (regression target)
    """
    series_clean = returns_series.dropna().astype(np.float32)

    # BUG FIX 5: Original guard was `len < window + seq_len + 1`.
    # But ret_series is ALREADY sliced to `window` rows in trainer.py via
    # `returns[etf].iloc[-win:]`, so series_clean has at most `window` rows.
    # Adding `window` to the guard means it ALWAYS fails → samples=0.
    # Correct minimum is just seq_len + 2 (need seq_len input + 1 target).
    if len(series_clean) < seq_len + 2:
        return None, None

    spikes = spike_encode(series_clean.values, spike_threshold)

    X, y = [], []
    for i in range(seq_len, len(spikes) - 1):
        X.append(spikes[i - seq_len:i])
        # Target: actual next-day return (not spike) — keeps regression meaningful
        y.append(series_clean.iloc[i + 1])

    if len(X) == 0:
        return None, None

    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_snn(X_train, y_train, input_size=1, hidden_size=32, output_size=1,
              num_steps=10, tau_mem=20.0, tau_syn=5.0, threshold=0.5,
              lr=1e-3, epochs=50, batch_size=32, device='cpu'):

    model     = SpikingNet(input_size, hidden_size, output_size, num_steps,
                           tau_mem, tau_syn, threshold).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n   = len(X_t)

    model.train()
    for epoch in range(epochs):
        indices    = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx  = indices[i:i + batch_size]
            Xb   = X_t[idx]
            yb   = y_t[idx]
            pred = model(Xb).squeeze(-1)   # (batch,)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            # BUG FIX 6: gradient clipping prevents exploding gradients through
            # the surrogate gradient path — without this, weights can go NaN
            # and outputs collapse to 0 after a few epochs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss: {total_loss/n:.6f}")

    return model


def predict_snn(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(X_t).squeeze(-1).cpu().numpy()
    return pred.flatten()
