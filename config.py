import os

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-spiking-nn-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

THRESHOLD_MULT = 1.0   # multiplier for rolling volatility
VOL_WINDOW = 252       # window length for volatility calculation

# ── Spike encoding ──────────────────────────────────────────────────────────
# BUG FIX (config): spike_threshold was hardcoded 0.002 in trainer.py.
# Moved here so it is tunable. 0.002 (0.2%) fires on ~60% of daily ETF returns
# which is reasonable. If you want sparser spikes raise to 0.005.
SPIKE_THRESHOLD = 0.002

# ── Spiking network architecture ────────────────────────────────────────────
INPUT_SIZE     = 20    # FIX: was 10 — with seq_len=10 and batch split 80/20
                       # you only get ~(win*0.8 - 10) training samples.
                       # 20 gives a richer temporal context without losing too
                       # many samples on the shortest 63d window.
HIDDEN_NEURONS = 64    # FIX: was 32 — doubled capacity; SNN with 1-feature
                       # input needs more width to learn meaningful patterns.
TAU_MEM        = 20.0  # membrane time constant (days)
TAU_SYN        = 5.0   # synaptic time constant (days)
THRESHOLD      = 0.5   # LIF firing threshold

# ── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE  = 5e-4  # FIX: was 1e-3 — slightly lower LR stabilises surrogate
                       # gradient training; reduces risk of loss diverging to 0.
EPOCHS         = 100   # FIX: was 50 — SNN with surrogate gradients needs more
                       # epochs to escape the near-zero initialisation region.
BATCH_SIZE     = 32
TOP_N          = 3
