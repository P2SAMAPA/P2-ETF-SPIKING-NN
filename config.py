import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
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

# Spike encoding parameters
THRESHOLD_MULT = 1.5      # multiple of moving average volatility to trigger spike
VOL_WINDOW = 20           # window for rolling volatility

# Spiking neural network parameters
INPUT_SIZE = 10           # number of past days used as input spike train length
HIDDEN_NEURONS = 32
OUTPUT_NEURONS = 1
TIME_STEPS = 20           # simulation time steps per input sample
TAU_MEM = 20.0            # membrane time constant (ms)
TAU_SYN = 5.0             # synaptic time constant
THRESHOLD = 0.5           # spike threshold
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

TOP_N = 3
