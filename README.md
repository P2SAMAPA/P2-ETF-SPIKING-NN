# Spiking Neural Network Engine

Implements a Leaky Integrate‑and‑Fire (LIF) spiking neural network for ETF return prediction. Returns are converted to spike trains via threshold encoding (returns exceeding rolling volatility). The SNN is trained using surrogate gradient descent to predict the next day's return. This neuromorphic approach captures timing information in market spikes.

- **Spike encoding:** return > threshold × rolling volatility
- **SNN architecture:** LIF layers (2‑layer fully connected)
- **Training:** surrogate gradient descent (MSE loss)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
