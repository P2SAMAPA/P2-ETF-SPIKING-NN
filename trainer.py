import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from spiking_model import create_spike_dataset, train_snn, predict_snn

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use percentile encoding (override config if not present)
    spike_percentile = getattr(config, 'SPIKE_PERCENTILE', 85)
    vol_window = getattr(config, 'VOL_WINDOW', 20)

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Spiking Neural Network) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + config.INPUT_SIZE + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + config.INPUT_SIZE + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_scores = {}
            for etf in tickers:
                if etf not in returns.columns:
                    continue
                ret_series = returns[etf].iloc[-win:]
                X, y = create_spike_dataset(ret_series, win,
                                            seq_len=config.INPUT_SIZE,
                                            percentile=spike_percentile,
                                            vol_window=vol_window)
                if X is None or len(X) < 10:
                    print(f"    {etf}: no data from create_spike_dataset")
                    continue
                split = int(0.8 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                try:
                    model = train_snn(X_train, y_train,
                                      input_size=1,
                                      hidden_size=config.HIDDEN_NEURONS,
                                      output_size=1,
                                      num_steps=config.INPUT_SIZE,
                                      tau_mem=config.TAU_MEM,
                                      tau_syn=config.TAU_SYN,
                                      threshold=config.THRESHOLD,
                                      lr=config.LEARNING_RATE,
                                      epochs=config.EPOCHS,
                                      batch_size=config.BATCH_SIZE,
                                      device=device)
                    last_X = X[-1:].reshape(1, config.INPUT_SIZE, 1)
                    pred = predict_snn(model, last_X)[0]
                    etf_scores[etf] = pred
                except Exception as e:
                    print(f"    {etf}: training failed: {e}")
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "snn_pred": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by SNN predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/spiking_nn_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Spiking Neural Network Engine complete ===")

if __name__ == "__main__":
    main()
