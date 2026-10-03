"""Verbatim(ish) reproduction of the original prototype from the Gemini conversation.

Kept for provenance and to reproduce the "R -> 0 overfitting" result. Differences
from the conversation: the three snippets (manual filter, pykalman version,
test bench + constrained MLE) are merged into one file, `plt.show()` is replaced
by saving a PNG, and live data falls back to the bundled CSV sample.
Do NOT use this for new work -- see src/kfnoise for the corrected version.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


class ManualKalmanFilter:
    def __init__(self, initial_price):
        self.x = initial_price
        self.P = 1.0

    def kalman_filter(self, prices, Q, R):
        n = len(prices)
        xs = np.zeros(n)
        P = 1.0
        x = prices[0]
        for t in range(n):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (prices[t] - x)
            P = (1 - K) * P_pred
            xs[t] = x
        return xs

    def fit_mle(self, prices):
        def nll(params):
            Q, R = np.exp(params)
            x = prices[0]; P = 1.0; log_lik = 0
            for z in prices:
                P_pred = P + Q
                F = P_pred + R
                v = z - x
                K = P_pred / F
                x = x + K * v
                P = (1 - K) * P_pred
                log_lik += -0.5 * (np.log(F) + v**2 / F)
            return -log_lik
        res = minimize(nll, [-5.0, -3.0], method="L-BFGS-B")
        return np.exp(res.x[0]), np.exp(res.x[1])

    def fit_constrained_mle(self, prices):
        diffs = np.diff(prices)
        empirical_R = np.var(diffs) * 0.5

        def nll(params):
            Q = np.exp(params[0]); R = empirical_R
            x = prices[0]; P = 1.0; log_lik = 0
            for z in prices:
                P_pred = P + Q
                F = P_pred + R
                v = z - x
                K = P_pred / F
                x = x + K * v
                P = (1 - K) * P_pred
                log_lik += -0.5 * (np.log(F) + v**2 / F)
            return -log_lik
        res = minimize(nll, [np.log(empirical_R)], method="L-BFGS-B")
        return np.exp(res.x[0]), empirical_R


def run_pykalman_filter(prices):
    from pykalman import KalmanFilter
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=prices[0], initial_state_covariance=1,
                      n_dim_state=1, n_dim_obs=1)
    kf = kf.em(prices, n_iter=10,
               em_vars=["transition_covariance", "observation_covariance"])
    state_means, _ = kf.filter(prices)
    return state_means.flatten(), kf.transition_covariance, kf.observation_covariance


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "btcusdt_1m_sample.csv", parse_dates=["timestamp"])
    prices, timestamps = df["close"].values, df["timestamp"]

    mkf = ManualKalmanFilter(prices[0])
    Q_m, R_m = mkf.fit_mle(prices)
    manual_states = mkf.kalman_filter(prices, Q_m, R_m)
    Q_c, R_c = mkf.fit_constrained_mle(prices)
    py_states, Q_py, R_py = run_pykalman_filter(prices)
    Q_py, R_py = float(np.ravel(Q_py)[0]), float(np.ravel(R_py)[0])

    print("--- Parameter Comparison (raw USD prices) ---")
    print(f"Manual MLE       -> Q: {Q_m:.5f}, R: {R_m:.5f}, Q/R: {Q_m / R_m:.5f}")
    print(f"Constrained MLE  -> Q: {Q_c:.5f}, R: {R_c:.5f}, Q/R: {Q_c / R_c:.5f}")
    print(f"PyKalman EM      -> Q: {Q_py:.5f}, R: {R_py:.5f}, Q/R: {Q_py / R_py:.5f}")

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, prices, color="lightgray", label="Raw BTC Price", linewidth=1)
    plt.plot(timestamps, manual_states, color="blue", label="Manual Filter (MLE)",
             linewidth=1, linestyle="--")
    plt.plot(timestamps, py_states, color="red", label="PyKalman (EM)", linewidth=1, alpha=0.7)
    plt.title(f"Signal Extraction: BTC/USDT (Last {len(prices)} minutes)")
    plt.legend(); plt.ylabel("Price ($)")
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, prices - manual_states, color="purple", linewidth=0.5,
             label="Estimated Noise (Bid-Ask Bounce)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Extracted Microstructure Noise"); plt.legend()
    plt.tight_layout()
    out = ROOT / "figures" / "legacy_test_bench.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"saved {out.relative_to(ROOT)}")
