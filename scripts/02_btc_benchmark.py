"""Phase 2-4 -- BTC/USDT 1-minute closes: manual MLE vs. constrained MLE vs. pykalman EM.

Data: live via ccxt / Binance public mirror, or the bundled 1000-minute sample
(data/btcusdt_1m_sample.csv, 2026-10-02 09:15 -> 2026-10-03 01:54 UTC).

The model is fitted on log-price in basis points (z = 1e4 * ln P), so Q and R
are in bps^2 per minute.

Usage:
    python scripts/02_btc_benchmark.py              # live, fall back to sample
    python scripts/02_btc_benchmark.py --source csv # reproduce README numbers
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfnoise import (fit_constrained_mle, fit_mle, kalman_filter,  # noqa: E402
                     neg_log_likelihood, profile_likelihood, rts_smoother)
from kfnoise import metrics as M  # noqa: E402
from kfnoise.data import load_prices, to_log_bps  # noqa: E402
from kfnoise.plotting import C, apply_style, plt  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"


def em_convergence(z, n_iter=300, step=10):
    """Track pykalman's EM log-likelihood vs. iteration (it converges slowly)."""
    from pykalman import KalmanFilter
    g0 = float(np.var(np.diff(z)))
    Z = z.reshape(-1, 1)
    kf = KalmanFilter(transition_matrices=[[1.0]], observation_matrices=[[1.0]],
                      transition_covariance=[[0.5 * g0]],
                      observation_covariance=[[0.25 * g0]],
                      initial_state_mean=[z[0]], initial_state_covariance=[[g0]])
    hist = []
    for it in range(0, n_iter + 1, step):
        if it > 0:
            kf = kf.em(Z, n_iter=step,
                       em_vars=["transition_covariance", "observation_covariance"])
        Q = float(np.ravel(kf.transition_covariance)[0])
        R = float(np.ravel(kf.observation_covariance)[0])
        hist.append((it, Q, R, neg_log_likelihood(z, Q, R)))
    return np.array(hist)


def main(source: str, em_iter: int):
    apply_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    df, src = load_prices(source)
    prices = df["close"].to_numpy(float)
    ts = df["timestamp"]
    z = to_log_bps(prices)
    print(f"Data: {len(z)} x 1-min closes from {src} ({ts.iloc[0]} -> {ts.iloc[-1]})")

    # ---- fits -------------------------------------------------------------
    fits = {
        "mle": fit_mle(z),
        "roll": fit_constrained_mle(z, method="roll"),
        "halfdiff": fit_constrained_mle(z, method="half_diff_var"),
    }
    t0 = time.time()
    hist = em_convergence(z, n_iter=em_iter)
    em_secs = time.time() - t0
    it, Qe, Re, nlle = hist[-1]
    from kfnoise.filter import FitResult
    fits["em"] = FitResult(Qe, Re, nlle, f"pykalman EM ({int(it)} iter)")
    hist10 = hist[hist[:, 0] == 10][0]
    fits["em10"] = FitResult(hist10[1], hist10[2], hist10[3], "pykalman EM (10 iter)")

    labels = {
        "mle": "Manual MLE (Q, R free)",
        "roll": "Constrained MLE, R = Roll estimator",
        "halfdiff": "Constrained MLE, R = Var(Δz)/2 (prototype)",
        "em10": "pykalman EM, 10 iter (prototype)",
        "em": f"pykalman EM, {int(it)} iter",
    }

    rows, filt = {}, {}
    for k, f in fits.items():
        r = kalman_filter(z, f.Q, f.R)
        filt[k] = r
        s = M.summarize(z, r.x_filt, r.innov)
        rows[k] = {"label": labels[k], "Q_bps2": f.Q, "R_bps2": f.R, "Q_over_R": f.snr,
                   "K_ss": f.steady_state_gain, "neg_loglik": f.nll, **s}

    # Lag vs. a 10-min SMA (the "zero-lag" claim)
    sma10 = M.sma(z, 10)
    lag_sma = M.lag_vs_reference(sma10, z)
    lag_kf = M.lag_vs_reference(filt["mle"].x_filt, z)

    # ---- print table --------------------------------------------------------
    print(f"\n{'method':48s} {'Q':>7s} {'R':>7s} {'Q/R':>7s} {'K_ss':>5s} "
          f"{'-logL':>8s} {'VarRed%':>7s} {'ACF1':>6s} {'LB p':>6s}")
    for k in ["mle", "roll", "halfdiff", "em10", "em"]:
        r = rows[k]
        print(f"{r['label']:48s} {r['Q_bps2']:7.2f} {r['R_bps2']:7.2f} {r['Q_over_R']:7.2f} "
              f"{r['K_ss']:5.3f} {r['neg_loglik']:8.2f} {r['variance_reduction_pct']:7.1f} "
              f"{r['acf1_filtered']:6.3f} {r['ljung_box_p_innov']:6.3f}")
    print(f"{'Raw 1-min log returns':48s} {'':7s} {'':7s} {'':7s} {'':5s} {'':8s} "
          f"{0:7.1f} {rows['mle']['acf1_raw']:6.3f}")
    noise_share = 2 * rows["mle"]["R_bps2"] / np.var(np.diff(z))
    print(f"\nNoise share of 1-min return variance (2R / Var(Δz)): {100 * noise_share:.1f}%")
    print(f"Lag (minutes) vs. raw price: Kalman filter = {lag_kf}, SMA(10) = {lag_sma}")
    print(f"EM convergence run: {em_secs:.1f}s for {int(it)} iterations")

    out = {"source": src, "n": int(len(z)), "start": str(ts.iloc[0]), "end": str(ts.iloc[-1]),
           "units": "Q, R in bps^2 of log-price per minute", "fits": rows,
           "noise_share_pct": float(100 * noise_share),
           "lag_minutes": {"kalman_mle": lag_kf, "sma10": lag_sma},
           "em_history": hist.tolist()}
    (RES / "btc_results.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures -------------------------------------------------------------
    usd = lambda x: np.exp(np.asarray(x) / 1e4)  # noqa: E731
    xs, _ = rts_smoother(filt["mle"], fits["mle"].Q)

    # (1) price + filters, full window and last 60 minutes
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1]})
    for ax, sl, ttl in [(axes[0], slice(None), f"BTC/USDT 1-min close — {len(z)} minutes"),
                        (axes[1], slice(-60, None), "Zoom: last 60 minutes")]:
        ax.plot(ts[sl], prices[sl], color=C["raw"], lw=1.2, marker="o" if sl.start else None,
                ms=3, label="Raw close")
        ax.plot(ts[sl], usd(filt["mle"].x_filt)[sl], color=C["s1"], lw=1.6,
                label=f"Kalman, MLE (K={rows['mle']['K_ss']:.2f})")
        ax.plot(ts[sl], usd(filt["em"].x_filt)[sl], color=C["s2"], lw=1.4, ls="--",
                label=f"Kalman, pykalman EM (K={rows['em']['K_ss']:.2f})")
        ax.plot(ts[sl], usd(filt["halfdiff"].x_filt)[sl], color=C["s4"], lw=1.6,
                label=f"Kalman, R=Var(Δz)/2 (K={rows['halfdiff']['K_ss']:.2f})")
        ax.set_title(ttl)
        ax.set_ylabel("USD")
    axes[0].legend(ncol=2, loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "btc_filters.png")
    plt.close(fig)

    # (2) extracted noise (observation - filtered state)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    noise_usd = prices - usd(xs)
    ax.plot(ts, noise_usd, color=C["s1"], lw=0.8)
    ax.axhline(0, color=C["truth"], lw=0.6)
    ax.set_title("Extracted microstructure noise: raw − RTS-smoothed efficient price")
    ax.set_ylabel("USD")
    fig.autofmt_xdate()
    fig.savefig(FIG / "btc_noise.png")
    plt.close(fig)

    # (3) ACF of returns
    nl = 15
    lags = np.arange(1, nl + 1)
    band = 1.96 / np.sqrt(len(z))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.27
    ax.bar(lags - w, M.acf(np.diff(z), nl)[1:], w, color=C["raw"], label="Raw returns")
    ax.bar(lags, M.acf(np.diff(filt["mle"].x_filt), nl)[1:], w, color=C["s1"],
           label="Filtered returns (MLE)")
    ax.bar(lags + w, M.acf(np.diff(filt["halfdiff"].x_filt), nl)[1:], w, color=C["s4"],
           label="Filtered returns (R = Var(Δz)/2)")
    ax.axhspan(-band, band, color="#e6e5e0", zorder=0)
    ax.axhline(0, color=C["truth"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(lags)
    ax.set_xlabel("lag (minutes)")
    ax.set_title("Autocorrelation of 1-min returns (shaded: 95% white-noise band)")
    ax.legend(ncol=3, loc="upper right")
    fig.savefig(FIG / "btc_acf.png")
    plt.close(fig)

    # (4) profile likelihood + EM convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    grid = np.linspace(-1.5, 3.0, 61)
    prof = profile_likelihood(z, grid)
    axes[0].plot(grid, prof - prof.min(), color=C["s1"])
    for k, col in [("mle", C["s1"]), ("em", C["s2"]), ("halfdiff", C["s4"])]:
        lq = np.log10(rows[k]["Q_over_R"])
        axes[0].axvline(lq, color=col, ls="--", lw=1.1)
        short = {"mle": "MLE", "em": f"EM ({int(it)} it)", "halfdiff": "R=Var(Δz)/2"}[k]
        axes[0].annotate(short,
                         (lq, 36 if k != "em" else 30),
                         xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=TEXT)
    axes[0].set_xlabel("log10(Q/R)")
    axes[0].set_ylabel("Δ(−log L)")
    axes[0].set_ylim(0, 40)
    axes[0].set_title("Profile likelihood over signal-to-noise ratio")
    gap = np.clip(hist[:, 3] - rows["mle"]["neg_loglik"], 1e-3, None)
    axes[1].plot(hist[:, 0], gap, color=C["s2"], marker="o", ms=3)
    axes[1].axvline(10, color=C["truth"], ls=":", lw=1)
    axes[1].annotate("prototype stopped here (10 iter)", (10, gap.max() / 3),
                     xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=TEXT)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("EM iteration")
    axes[1].set_ylabel("−log L − (−log L)_MLE")
    axes[1].set_title("pykalman EM: likelihood gap to the direct MLE")
    fig.tight_layout()
    fig.savefig(FIG / "btc_likelihood.png")
    plt.close(fig)
    print("saved figures/ and results/btc_results.json")


TEXT = "#52514e"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["auto", "live", "csv"], default="auto")
    p.add_argument("--em-iter", type=int, default=300)
    a = p.parse_args()
    main(a.source, a.em_iter)
