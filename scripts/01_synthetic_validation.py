"""Phase 1/3 -- validate the hand-written filter + MLE on data with known ground truth.

1. Single realisation (the original prototype's experiment: n=500, sigma_w=0.5, sigma_v=2).
2. Monte-Carlo: does MLE recover (Q, R) without bias? Does the Roll moment
   estimator agree? How biased is the prototype's R = Var(dz)/2 heuristic?

Usage:  python scripts/01_synthetic_validation.py [--n-mc 300]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfnoise import (fit_constrained_mle, fit_mle, half_diff_variance,  # noqa: E402
                     kalman_filter, roll_noise_variance, rts_smoother)
from kfnoise import metrics as M  # noqa: E402
from kfnoise.data import simulate_local_level  # noqa: E402
from kfnoise.plotting import C, apply_style, plt  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"


def single_run():
    true_q, true_r = 0.5 ** 2, 2.0 ** 2
    truth, obs = simulate_local_level(n=500, q_std=0.5, r_std=2.0, seed=42)
    fit = fit_mle(obs)
    res = kalman_filter(obs, fit.Q, fit.R)
    xs, _ = rts_smoother(res, fit.Q)
    summary = M.summarize(obs, res.x_filt, res.innov, truth)
    summary["rmse_smoothed_vs_truth"] = float(np.sqrt(np.mean((xs - truth) ** 2)))
    print(f"True Q = {true_q:.4f} | MLE Q = {fit.Q:.4f}")
    print(f"True R = {true_r:.4f} | MLE R = {fit.R:.4f}")
    print(f"Steady-state gain K = {fit.steady_state_gain:.3f}")
    print(f"RMSE vs truth: raw {summary['rmse_raw_vs_truth']:.3f} | filtered "
          f"{summary['rmse_filtered_vs_truth']:.3f} | smoothed "
          f"{summary['rmse_smoothed_vs_truth']:.3f}")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    t = np.arange(obs.size)
    ax.plot(t, obs, color=C["raw"], lw=1, label="Observed (noisy)")
    ax.plot(t, truth, color=C["truth"], lw=1.2, ls="--", label="True efficient price (hidden)")
    ax.plot(t, res.x_filt, color=C["s1"], lw=1.6, label="Kalman filter (causal)")
    ax.plot(t, xs, color=C["s2"], lw=1.6, label="RTS smoother (acausal)")
    ax.set_title(f"Synthetic local-level model — MLE: Q={fit.Q:.3f} (true {true_q:.2f}), "
                 f"R={fit.R:.3f} (true {true_r:.2f})")
    ax.set_xlabel("step")
    ax.set_ylabel("price")
    ax.legend(ncol=4, loc="upper left")
    fig.savefig(FIG / "synthetic_filter.png")
    plt.close(fig)
    return {"true_Q": true_q, "true_R": true_r, "mle_Q": fit.Q, "mle_R": fit.R,
            "K_ss": fit.steady_state_gain, **summary}


def monte_carlo(n_mc: int, n: int = 1000):
    true_q, true_r = 0.5 ** 2, 2.0 ** 2
    rows = []
    for i in range(n_mc):
        _, obs = simulate_local_level(n=n, q_std=0.5, r_std=2.0, seed=1000 + i)
        f = fit_mle(obs)
        rows.append((f.Q, f.R, roll_noise_variance(obs), half_diff_variance(obs)))
    a = np.array(rows)
    out = {
        "n_mc": n_mc, "n_obs": n, "true_Q": true_q, "true_R": true_r,
        "mle_Q_mean": float(a[:, 0].mean()), "mle_Q_std": float(a[:, 0].std()),
        "mle_R_mean": float(a[:, 1].mean()), "mle_R_std": float(a[:, 1].std()),
        "roll_R_mean": float(a[:, 2].mean()), "roll_R_std": float(a[:, 2].std()),
        "halfdiff_R_mean": float(a[:, 3].mean()), "halfdiff_R_std": float(a[:, 3].std()),
    }
    print(f"\nMonte Carlo ({n_mc} paths x {n} obs):")
    print(f"  MLE  Q = {out['mle_Q_mean']:.3f} ± {out['mle_Q_std']:.3f}   (true {true_q})")
    print(f"  MLE  R = {out['mle_R_mean']:.3f} ± {out['mle_R_std']:.3f}   (true {true_r})")
    print(f"  Roll R = {out['roll_R_mean']:.3f} ± {out['roll_R_std']:.3f}")
    print(f"  Var(dz)/2 R = {out['halfdiff_R_mean']:.3f} ± {out['halfdiff_R_std']:.3f}"
          "   <- biased: equals R + Q/2")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(a[:, 0], bins=30, color=C["s1"], edgecolor="white", linewidth=1)
    axes[0].axvline(true_q, color=C["truth"], ls="--", lw=1.2)
    axes[0].annotate("true Q", (true_q, axes[0].get_ylim()[1] * 0.92),
                     xytext=(4, 0), textcoords="offset points", fontsize=9)
    axes[0].set_title("MLE of Q (process variance)")
    bins = np.linspace(a[:, 1:].min(), a[:, 1:].max(), 40)
    axes[1].hist(a[:, 1], bins=bins, color=C["s1"], alpha=0.85, label="MLE",
                 edgecolor="white", linewidth=0.8)
    axes[1].hist(a[:, 2], bins=bins, color=C["s3"], alpha=0.7, label="Roll moment estimator",
                 edgecolor="white", linewidth=0.8)
    axes[1].hist(a[:, 3], bins=bins, color=C["s2"], alpha=0.85, label="Var(Δz)/2 (prototype)",
                 edgecolor="white", linewidth=0.8)
    axes[1].axvline(true_r, color=C["truth"], ls="--", lw=1.2)
    axes[1].set_title("Estimators of R (noise variance)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_monte_carlo.png")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-mc", type=int, default=300)
    args = p.parse_args()
    apply_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = {"single_run": single_run(), "monte_carlo": monte_carlo(args.n_mc)}
    (RES / "synthetic_results.json").write_text(json.dumps(out, indent=2))
    print("\nsaved figures/ and results/synthetic_results.json")
