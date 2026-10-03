"""Local-level Kalman filter for market microstructure noise.

State-space model (all quantities in log-price units chosen by the caller,
e.g. basis points of log-price):

    state:        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   (efficient price, random walk)
    measurement:  z_t = x_t + v_t,       v_t ~ N(0, R)   (observed price = efficient + noise)

Physics analogy: x_t is the "signal" (e.g. a GW strain waveform), v_t is the
"detector noise" (here: bid-ask bounce, price discreteness, latency).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass
class FilterResult:
    """Output of a forward Kalman pass."""

    x_filt: np.ndarray   # E[x_t | z_1..z_t]           (causal estimate)
    P_filt: np.ndarray   # Var[x_t | z_1..z_t]
    x_pred: np.ndarray   # E[x_t | z_1..z_{t-1}]
    P_pred: np.ndarray   # Var[x_t | z_1..z_{t-1}]
    innov: np.ndarray    # v_t = z_t - x_pred_t        (innovations)
    F: np.ndarray        # Var[v_t] = P_pred_t + R
    K: np.ndarray        # Kalman gain


@dataclass
class FitResult:
    """Output of a parameter fit."""

    Q: float
    R: float
    nll: float
    method: str
    success: bool = True

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio q = Q/R (np.inf if R == 0)."""
        return np.inf if self.R == 0 else self.Q / self.R

    @property
    def steady_state_gain(self) -> float:
        return steady_state_gain(self.Q, self.R)


# ---------------------------------------------------------------------------
# Core recursions
# ---------------------------------------------------------------------------
def kalman_filter(z, Q: float, R: float, x0: float | None = None,
                  P0: float | None = None) -> FilterResult:
    """Run the scalar local-level Kalman filter.

    Parameters
    ----------
    z : array_like
        Observations (e.g. log-price in bps).
    Q, R : float
        Process and measurement noise variances.
    x0, P0 : float, optional
        Prior mean / variance of x_0. Defaults to a *diffuse* prior
        (x0 = z[0], P0 = 1e7 * var(diff(z))) so the result does not depend on
        the price scale -- the original prototype hard-coded P0 = 1, which is
        meaningless when prices are ~1e5 USD.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    if x0 is None:
        x0 = z[0]
    if P0 is None:
        P0 = 1e7 * max(np.var(np.diff(z)), 1e-12)

    x_filt = np.empty(n)
    P_filt = np.empty(n)
    x_pred = np.empty(n)
    P_pred = np.empty(n)
    innov = np.empty(n)
    F = np.empty(n)
    K = np.empty(n)

    x, P = float(x0), float(P0)
    for t in range(n):
        # 1. Time update (predict): random walk => x_{t|t-1} = x_{t-1|t-1}
        xp = x
        Pp = P + Q
        # 2. Measurement update (correct)
        Ft = Pp + R
        Kt = Pp / Ft
        vt = z[t] - xp
        x = xp + Kt * vt
        P = (1.0 - Kt) * Pp

        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = x, P
        innov[t], F[t], K[t] = vt, Ft, Kt

    return FilterResult(x_filt, P_filt, x_pred, P_pred, innov, F, K)


def rts_smoother(res: FilterResult, Q: float) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother (acausal: uses future data).

    Good for historical analysis; NOT usable for live trading (look-ahead).
    """
    n = res.x_filt.size
    xs = res.x_filt.copy()
    Ps = res.P_filt.copy()
    for t in range(n - 2, -1, -1):
        Pp_next = res.P_filt[t] + Q
        J = res.P_filt[t] / Pp_next
        xs[t] = res.x_filt[t] + J * (xs[t + 1] - res.x_filt[t])
        Ps[t] = res.P_filt[t] + J ** 2 * (Ps[t + 1] - Pp_next)
    return xs, Ps


def steady_state_gain(Q: float, R: float) -> float:
    """Closed-form steady-state Kalman gain of the local-level model.

    With q = Q/R the Riccati equation gives K = (-q + sqrt(q^2 + 4q)) / 2.
    K -> 1 as q -> inf (trust the data), K -> 0 as q -> 0 (trust the model).
    In steady state the filter is exactly an EWMA with alpha = K.
    """
    if R <= 0:
        return 1.0
    q = Q / R
    return float((-q + np.sqrt(q * q + 4.0 * q)) / 2.0)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
def neg_log_likelihood(z, Q: float, R: float, burn: int = 1) -> float:
    """Exact Gaussian negative log-likelihood via the prediction-error decomposition.

        -log L = 0.5 * sum_t [ log(2 pi F_t) + v_t^2 / F_t ]

    The first `burn` innovations are dropped (diffuse initialisation).
    """
    res = kalman_filter(z, Q, R)
    v, F = res.innov[burn:], res.F[burn:]
    return 0.5 * float(np.sum(np.log(2.0 * np.pi * F) + v * v / F))


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------
def fit_mle(z, init: tuple[float, float] | None = None) -> FitResult:
    """Unconstrained MLE of (Q, R), optimised in log-space (enforces positivity).

    This is the method that, on 1-minute BTC closes, drives R -> 0 and turns
    the filter into the identity map (K -> 1). See README "Overfitting" section.
    """
    z = np.asarray(z, dtype=float)
    if init is None:
        g0 = np.var(np.diff(z))
        init = (0.5 * g0, 0.25 * g0)
    p0 = np.log(np.asarray(init, dtype=float))

    def obj(p):
        return neg_log_likelihood(z, np.exp(p[0]), np.exp(p[1]))

    # log(R) bounded below so that "R -> 0" shows up as hitting the bound
    # instead of the optimiser wandering off to -inf.
    g0 = np.var(np.diff(z))
    lo = np.log(g0 * 1e-8)
    bounds = [(lo, np.log(g0 * 1e3)), (lo, np.log(g0 * 1e3))]
    out = minimize(obj, p0, method="L-BFGS-B", bounds=bounds)
    return FitResult(float(np.exp(out.x[0])), float(np.exp(out.x[1])),
                     float(out.fun), "MLE (unconstrained)", bool(out.success))


def roll_noise_variance(z) -> float:
    """Moment estimator of R from the first-order autocovariance of increments.

    For the local-level model, dz_t = w_t + v_t - v_{t-1}, so
        Var(dz)          = Q + 2R
        Cov(dz_t, dz_{t-1}) = -R
    hence R_hat = -gamma_1 (clipped at 0). This is the Roll (1984) bid-ask
    bounce estimator; it needs no assumption about Q.
    """
    dz = np.diff(np.asarray(z, dtype=float))
    dz = dz - dz.mean()
    gamma1 = np.mean(dz[1:] * dz[:-1])
    return float(max(-gamma1, 0.0))


def half_diff_variance(z) -> float:
    """The heuristic used in the original prototype: R = Var(dz) / 2.

    Since Var(dz) = Q + 2R, this is only correct when Q = 0, i.e. it attributes
    *all* price movement to noise. It is an upper bound on R and biases the
    filter towards over-smoothing (lag). Kept for comparison.
    """
    return float(0.5 * np.var(np.diff(np.asarray(z, dtype=float))))


def fit_constrained_mle(z, R: float | None = None, method: str = "roll") -> FitResult:
    """MLE of Q with R held fixed ("measure the noise floor, fit the signal").

    Parameters
    ----------
    R : float, optional
        Fixed measurement-noise variance. If None, estimated with `method`.
    method : {"roll", "half_diff_var"}
        "roll"          -> R = -Cov(dz_t, dz_{t-1})  (recommended)
        "half_diff_var" -> R = Var(dz)/2             (original prototype)
    """
    z = np.asarray(z, dtype=float)
    if R is None:
        R = roll_noise_variance(z) if method == "roll" else half_diff_variance(z)
    R = max(R, 1e-12)
    g0 = np.var(np.diff(z))
    lo, hi = np.log(g0 * 1e-8), np.log(g0 * 1e3)
    out = minimize_scalar(lambda lq: neg_log_likelihood(z, np.exp(lq), R),
                          bounds=(lo, hi), method="bounded")
    label = f"Constrained MLE (R: {method})"
    return FitResult(float(np.exp(out.x)), float(R), float(out.fun), label,
                     bool(out.success))


def profile_likelihood(z, log10_q_grid) -> np.ndarray:
    """Concentrated -log L as a function of q = Q/R (R profiled out analytically).

    For a fixed q, the filter innovations scale with R, so the MLE of R is
    R_hat(q) = mean(v_t^2 / f_t) where f_t = F_t / R. Useful to *see* whether
    the likelihood surface is flat or has its maximum at the boundary.
    """
    z = np.asarray(z, dtype=float)
    out = []
    for lq in np.atleast_1d(log10_q_grid):
        q = 10.0 ** lq
        res = kalman_filter(z, q, 1.0)   # R = 1 => F_t = f_t
        v, f = res.innov[1:], res.F[1:]
        R_hat = np.mean(v * v / f)
        n = v.size
        out.append(0.5 * (n * np.log(2 * np.pi * R_hat) + np.sum(np.log(f)) + n))
    return np.asarray(out)
