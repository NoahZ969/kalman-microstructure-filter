"""Library benchmark: the same local-level model fitted with pykalman's EM algorithm."""
from __future__ import annotations

import numpy as np

from .filter import FitResult


def run_pykalman(z, n_iter: int = 50, smooth: bool = False):
    """Fit Q (transition_covariance) and R (observation_covariance) with EM.

    Fixes vs. the original prototype:
      * n_iter raised from 10 to 50 -- EM converges slowly (linearly) and
        10 iterations were typically not enough for the R estimate to settle;
      * the initial covariances are scaled to the data instead of 1.0;
      * `smooth` makes the causal/acausal choice explicit (the prototype's
        comment said "smooth" but it called `filter`).

    Returns
    -------
    states : np.ndarray   filtered (or smoothed) state means
    fit    : FitResult
    """
    from pykalman import KalmanFilter  # imported lazily: optional dependency

    z = np.asarray(z, dtype=float)
    g0 = float(np.var(np.diff(z)))
    kf = KalmanFilter(
        transition_matrices=[[1.0]],      # x_t = x_{t-1} + w_t
        observation_matrices=[[1.0]],     # z_t = x_t + v_t
        transition_covariance=[[0.5 * g0]],
        observation_covariance=[[0.25 * g0]],
        initial_state_mean=[z[0]],
        initial_state_covariance=[[g0]],
        n_dim_state=1,
        n_dim_obs=1,
    )
    kf = kf.em(z.reshape(-1, 1), n_iter=n_iter,
               em_vars=["transition_covariance", "observation_covariance"])
    means, _ = (kf.smooth if smooth else kf.filter)(z.reshape(-1, 1))
    Q = float(np.asarray(kf.transition_covariance).ravel()[0])
    R = float(np.asarray(kf.observation_covariance).ravel()[0])
    nll = float(-kf.loglikelihood(z.reshape(-1, 1)))
    return means.ravel(), FitResult(Q, R, nll, f"pykalman EM ({n_iter} iter)")
