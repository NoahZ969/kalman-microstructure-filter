"""Phase-4 diagnostics: the numbers a quant cares about, not just the plot."""
from __future__ import annotations

import numpy as np
from scipy import stats


def acf(x, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation function, lags 0..nlags."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[k:], x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value (H0: no autocorrelation up to `lags`)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    r = acf(x, lags)[1:]
    Qstat = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(Qstat), float(stats.chi2.sf(Qstat, lags))


def sma(x, window: int) -> np.ndarray:
    """Trailing simple moving average (causal), NaN-padded."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def lag_vs_reference(est, ref, max_lag: int = 30) -> int:
    """Lag (in samples) that maximises the correlation of est's increments with ref's.

    0 means the estimate is in phase with the reference; k > 0 means it trails by k.
    """
    de, dr = np.diff(est), np.diff(ref)
    mask = ~(np.isnan(de) | np.isnan(dr))
    de, dr = de[mask], dr[mask]
    best, best_c = 0, -np.inf
    for k in range(0, max_lag + 1):
        c = np.corrcoef(de[k:], dr[:de.size - k])[0, 1]
        if c > best_c:
            best, best_c = k, c
    return best


def summarize(raw, filtered, innovations=None, truth=None) -> dict:
    """Variance reduction, lag-1 autocorrelation, whiteness of innovations, RMSE."""
    r_raw = np.diff(np.asarray(raw, float))
    r_flt = np.diff(np.asarray(filtered, float))
    out = {
        "var_raw_returns": float(np.var(r_raw)),
        "var_filtered_returns": float(np.var(r_flt)),
        "variance_reduction_pct": float(100 * (1 - np.var(r_flt) / np.var(r_raw))),
        "acf1_raw": float(acf(r_raw, 1)[1]),
        "acf1_filtered": float(acf(r_flt, 1)[1]),
    }
    if innovations is not None:
        q, p = ljung_box(np.asarray(innovations)[1:], 10)
        out.update({"ljung_box_Q10_innov": q, "ljung_box_p_innov": p})
    if truth is not None:
        truth = np.asarray(truth, float)
        out["rmse_raw_vs_truth"] = float(np.sqrt(np.mean((raw - truth) ** 2)))
        out["rmse_filtered_vs_truth"] = float(np.sqrt(np.mean((filtered - truth) ** 2)))
    return out
