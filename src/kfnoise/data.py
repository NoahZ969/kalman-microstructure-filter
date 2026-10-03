"""Data sources: synthetic ground truth, live crypto candles, and the bundled CSV sample."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "data" / "btcusdt_1m_sample.csv"


def simulate_local_level(n: int = 500, q_std: float = 0.5, r_std: float = 2.0,
                         x0: float = 100.0, seed: int | None = 42):
    """Random-walk efficient price + i.i.d. Gaussian microstructure noise.

    Returns (true_price, observed_price). True Q = q_std**2, R = r_std**2.
    """
    rng = np.random.default_rng(seed)
    true = x0 + np.cumsum(rng.normal(0.0, q_std, n))
    obs = true + rng.normal(0.0, r_std, n)
    return true, obs


def fetch_binance_public(symbol: str = "BTCUSDT", interval: str = "1m",
                         limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Binance's public market-data mirror (no API key,
    works from regions where api.binance.com is geo-blocked)."""
    url = ("https://data-api.binance.vision/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={limit}")
    with urllib.request.urlopen(url, timeout=20) as r:
        rows = json.loads(r.read())
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]].astype(float)
    return df


def fetch_ccxt(symbol: str = "BTC/USDT", timeframe: str = "1m", limit: int = 1000,
               exchanges=("binance", "binanceus", "kraken", "coinbase")) -> pd.DataFrame:
    """Fetch OHLCV through ccxt, trying several exchanges in order."""
    import ccxt  # optional dependency

    errors = []
    for name in exchanges:
        try:
            ex = getattr(ccxt, name)()
            sym = symbol if symbol in ex.load_markets() else symbol.replace("USDT", "USD")
            ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low",
                                              "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.attrs["source"] = f"{name}:{sym}"
            return df
        except Exception as e:  # noqa: BLE001 - try the next exchange
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("all ccxt exchanges failed -> " + "; ".join(errors))


def load_prices(source: str = "auto", limit: int = 1000) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame with 'timestamp' and 'close', description of source).

    source: "live" (ccxt, then Binance public mirror), "csv" (bundled sample),
            or "auto" (live, falling back to the bundled sample).
    """
    if source in ("live", "auto"):
        try:
            df = fetch_ccxt(limit=limit)
            return df, df.attrs.get("source", "ccxt")
        except Exception as e1:  # noqa: BLE001
            try:
                return fetch_binance_public(limit=limit), "binance public mirror"
            except Exception as e2:  # noqa: BLE001
                if source == "live":
                    raise RuntimeError(f"live fetch failed: {e1} | {e2}") from e2
                print(f"[data] live fetch failed ({type(e2).__name__}); "
                      "using bundled CSV sample")
    df = pd.read_csv(SAMPLE_CSV, parse_dates=["timestamp"])
    return df, f"bundled sample ({SAMPLE_CSV.name})"


def to_log_bps(prices) -> np.ndarray:
    """log-price in basis points: well-scaled for the optimiser, model-consistent
    (the local-level model is specified on log-prices)."""
    return 1e4 * np.log(np.asarray(prices, dtype=float))
