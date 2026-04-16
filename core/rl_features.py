"""RL Feature extraction — shared between training and inference.

Sprint 6E: Extended the state vector from 10 to 16 dimensions. The previous
10-dim state under-represented volatility scale/structure (only ATR%) and
session context (only hour), and used a naive RSI clip that treated every
symbol the same. Concretely:

  old (10 dims): RSI_norm, BB_%B, MACD_hist, vol_ratio, ATR_%,
                 mom_5, mom_20, hour_sin, hour_cos, vol_ratio_proxy

  new (16 dims): rsi_z, bb_%B, macd_hist, atr_%,
                 mom_5, mom_20, hour_sin, hour_cos,
                 vol_z (replaces vol_ratio),
                 realized_vol_5, realized_vol_20, vol_of_vol, spread_z,
                 time_to_close,
                 regime_stress, regime_crisis

Why the changes:
  - rsi_z is tanh(z-score) instead of raw/100. RSI=70 means different things
    for steady ETFs vs volatile crypto; the z-score captures how anomalous
    the current reading is relative to the symbol's own recent history.
  - vol_z replaces vol_ratio for the same reason — ratios bias toward heavy
    volume names; z-scores are symbol-agnostic.
  - realized_vol_5/20 + vol_of_vol give the agent an explicit volatility
    scale signal (useful for selecting between momentum/mean-rev) plus a
    volatility-of-volatility signal (useful for detecting regime shifts).
  - spread_z (using (high-low)/close as a proxy) flags liquidity stress.
  - time_to_close lets the agent prefer faster strategies near session end.
  - regime_stress/regime_crisis are one-hot bits from the Sprint 6C detector.
    During training they're 0 (the detector runs live, not on historical
    minute bars), so the model's weights for these inputs stay small and
    the policy doesn't pivot wildly when they flip at inference time.

All features are kept in roughly [-1, 1] or [0, 1]. NaNs are replaced with 0.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

# Sprint 6E: 10 → 16. Keep a backward-compat alias so any callsite that
# imported NUM_FEATURES still picks up the new value.
NUM_FEATURES = 16

# Strategy names in fixed order (action index → strategy key)
STRATEGY_KEYS = [
    "mean_reversion_aggressive",
    "mean_reversion",
    "volume_profile",
    "momentum",
    "macd_crossover",
    "triple_ema",
    "rsi_divergence",
    "scalper",
    "ensemble",
]

NUM_STRATEGIES = len(STRATEGY_KEYS)


# ── technical-indicator helpers (unchanged) ─────────────────────────────────


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger_pct_b(series: pd.Series, period: int = 20,
                            std: float = 2.0) -> pd.Series:
    """Compute Bollinger Band %B (position within bands)."""
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    band_width = upper - lower
    return (series - lower) / band_width.replace(0, np.nan)


def compute_macd_hist(series: pd.Series) -> pd.Series:
    """Compute MACD histogram (MACD - Signal)."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# ── Sprint 6E helpers ───────────────────────────────────────────────────────


def _tanh_zscore(series: pd.Series, window: int = 20) -> float:
    """Tanh of the z-score of the latest value vs the trailing `window`.

    Using tanh bounds the output to (-1, 1) without clipping — the tails
    are squashed smoothly so a 5σ move doesn't look like a 3σ move to the
    model, but both sit at saturation.
    """
    if series is None:
        return 0.0
    clean = series.dropna()
    if len(clean) < window + 1:
        # Fall back to whatever we have, but need at least 2 samples for std.
        if len(clean) < 2:
            return 0.0
        recent = clean
    else:
        recent = clean.tail(window)
    mean = recent.mean()
    std = recent.std()
    if not np.isfinite(std) or std < 1e-9:
        return 0.0
    latest = clean.iloc[-1]
    if pd.isna(latest):
        return 0.0
    return float(np.tanh((latest - mean) / std))


def _realized_vol(close: pd.Series, window: int) -> float:
    """Std of the last `window` percent-returns. Not annualized — raw signal."""
    if close is None or len(close) < window + 1:
        return 0.0
    rets = close.pct_change().dropna()
    if len(rets) < max(2, window):
        # Still compute from whatever we have if we have ≥ 2 samples.
        if len(rets) < 2:
            return 0.0
        sample = rets
    else:
        sample = rets.tail(window)
    val = sample.std()
    if not np.isfinite(val):
        return 0.0
    return float(val)


def _vol_of_vol(close: pd.Series, short: int = 5, window: int = 20) -> float:
    """Std of the rolling `short`-bar realized vol over the last `window` bars.

    Spikes when volatility is itself becoming more erratic — a leading
    indicator for regime change that a single vol number misses.
    """
    if close is None or len(close) < short + window:
        return 0.0
    rets = close.pct_change()
    rolling = rets.rolling(short).std().dropna()
    if len(rolling) < 2:
        return 0.0
    sample = rolling.tail(window)
    if len(sample) < 2:
        return 0.0
    val = sample.std()
    if not np.isfinite(val):
        return 0.0
    return float(val)


def _spread_proxy_z(df: pd.DataFrame, window: int = 20) -> float:
    """Tanh z-score of (high-low)/close — proxy for liquidity / spread."""
    if df is None or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0
    closes = df["close"].replace(0, np.nan)
    spread = ((df["high"] - df["low"]) / closes).dropna()
    if len(spread) < 2:
        return 0.0
    return _tanh_zscore(spread, window=window)


def _time_to_close(df: pd.DataFrame) -> float:
    """Minutes until 16:00 (equity close), normalized to [0, 1].

    For crypto (24/7) or missing timestamps we return 0.5 — a neutral value
    that won't bias the policy. The equity close proxy is fine for crypto
    because the policy we want ("near session end, prefer faster strategies")
    is equity-specific and should be inert for 24/7 markets.
    """
    idx = df.index if df is not None else None
    if idx is None or not hasattr(idx, "hour"):
        return 0.5
    try:
        last = idx[-1]
        minute_of_day = last.hour * 60 + last.minute
    except Exception:
        return 0.5
    eq_close = 16 * 60
    # Clamp: before open → full session left, after close → already past.
    diff_min = eq_close - minute_of_day
    if diff_min < 0:
        return 0.0
    # Equity session is 6.5h = 390 min; normalize by that for fine resolution
    # within the trading day, clamped to [0, 1].
    return float(max(0.0, min(1.0, diff_min / 390.0)))


# ── public entry point ─────────────────────────────────────────────────────


def extract_features(df: pd.DataFrame,
                     regime: Optional[str] = None) -> Optional[np.ndarray]:
    """Extract 16-dimensional feature vector from OHLCV DataFrame.

    Args:
        df: DataFrame with columns close, high, low, volume. May have a
            datetime index (preferred — enables hour + time_to_close features).
        regime: Optional current market regime string from `core.regime_detector`.
            One of 'low_vol' / 'normal' / 'high_vol' / 'crisis'. When None, the
            regime bits are set to 0 (the default used during training, since
            the regime detector runs on live equity bars, not the minute data
            RL trains on).

    Returns:
        numpy array of shape (NUM_FEATURES,) or None if insufficient data.
    """
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    latest = close.iloc[-1]

    # 1. RSI as tanh(z-score) — symbol-agnostic overbought/oversold signal.
    rsi_series = compute_rsi(close)
    rsi_z = _tanh_zscore(rsi_series)

    # 2. Bollinger %B (same formulation as before; already scale-invariant).
    bb_pct_b = compute_bollinger_pct_b(close)
    bb_val = bb_pct_b.iloc[-1] if not pd.isna(bb_pct_b.iloc[-1]) else 0.5
    bb_val = max(-1.0, min(2.0, bb_val))

    # 3. MACD histogram normalized by price, clipped to [-1, 1].
    macd_hist = compute_macd_hist(close)
    macd_raw = macd_hist.iloc[-1] / latest if not pd.isna(macd_hist.iloc[-1]) else 0.0
    macd_val = max(-0.1, min(0.1, macd_raw)) * 10

    # 4. ATR as a percentage of price, clipped.
    if {"high", "low"}.issubset(df.columns):
        atr = compute_atr(df)
        atr_raw = atr.iloc[-1] / latest if not pd.isna(atr.iloc[-1]) else 0.02
        atr_pct = min(0.1, atr_raw) * 10
    else:
        atr_pct = 0.2

    # 5. 5-bar momentum.
    mom_5d = 0.0
    if len(close) >= 6:
        mom_5d = (latest - close.iloc[-5]) / close.iloc[-5]
    mom_5d = max(-0.2, min(0.2, mom_5d)) * 5

    # 6. 20-bar momentum.
    mom_20d = 0.0
    if len(close) >= 21:
        mom_20d = (latest - close.iloc[-20]) / close.iloc[-20]
    mom_20d = max(-0.5, min(0.5, mom_20d)) * 2

    # 7-8. Cyclical hour encoding (same as old — works for any tzaware index).
    if hasattr(df.index, "hour"):
        try:
            hour = df.index[-1].hour
        except Exception:
            hour = 12
    else:
        hour = 12
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # 9. Volume z-score (replaces raw ratio — symbol-agnostic).
    if "volume" in df.columns and len(df) >= 21:
        vol_z = _tanh_zscore(df["volume"].astype(float))
    else:
        vol_z = 0.0

    # 10-11. Raw realized volatility at short and medium horizons, clipped.
    rv_5_raw = _realized_vol(close, window=5)
    rv_20_raw = _realized_vol(close, window=20)
    # Most intraday returns are in [0, 1%] range; clip at 5% for tail safety.
    rv_5 = min(0.05, rv_5_raw) * 20     # → [0, 1]
    rv_20 = min(0.05, rv_20_raw) * 20   # → [0, 1]

    # 12. Vol-of-vol over 20 bars of 5-bar realized vol.
    vov_raw = _vol_of_vol(close, short=5, window=20)
    vov = min(0.03, vov_raw) * 33.33    # → ~[0, 1]

    # 13. Spread proxy z-score (tanh-bounded).
    spread_z = _spread_proxy_z(df, window=20)

    # 14. Time-to-close (normalized minutes).
    ttc = _time_to_close(df)

    # 15-16. Regime one-hot bits. Two dims chosen so both low_vol (trending)
    # and normal encode as (0, 0) — the realized-vol features already
    # distinguish those — while the bits only flip for stress regimes where
    # strategy preference changes qualitatively.
    regime_stress = 1.0 if regime in ("high_vol", "crisis") else 0.0
    regime_crisis = 1.0 if regime == "crisis" else 0.0

    features = np.array([
        rsi_z, bb_val, macd_val, atr_pct,
        mom_5d, mom_20d, hour_sin, hour_cos,
        vol_z,
        rv_5, rv_20, vov, spread_z, ttc,
        regime_stress, regime_crisis,
    ], dtype=np.float32)

    # Any residual NaN → 0 so downstream MLP never trains on NaN gradients.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
