"""RL Feature extraction — shared between training and inference.

Extracts a 10-dimensional state vector from OHLCV DataFrame:
  [RSI_norm, BB_%B, MACD_hist_norm, vol_ratio, ATR_pct,
   mom_5d, mom_20d, hour_sin, hour_cos, regime_proxy]

All features are normalized to roughly [-1, 1] or [0, 1] range.
"""

import math
import numpy as np
import pandas as pd

# Number of features in the state vector
NUM_FEATURES = 10

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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger_pct_b(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
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


def extract_features(df: pd.DataFrame) -> np.ndarray | None:
    """Extract 10-dimensional feature vector from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: close, high, low, volume.
            Optionally has a datetime index for hour features.

    Returns:
        numpy array of shape (NUM_FEATURES,) or None if insufficient data.
    """
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    latest = close.iloc[-1]

    # 1. RSI normalized to [0, 1]
    rsi = compute_rsi(close)
    rsi_norm = rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5

    # 2. Bollinger %B (already ~[0, 1], can go outside)
    bb_pct_b = compute_bollinger_pct_b(close)
    bb_val = bb_pct_b.iloc[-1] if not pd.isna(bb_pct_b.iloc[-1]) else 0.5
    bb_val = max(-1.0, min(2.0, bb_val))

    # 3. MACD histogram normalized by price
    macd_hist = compute_macd_hist(close)
    macd_val = macd_hist.iloc[-1] / latest if not pd.isna(macd_hist.iloc[-1]) else 0.0
    macd_val = max(-0.1, min(0.1, macd_val)) * 10  # scale to [-1, 1]

    # 4. Volume ratio (current vs 20-day average)
    if "volume" in df.columns:
        avg_vol = df["volume"].tail(20).mean()
        vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        vol_ratio = min(3.0, vol_ratio) / 3.0  # normalize to [0, 1]
    else:
        vol_ratio = 0.5

    # 5. ATR as percentage of price
    if all(col in df.columns for col in ["high", "low"]):
        atr = compute_atr(df)
        atr_pct = atr.iloc[-1] / latest if not pd.isna(atr.iloc[-1]) else 0.02
        atr_pct = min(0.1, atr_pct) * 10  # normalize to [0, 1]
    else:
        atr_pct = 0.2

    # 6-7. Momentum (5-day and 20-day returns)
    mom_5d = (latest - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0.0
    mom_5d = max(-0.2, min(0.2, mom_5d)) * 5  # scale to [-1, 1]

    mom_20d = (latest - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0.0
    mom_20d = max(-0.5, min(0.5, mom_20d)) * 2  # scale to [-1, 1]

    # 8-9. Hour features (cyclical encoding)
    if hasattr(df.index, 'hour'):
        hour = df.index[-1].hour
    else:
        hour = 12  # default to noon if no timestamp
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # 10. Regime proxy: rolling 20-day volatility vs 60-day
    if len(close) >= 60:
        vol_20 = close.pct_change().tail(20).std()
        vol_60 = close.pct_change().tail(60).std()
        regime = vol_20 / vol_60 if vol_60 > 0 else 1.0
        regime = max(0.0, min(3.0, regime)) / 3.0  # normalize to [0, 1]
    else:
        regime = 0.5

    features = np.array([
        rsi_norm, bb_val, macd_val, vol_ratio, atr_pct,
        mom_5d, mom_20d, hour_sin, hour_cos, regime
    ], dtype=np.float32)

    # Replace any NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    return features
