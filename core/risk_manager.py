"""Risk management — position sizing, stop-losses, drawdown protection."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from utils.logger import log


@dataclass
class TrailingStop:
    symbol: str
    entry_price: float
    highest_price: float
    stop_pct: float = Config.TRAILING_STOP_PCT
    entry_time: float = 0.0

    @property
    def stop_price(self) -> float:
        return self.highest_price * (1 - self.stop_pct)

    def update(self, current_price: float) -> bool:
        """Update trailing stop. Returns True if stop was triggered."""
        import math
        if math.isnan(current_price) or math.isinf(current_price):
            return False  # Don't trigger or update on bad data
        if current_price > self.highest_price:
            self.highest_price = current_price
        return current_price <= self.stop_price

    def hours_held(self) -> float:
        if self.entry_time <= 0:
            return 0.0
        return (time.time() - self.entry_time) / 3600.0


@dataclass
class RiskManager:
    starting_equity: float = 0.0
    daily_start_equity: float = 0.0
    trailing_stops: dict[str, TrailingStop] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    daily_drawdown: float = 0.0
    _last_daily_reset: float = 0.0

    def initialize(self, equity: float):
        self.starting_equity = equity
        self.daily_start_equity = equity
        self._last_daily_reset = time.time()

    def check_daily_reset(self, current_equity: float):
        """Reset daily tracking at calendar day boundary (market timezone)."""
        now = datetime.now(Config.MARKET_TZ)
        last_reset_dt = datetime.fromtimestamp(self._last_daily_reset, tz=Config.MARKET_TZ) if self._last_daily_reset else now
        if now.date() > last_reset_dt.date():
            self.daily_start_equity = current_equity
            self._last_daily_reset = time.time()
            self.daily_drawdown = 0.0
            self.halted = False
            self.halt_reason = ""
            log.info("Daily risk counters reset (new trading day)")

    def check_drawdown(self, current_equity: float) -> bool:
        """Check if daily drawdown limit is hit. Returns True if trading should stop."""
        self.check_daily_reset(current_equity)

        if self.daily_start_equity <= 0:
            self.halted = True
            self.halt_reason = "Daily start equity is zero or negative — cannot compute drawdown"
            log.warning(self.halt_reason)
            return True

        drawdown = (self.daily_start_equity - current_equity) / self.daily_start_equity
        self.daily_drawdown = drawdown

        if drawdown >= Config.DAILY_DRAWDOWN_LIMIT:
            self.halted = True
            self.halt_reason = f"Daily drawdown limit hit: {drawdown:.1%}"
            log.warning(self.halt_reason)
            return True
        return False

    def check_intraday_halt(self, current_equity: float,
                           max_daily_loss_usd: float,
                           unrealized_pnl: float = 0.0) -> bool:
        """Sprint 5D: Dollar-aware intraday P&L circuit breaker.

        Halts trading when today's realized loss exceeds the profile's
        dollar-based threshold, OR when unrealized losses are 1.5x that (early
        warning). Profile-aware caps mean a $800 account halts at $24 (Beginner
        3%-or-$50 cap), not the $80 that a pure 10% daily DD would allow.

        Args:
            current_equity: The broker's current account equity.
            max_daily_loss_usd: The profile's dollar threshold
                (from UserProfile.max_daily_loss_usd(equity)).
            unrealized_pnl: Sum of mark-to-market for open positions
                (0 for fully-realized accounting, negative for open losses).

        Returns:
            True if trading should halt, False otherwise.
        """
        self.check_daily_reset(current_equity)

        if self.daily_start_equity <= 0 or max_daily_loss_usd <= 0:
            return False

        # Realized loss = how far current equity (excluding open MTM) is below start
        # current_equity already includes unrealized PnL in most broker APIs, so we
        # decompose: realized = (current_equity - unrealized) - daily_start
        realized_pnl = (current_equity - unrealized_pnl) - self.daily_start_equity
        threshold = max_daily_loss_usd

        if realized_pnl <= -threshold:
            self.halted = True
            self.halt_reason = (
                f"Intraday realized loss ${realized_pnl:.2f} breached "
                f"threshold -${threshold:.2f}"
            )
            log.warning(self.halt_reason)
            return True

        # Unrealized early-warning (1.5x realized threshold): paper loss but not yet taken
        early_warn_threshold = threshold * 1.5
        if unrealized_pnl <= -early_warn_threshold:
            self.halted = True
            self.halt_reason = (
                f"Intraday unrealized loss ${unrealized_pnl:.2f} breached "
                f"early-warning threshold -${early_warn_threshold:.2f}"
            )
            log.warning(self.halt_reason)
            return True

        return False

    def calculate_position_size(self, equity: float) -> float:
        """Calculate max dollar amount for a new position."""
        return equity * Config.MAX_POSITION_PCT

    def estimate_var_pct(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Sprint 5F: parametric VaR estimate from recent returns.

        Returns the expected worst-case loss fraction at the given confidence
        level — e.g. 0.05 means "we expect to lose at most 5% of position value
        in 95% of outcomes". Used to cap individual position sizes so a single
        trade can't risk more than `profile.max_var_contribution_pct` of equity.

        Parametric (normal-distribution) approach: VaR ≈ z * sigma where z is
        the standard score for the confidence level (1.645 for 95%, 2.326 for 99%).
        Faster than Monte Carlo and good enough for a per-trade pre-trade gate.

        Returns 0.0 if there's not enough data to estimate (caller should treat
        as "no constraint" — fail-open).
        """
        if df is None or len(df) < 10:
            return 0.0
        try:
            closes = df["close"].astype(float).values
        except (KeyError, AttributeError):
            return 0.0
        if len(closes) < 2:
            return 0.0
        # Bar-to-bar returns
        returns = np.diff(closes) / closes[:-1]
        # Standard scores for common confidence levels
        z_scores = {0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(round(confidence, 2), 1.645)
        sigma = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        return max(0.0, z * sigma)

    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if trailing stop was hit for a position."""
        stop = self.trailing_stops.get(symbol)
        if stop is None:
            return False
        triggered = stop.update(current_price)
        if triggered:
            log.warning(
                f"Trailing stop triggered for {symbol} at ${current_price:.2f} "
                f"(stop: ${stop.stop_price:.2f}, high: ${stop.highest_price:.2f})"
            )
        return triggered

    def register_entry(self, symbol: str, entry_price: float, df: pd.DataFrame = None):
        """Register a new position for trailing stop tracking."""
        # v3: ATR-based stops when enabled and data available
        if Config.ATR_STOPS_ENABLED and df is not None and len(df) >= 15:
            multiplier = Config.ATR_STOP_MULTIPLIER_CRYPTO if Config.is_crypto(symbol) else Config.ATR_STOP_MULTIPLIER
            stop_pct = self.calculate_atr_stop_pct(df, multiplier)
        else:
            # Wider stops for crypto (volatile), tighter for equities
            if Config.is_crypto(symbol):
                stop_pct = 0.04   # 4% for crypto
            else:
                stop_pct = 0.025  # 2.5% for equities

        self.trailing_stops[symbol] = TrailingStop(
            symbol=symbol,
            entry_price=entry_price,
            highest_price=entry_price,
            stop_pct=stop_pct,
            entry_time=time.time(),
        )
        log.info(f"Trailing stop registered for {symbol} at ${entry_price:.2f} (stop: {stop_pct:.1%})")

    def unregister(self, symbol: str):
        self.trailing_stops.pop(symbol, None)

    def can_trade(self, current_equity: float) -> bool:
        """Master check — can we open new positions?"""
        if self.halted:
            return False
        if self.check_drawdown(current_equity):
            return False
        return True

    def is_max_hold_exceeded(self, symbol: str) -> bool:
        stop = self.trailing_stops.get(symbol)
        if stop is None:
            return False
        max_hours = Config.MAX_HOLD_HOURS_CRYPTO if Config.is_crypto(symbol) else Config.MAX_HOLD_HOURS
        if max_hours <= 0:
            return False
        hours = stop.hours_held()
        if hours >= max_hours:
            log.warning(f"Max hold exceeded: {symbol} held {hours:.1f}h >= {max_hours}h limit")
            return True
        return False

    # ── v3 Sprint 2: ATR-based stops ────────────────────────────────

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute Average True Range from OHLC data."""
        if df is None or len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    @staticmethod
    def calculate_atr_stop_pct(df: pd.DataFrame, multiplier: float = 2.0) -> float:
        """Compute ATR-based stop-loss percentage.

        Returns a percentage (e.g., 0.03 for 3%) clamped between 0.5% and 5%.
        """
        if df is None or len(df) < 15:
            return 0.02  # fallback to default
        atr = RiskManager.calculate_atr(df)
        current_price = float(df["close"].iloc[-1])
        if current_price <= 0:
            return 0.02
        atr_pct = (atr / current_price) * multiplier
        return max(0.005, min(atr_pct, 0.08))

    # ── v3 Sprint 2: Volatility-adjusted sizing ────────────────────

    @staticmethod
    def calculate_volatility_adjusted_size(equity: float, df: pd.DataFrame,
                                           base_pct: float = 0.15) -> float:
        """Scale position size inversely with recent volatility.

        High volatility → smaller position, low volatility → larger position.
        Returns dollar amount clamped between 1% and base_pct of equity.
        """
        if df is None or len(df) < 20:
            return equity * base_pct

        returns = df["close"].pct_change().dropna()
        if len(returns) < 10:
            return equity * base_pct

        recent_vol = float(returns.tail(20).std())
        if recent_vol <= 0:
            return equity * base_pct

        # Baseline: typical daily vol for equities ~1-2%
        baseline_vol = 0.015
        vol_ratio = baseline_vol / recent_vol  # >1 if low vol, <1 if high vol
        vol_ratio = max(0.2, min(vol_ratio, 2.0))  # clamp scaling factor

        adjusted_pct = base_pct * vol_ratio
        adjusted_pct = max(0.01, min(adjusted_pct, base_pct))

        return equity * adjusted_pct

    # ── v3 Sprint 2: Correlation check ──────────────────────────────

    @staticmethod
    def check_correlation(new_df: pd.DataFrame, existing_dfs: dict[str, pd.DataFrame],
                          threshold: float = 0.7) -> bool:
        """Check if a new position is too correlated with existing positions.

        Returns True if safe to enter (no high correlation found).
        Returns False if any existing position has correlation >= threshold.
        """
        if new_df is None or len(new_df) < 20 or not existing_dfs:
            return True

        new_returns = new_df["close"].pct_change().dropna().tail(50)
        if len(new_returns) < 10:
            return True

        for sym, ex_df in existing_dfs.items():
            if ex_df is None or len(ex_df) < 20:
                continue
            ex_returns = ex_df["close"].pct_change().dropna().tail(50)
            if len(ex_returns) < 10:
                continue

            # Align lengths
            min_len = min(len(new_returns), len(ex_returns))
            corr = float(np.corrcoef(
                new_returns.values[-min_len:],
                ex_returns.values[-min_len:],
            )[0, 1])

            if abs(corr) >= threshold:
                log.info(f"Correlation check: blocked — corr with {sym} = {corr:.2f} (threshold: {threshold})")
                return False

        return True
