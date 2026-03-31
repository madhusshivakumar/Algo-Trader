"""Risk management — position sizing, stop-losses, drawdown protection."""

import time
from dataclasses import dataclass, field
from config import Config
from utils.logger import log


@dataclass
class TrailingStop:
    symbol: str
    entry_price: float
    highest_price: float
    stop_pct: float = Config.TRAILING_STOP_PCT

    @property
    def stop_price(self) -> float:
        return self.highest_price * (1 - self.stop_pct)

    def update(self, current_price: float) -> bool:
        """Update trailing stop. Returns True if stop was triggered."""
        if current_price > self.highest_price:
            self.highest_price = current_price
        return current_price <= self.stop_price


@dataclass
class RiskManager:
    starting_equity: float = 0.0
    daily_start_equity: float = 0.0
    trailing_stops: dict[str, TrailingStop] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    _last_daily_reset: float = 0.0

    def initialize(self, equity: float):
        self.starting_equity = equity
        self.daily_start_equity = equity
        self._last_daily_reset = time.time()

    def check_daily_reset(self, current_equity: float):
        """Reset daily tracking every 24h."""
        if time.time() - self._last_daily_reset > 86400:
            self.daily_start_equity = current_equity
            self._last_daily_reset = time.time()
            self.halted = False
            self.halt_reason = ""
            log.info("Daily risk counters reset")

    def check_drawdown(self, current_equity: float) -> bool:
        """Check if daily drawdown limit is hit. Returns True if trading should stop."""
        self.check_daily_reset(current_equity)

        if self.daily_start_equity <= 0:
            return False

        drawdown = (self.daily_start_equity - current_equity) / self.daily_start_equity

        if drawdown >= Config.DAILY_DRAWDOWN_LIMIT:
            self.halted = True
            self.halt_reason = f"Daily drawdown limit hit: {drawdown:.1%}"
            log.warning(self.halt_reason)
            return True
        return False

    def calculate_position_size(self, equity: float) -> float:
        """Calculate max dollar amount for a new position."""
        return equity * Config.MAX_POSITION_PCT

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

    def register_entry(self, symbol: str, entry_price: float):
        """Register a new position for trailing stop tracking."""
        # Tighter stops for crypto (high frequency), wider for equities
        if Config.is_crypto(symbol):
            stop_pct = 0.015  # 1.5% for crypto
        else:
            stop_pct = 0.02   # 2.0% for equities

        self.trailing_stops[symbol] = TrailingStop(
            symbol=symbol,
            entry_price=entry_price,
            highest_price=entry_price,
            stop_pct=stop_pct,
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
