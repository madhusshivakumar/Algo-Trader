"""Config hot-reload — detects .env changes and reloads safe parameters.

Only risk parameters, thresholds, and feature flags are reloadable.
API keys and symbol lists require a full restart.
"""

import os
from datetime import datetime
from typing import Optional

from utils.logger import log


# Parameters that can be safely reloaded without restart
RELOADABLE_KEYS = {
    # Risk parameters
    "MAX_POSITION_PCT", "STOP_LOSS_PCT", "DAILY_DRAWDOWN_LIMIT", "TRAILING_STOP_PCT",
    # Strategy parameters
    "MOMENTUM_LOOKBACK", "RSI_OVERSOLD", "RSI_OVERBOUGHT",
    "BOLLINGER_PERIOD", "BOLLINGER_STD",
    # Feature flags
    "SENTIMENT_ENABLED", "LLM_ANALYST_ENABLED", "RL_STRATEGY_ENABLED",
    "ORDER_MANAGEMENT_ENABLED", "STATE_PERSISTENCE_ENABLED",
    "ATR_STOPS_ENABLED", "VOLATILITY_SIZING_ENABLED", "CORRELATION_CHECK_ENABLED",
    "PARALLEL_FETCH_ENABLED", "DRIFT_DETECTION_ENABLED",
    "MTF_ENABLED",
    "POSITION_RECONCILIATION_ENABLED", "RECONCILIATION_AUTO_FIX",
    "SENTIMENT_FRESHNESS_CHECK",
    # Thresholds
    "SENTIMENT_WEIGHT", "LLM_CONVICTION_WEIGHT", "MTF_WEIGHT",
    "ATR_STOP_MULTIPLIER", "CORRELATION_THRESHOLD",
    "SENTIMENT_MAX_AGE_HOURS", "LLM_FRESHNESS_CHECK", "LLM_MAX_AGE_HOURS",
    "ORDER_STALE_TIMEOUT_SECONDS", "RECONCILIATION_INTERVAL_CYCLES",
    "RECONCILIATION_ENTRY_TOLERANCE",
    "LLM_BUDGET_DAILY",
}


class ConfigReloader:
    def __init__(self, env_path: str = ".env"):
        self.env_path = env_path
        self._last_mtime: float = 0.0
        self._last_reload_time: Optional[datetime] = None
        self._update_mtime()

    def _update_mtime(self):
        """Cache the current mtime of the .env file."""
        try:
            self._last_mtime = os.stat(self.env_path).st_mtime
        except OSError:
            self._last_mtime = 0.0

    def check_for_changes(self) -> bool:
        """Check if the .env file has been modified since last check."""
        try:
            current_mtime = os.stat(self.env_path).st_mtime
            return current_mtime != self._last_mtime
        except OSError:
            return False

    def reload(self) -> dict[str, str]:
        """Reload changed values from .env. Returns dict of changed keys.

        Only reloads keys in RELOADABLE_KEYS. Updates the Config class directly.
        """
        if not os.path.exists(self.env_path):
            return {}

        # Parse .env file
        new_values = {}
        try:
            with open(self.env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Strip surrounding quotes (single or double)
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    if key in RELOADABLE_KEYS:
                        new_values[key] = value
        except OSError:
            return {}

        # Compare with current Config values and apply changes
        from config import Config
        changed = {}

        for key, new_val in new_values.items():
            current = getattr(Config, key, None)
            if current is None:
                continue

            # Convert to the right type based on current value
            try:
                if isinstance(current, bool):
                    typed_val = new_val.lower() == "true"
                elif isinstance(current, float):
                    typed_val = float(new_val)
                elif isinstance(current, int):
                    typed_val = int(new_val)
                else:
                    typed_val = new_val

                if typed_val != current:
                    setattr(Config, key, typed_val)
                    changed[key] = new_val
                    log.info(f"Config reloaded: {key} = {new_val}")
            except (ValueError, TypeError):
                log.warning(f"Config reload: could not parse {key}={new_val}")

        self._update_mtime()
        self._last_reload_time = datetime.now()
        return changed

    def get_reloadable_keys(self) -> list[str]:
        """Return list of keys that can be hot-reloaded."""
        return sorted(RELOADABLE_KEYS)

    @property
    def last_reload_time(self) -> Optional[datetime]:
        return self._last_reload_time
