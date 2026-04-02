"""Position reconciliation — detects drift between engine state and broker positions.

Compares the engine's internal trailing stop registry with the broker's actual
positions and flags mismatches: orphaned stops (engine tracking a position the
broker doesn't have), untracked positions (broker has a position the engine
doesn't know about), and entry price drift.

Runs periodically inside the engine loop, gated by POSITION_RECONCILIATION_ENABLED.
"""

from dataclasses import dataclass, field

from config import Config
from utils.logger import log


@dataclass
class Mismatch:
    symbol: str
    issue: str
    detail: str = ""


@dataclass
class ReconciliationResult:
    """Result of a single reconciliation run."""
    mismatches: list[Mismatch] = field(default_factory=list)
    broker_symbols: set[str] = field(default_factory=set)
    engine_symbols: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return len(self.mismatches) == 0

    @property
    def count(self) -> int:
        return len(self.mismatches)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol for comparison (strip '/' from crypto symbols)."""
    return symbol.replace("/", "")


def _build_broker_map(positions: list[dict]) -> dict[str, dict]:
    """Build a normalized-symbol -> position dict from broker positions."""
    result = {}
    for p in positions:
        norm = _normalize_symbol(p["symbol"])
        result[norm] = p
    return result


def _build_engine_map(trailing_stops: dict) -> dict[str, object]:
    """Build a normalized-symbol -> TrailingStop dict from engine state."""
    result = {}
    for symbol, stop in trailing_stops.items():
        norm = _normalize_symbol(symbol)
        result[norm] = stop
    return result


class PositionReconciler:
    """Compares engine trailing-stop state with broker positions."""

    def __init__(self, entry_price_tolerance: float = 0.02):
        """
        Args:
            entry_price_tolerance: Max allowed relative difference between
                engine entry_price and broker avg_entry_price (default 2%).
        """
        self.entry_price_tolerance = entry_price_tolerance
        self._last_result: ReconciliationResult | None = None

    def reconcile(self, trailing_stops: dict, broker_positions: list[dict],
                  tracked_symbols: list[str] | None = None) -> ReconciliationResult:
        """Run reconciliation between engine state and broker positions.

        Args:
            trailing_stops: Engine's risk manager trailing_stops dict.
            broker_positions: Result of broker.get_positions().
            tracked_symbols: Optional list of symbols the engine is configured
                to trade (Config.SYMBOLS). If provided, untracked positions
                outside this set are flagged as informational only.

        Returns:
            ReconciliationResult with any mismatches found.
        """
        result = ReconciliationResult()

        broker_map = _build_broker_map(broker_positions)
        engine_map = _build_engine_map(trailing_stops)

        result.broker_symbols = set(broker_map.keys())
        result.engine_symbols = set(engine_map.keys())

        # Normalize tracked symbols for comparison
        tracked_norm = None
        if tracked_symbols:
            tracked_norm = {_normalize_symbol(s) for s in tracked_symbols}

        # 1. Orphaned stops: engine tracks a position the broker doesn't have
        for norm_sym, stop in engine_map.items():
            if norm_sym not in broker_map:
                result.mismatches.append(Mismatch(
                    symbol=stop.symbol,
                    issue="orphaned_stop",
                    detail=f"Engine has trailing stop (entry ${stop.entry_price:.2f}) "
                           f"but broker has no position",
                ))

        # 2. Untracked positions: broker has a position the engine doesn't track
        for norm_sym, pos in broker_map.items():
            if norm_sym not in engine_map:
                # If we know the symbol list, distinguish between
                # "our symbol but untracked" vs "external position"
                if tracked_norm and norm_sym not in tracked_norm:
                    issue = "external_position"
                    detail = (f"Broker has position in {pos['symbol']} "
                              f"(qty: {pos['qty']}, value: ${pos['market_value']:.2f}) "
                              f"which is not in the engine's symbol list")
                else:
                    issue = "untracked_position"
                    detail = (f"Broker has position in {pos['symbol']} "
                              f"(qty: {pos['qty']}, value: ${pos['market_value']:.2f}) "
                              f"but engine has no trailing stop registered")
                result.mismatches.append(Mismatch(
                    symbol=pos["symbol"],
                    issue=issue,
                    detail=detail,
                ))

        # 3. Entry price drift: both have the position but entry prices diverge
        for norm_sym in engine_map.keys() & broker_map.keys():
            stop = engine_map[norm_sym]
            pos = broker_map[norm_sym]

            engine_entry = stop.entry_price
            broker_entry = pos.get("avg_entry_price", 0)

            if broker_entry > 0 and engine_entry > 0:
                drift = abs(engine_entry - broker_entry) / broker_entry
                if drift > self.entry_price_tolerance:
                    result.mismatches.append(Mismatch(
                        symbol=stop.symbol,
                        issue="entry_price_drift",
                        detail=(f"Engine entry ${engine_entry:.2f} vs broker "
                                f"avg entry ${broker_entry:.2f} "
                                f"(drift: {drift:.1%})"),
                    ))

        self._last_result = result
        return result

    def auto_fix_orphaned_stops(self, trailing_stops: dict,
                                result: ReconciliationResult) -> list[str]:
        """Remove orphaned trailing stops that have no broker position.

        Returns list of symbols that were cleaned up.
        """
        cleaned = []
        for m in result.mismatches:
            if m.issue == "orphaned_stop":
                norm = _normalize_symbol(m.symbol)
                # Find the original key in trailing_stops
                for key in list(trailing_stops.keys()):
                    if _normalize_symbol(key) == norm:
                        del trailing_stops[key]
                        cleaned.append(m.symbol)
                        log.info(f"Reconciler: removed orphaned stop for {m.symbol}")
                        break
        return cleaned

    @property
    def last_result(self) -> ReconciliationResult | None:
        return self._last_result
