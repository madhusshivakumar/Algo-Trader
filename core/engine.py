"""Main trading engine — handles both crypto (24/7) and equities (market hours)."""

import signal
import time
import traceback
from datetime import datetime

from core.broker import Broker
from core.risk_manager import RiskManager
from strategies.router import compute_signals as route_signals, get_strategy, get_strategy_key
from core.trade_explainer import explain as explain_trade
from config import Config
from utils.logger import log


class TradingEngine:
    # Minimum seconds between trades on the same symbol
    COOLDOWN_CRYPTO = 2700  # 45 minutes
    COOLDOWN_EQUITY = 300   # 5 minutes

    def __init__(self):
        self.broker = Broker()
        self.risk = RiskManager()
        self.cycle_count = 0
        self._shutdown_requested = False
        self._reload_requested = False
        # Track which equity positions were opened today (PDT protection)
        self._equity_buys_today: dict[str, datetime] = {}
        # Cooldown: last trade timestamp per symbol
        self._last_trade_time: dict[str, float] = {}
        # Daily trade counter (reset on daily reset)
        self._daily_trade_count: dict[str, int] = {}  # "crypto" / "equity" → count
        self._daily_trade_date: str = ""

        # Sprint 5: consecutive broker failures (alert after 3 in a row)
        self._consecutive_broker_failures: dict[str, int] = {}  # method → count
        # Sprint 5: track one-shot alert signals so we don't spam
        self._alerted_degraded: set[str] = set()           # symbols degraded-alerted
        self._alerted_external_positions: set[str] = set() # symbols external-alerted
        self._alerted_drawdown_halt: bool = False          # daily DD halt alert sent?
        self._alerted_intraday_halt: bool = False          # intraday halt alert sent?

        # v3: State persistence (optional)
        self.state_store = None
        if Config.STATE_PERSISTENCE_ENABLED:
            from core.state_store import StateStore
            self.state_store = StateStore()

        # v3: Order management (optional)
        self.order_manager = None
        if Config.ORDER_MANAGEMENT_ENABLED:
            from core.order_manager import OrderManager
            self.order_manager = OrderManager(self.broker, self.state_store)

        # v3: Parallel data fetching (optional)
        self.data_fetcher = None
        if Config.PARALLEL_FETCH_ENABLED:
            from core.data_fetcher import DataFetcher
            self.data_fetcher = DataFetcher(self.broker, Config.FETCH_WORKERS)

        # v3: Config hot-reload (optional)
        self.config_reloader = None
        if Config.HOT_RELOAD_ENABLED:
            from core.config_reloader import ConfigReloader
            self.config_reloader = ConfigReloader()

        # v3: Drift detection (optional)
        self.drift_detector = None
        if Config.DRIFT_DETECTION_ENABLED:
            from core.drift_detector import DriftDetector
            self.drift_detector = DriftDetector(
                lookback_days=Config.DRIFT_LOOKBACK_DAYS,
                min_trades=Config.DRIFT_MIN_TRADES,
            )

        # v3: Position reconciliation (optional)
        self.position_reconciler = None
        if Config.POSITION_RECONCILIATION_ENABLED:
            from core.position_reconciler import PositionReconciler
            self.position_reconciler = PositionReconciler(
                entry_price_tolerance=Config.RECONCILIATION_ENTRY_TOLERANCE,
            )

        # v3: Transaction cost modeling (optional)
        self.cost_model = None
        if Config.TC_ENABLED:
            from core.transaction_costs import TransactionCostModel
            self.cost_model = TransactionCostModel()

        # v3: DB rotation (optional)
        self.db_rotator = None
        if Config.DB_ROTATION_ENABLED:
            from utils.db_rotation import DBRotator
            from utils.logger import DB_PATH
            self.db_rotator = DBRotator(DB_PATH)

        # v3: VWAP/TWAP execution (optional)
        self.execution_manager = None
        if Config.VWAP_TWAP_ENABLED:
            from core.execution_algo import ExecutionAlgoManager
            self.execution_manager = ExecutionAlgoManager()

        # v3: Portfolio optimization (optional)
        self.portfolio_optimizer = None
        if Config.PORTFOLIO_OPTIMIZATION_ENABLED:
            from core.portfolio_optimizer import PortfolioOptimizer
            self.portfolio_optimizer = PortfolioOptimizer()

        # Alerting (optional)
        self.alert_manager = None
        if Config.ALERTING_ENABLED:
            from core.alerting import AlertManager
            self.alert_manager = AlertManager()

        # Sprint 6C: regime detector — classifies current market vol regime from
        # SPY's realized vol. Updated once per hour during _run_cycle; gates
        # strategy selection in `route_signals` via the `regime` kwarg.
        from core.regime_detector import RegimeDetector
        self.regime_detector = RegimeDetector(
            window=Config.REGIME_VOL_WINDOW,
            max_history=Config.REGIME_HISTORY_MAX,
        )
        self._last_regime_update_ts: float = 0.0

    def _check_db_rotation(self):
        """Check if the trades DB needs rotation and rotate if so."""
        if not self.db_rotator:
            return
        try:
            if self.db_rotator.should_rotate(
                max_rows=Config.DB_ROTATION_MAX_ROWS,
                max_age_days=Config.DB_ROTATION_MAX_AGE_DAYS,
            ):
                self.db_rotator.rotate(keep_recent_days=Config.DB_ROTATION_KEEP_DAYS)
        except Exception as e:
            log.error(f"DB rotation error: {e}")

    def _maybe_update_regime(self):
        """Sprint 6C: refresh the market regime classification.

        Fetches SPY daily bars at most once per hour (wall-clock). SPY-based
        realized vol doesn't swing intraday, so tighter polling is wasted
        broker calls. Broker failures are swallowed — regime stays stale but
        trading continues.
        """
        detector = getattr(self, "regime_detector", None)
        if detector is None:
            return
        now = time.time()
        last = getattr(self, "_last_regime_update_ts", 0.0)
        if now - last < Config.REGIME_UPDATE_INTERVAL_SEC:
            return
        try:
            # SPY is the canonical vol benchmark. We ask for ~40 days of daily
            # bars so a 20-day rolling std has headroom.
            spy_df = self.broker.get_historical_bars(
                "SPY", days=Config.REGIME_VOL_WINDOW * 2, timeframe="1Day"
            )
            if spy_df is None or len(spy_df) < 2:
                return
            snapshot = detector.update(spy_df)
            self._last_regime_update_ts = now
            if snapshot is not None and self.cycle_count % 60 == 0:
                log.info(
                    f"Regime: {snapshot.regime} "
                    f"(SPY 20d vol {snapshot.annualized_vol:.1%})"
                )
        except Exception as e:
            # Never let a regime-detection hiccup break a cycle.
            log.warning(f"Regime update failed: {e}")

    def _load_persisted_state(self):
        """Restore runtime state from SQLite (if state persistence enabled)."""
        if not self.state_store:
            return
        try:
            state = self.state_store.load_engine_state()

            # Restore cooldowns
            self._last_trade_time = state.get("cooldowns", {})

            # Restore PDT buy records
            for symbol, dt_str in state.get("pdt_buys", {}).items():
                try:
                    self._equity_buys_today[symbol] = datetime.fromisoformat(dt_str)
                except (ValueError, TypeError):
                    pass

            # Restore trailing stops
            from core.risk_manager import TrailingStop
            for symbol, data in state.get("trailing_stops", {}).items():
                self.risk.trailing_stops[symbol] = TrailingStop(
                    symbol=symbol,
                    entry_price=data["entry_price"],
                    highest_price=data["highest_price"],
                    stop_pct=data["stop_pct"],
                    entry_time=data.get("entry_time", 0.0),
                )

            # Restore scalars
            scalars = state.get("scalars", {})
            if scalars.get("halted"):
                self.risk.halted = True
                self.risk.halt_reason = scalars.get("halt_reason", "")
            if scalars.get("daily_start_equity"):
                self.risk.daily_start_equity = scalars["daily_start_equity"]
            if scalars.get("cycle_count"):
                self.cycle_count = scalars["cycle_count"]

            log.info(f"Restored state: {len(self._last_trade_time)} cooldowns, "
                     f"{len(self.risk.trailing_stops)} trailing stops, "
                     f"{len(self._equity_buys_today)} PDT records")
        except Exception as e:
            log.warning(f"Could not restore persisted state: {e}")

    def _save_persisted_state(self):
        """Save runtime state to SQLite (if state persistence enabled)."""
        if not self.state_store:
            return
        try:
            trailing_stops = {
                sym: {
                    "entry_price": stop.entry_price,
                    "highest_price": stop.highest_price,
                    "stop_pct": stop.stop_pct,
                    "entry_time": stop.entry_time,
                }
                for sym, stop in self.risk.trailing_stops.items()
            }
            pdt_buys = {
                sym: dt.isoformat()
                for sym, dt in self._equity_buys_today.items()
            }
            scalars = {
                "halted": self.risk.halted,
                "halt_reason": self.risk.halt_reason,
                "daily_start_equity": self.risk.daily_start_equity,
                "cycle_count": self.cycle_count,
            }
            self.state_store.save_engine_state(
                trailing_stops, self._last_trade_time, pdt_buys, scalars
            )
        except Exception as e:
            log.warning(f"Could not save state: {e}")

    def initialize(self):
        """Set up initial state."""
        account = self.broker.get_account()
        log.info(f"Account equity: ${account['equity']:.2f} | Cash: ${account['cash']:.2f}")
        self.risk.initialize(account["equity"])

        # Resolve capital-tier profile from account equity (see core/user_profile.py)
        profile = Config.set_profile_from_equity(float(account["equity"]))
        log.info(
            f"User profile: {profile.name.upper()} — "
            f"equity sizing {profile.base_position_pct_equity:.1%}, "
            f"crypto sizing {profile.base_position_pct_crypto:.1%}, "
            f"Kelly {profile.kelly_fraction:.2f}, "
            f"daily loss halt ${profile.max_daily_loss_usd(float(account['equity'])):,.0f}"
        )

        # Restore persisted state (must come after risk.initialize)
        self._load_persisted_state()

        # Detect orphan positions (broker positions without trailing stops)
        self._detect_orphan_positions()

        # v3: Check DB rotation on startup
        self._check_db_rotation()

        if Config.is_paper():
            log.warning("Running in PAPER trading mode")
        else:
            log.warning("Running in LIVE trading mode — real money!")
            # Sprint 5G: alert when paper→live override is used
            if (getattr(Config, "_LIVE_GATE_OVERRIDE_USED", False)
                    and self.alert_manager):
                self.alert_manager.live_mode_override()

        log.info(f"Crypto symbols: {', '.join(Config.CRYPTO_SYMBOLS)}")
        log.info(f"Equity symbols: {', '.join(Config.EQUITY_SYMBOLS)}")
        log.info(f"Max position: {Config.MAX_POSITION_PCT:.0%} | Stop-loss: {Config.STOP_LOSS_PCT:.1%}")
        log.info(f"Daily drawdown limit: {Config.DAILY_DRAWDOWN_LIMIT:.0%}")

        if Config.PDT_PROTECTION:
            log.info(f"PDT protection: ON (day trades used: {account['daytrade_count']}/3)")

        log.info("Strategy assignments:")
        for sym in Config.CRYPTO_SYMBOLS + Config.EQUITY_SYMBOLS:
            name, _ = get_strategy(sym)
            log.info(f"  {sym:8s} → {name}")

    def run_cycle(self):
        """Run one trading cycle for all symbols."""
        self.cycle_count += 1
        self._cached_position_dfs = None  # Clear correlation cache each cycle
        try:
            account = self.broker.get_account()
            self._record_broker_success("get_account")
        except Exception as e:
            self._record_broker_failure("get_account", e)
            # Can't run a cycle without equity — skip this tick
            return
        equity = account["equity"]

        # v3: Poll pending orders for status updates
        if self.order_manager:
            try:
                newly_terminal = self.order_manager.poll_pending_orders()
                self._handle_filled_orders(newly_terminal)
            except Exception as e:
                log.error(f"Error polling orders: {e}")

        # v3: Cancel stale orders every 5 cycles
        if self.order_manager and self.cycle_count % 5 == 0:
            try:
                self.order_manager.cancel_stale_orders(Config.ORDER_STALE_TIMEOUT_SECONDS)
            except Exception as e:
                log.error(f"Error canceling stale orders: {e}")

        # v3: VWAP/TWAP execution — dispatch due child orders and poll fills
        if self.execution_manager:
            try:
                self.execution_manager.tick(self.broker)
                completed = self.execution_manager.poll_children(self.broker)
                self._handle_completed_executions(completed)
                # Clean up old plans every 100 cycles
                if self.cycle_count % 100 == 0:
                    self.execution_manager.cleanup_completed()
            except Exception as e:
                log.error(f"Execution algo error: {e}")

        # v3: Config hot-reload every 30 cycles
        if self.config_reloader and self.cycle_count % 30 == 0:
            try:
                if self.config_reloader.check_for_changes():
                    changed = self.config_reloader.reload()
                    if changed:
                        log.info(f"Config reloaded: {', '.join(changed.keys())}")
            except Exception as e:
                log.error(f"Config reload error: {e}")

        # v3: Portfolio optimization recompute
        if (self.portfolio_optimizer and Config.MEAN_VARIANCE_ENABLED
                and self.portfolio_optimizer.should_recompute(self.cycle_count)):
            try:
                self._recompute_allocation(equity)
            except Exception as e:
                log.error(f"Portfolio optimization error: {e}")

        # Sprint 6C: regime detection — refresh at most once per hour. SPY daily
        # vol doesn't change intraday enough to warrant tighter polling, and
        # every update costs one broker call.
        self._maybe_update_regime()

        # v3: Drift detection every 100 cycles
        if self.drift_detector and self.cycle_count % 100 == 0:
            try:
                alerts = self.drift_detector.should_alert()
                for alert in alerts:
                    log.warning(alert)
            except Exception as e:
                log.error(f"Drift detection error: {e}")

        # v3: Position reconciliation
        if (self.position_reconciler and
                self.cycle_count % Config.RECONCILIATION_INTERVAL_CYCLES == 0):
            self._run_reconciliation()

        # v3: Save state every cycle
        self._save_persisted_state()

        # Snapshot equity every 10 cycles
        if self.cycle_count % 10 == 0:
            log.snapshot(equity, account["cash"])

        # v3: DB rotation check every 1000 cycles
        if self.db_rotator and self.cycle_count % 1000 == 0:
            self._check_db_rotation()

        # Sprint 5D: Intraday $ P&L circuit breaker (profile-aware).
        # Runs BEFORE the percentage-based check so tight $ floors for small
        # accounts catch losses earlier than the 10% equity drawdown.
        profile = getattr(Config, "_PROFILE", None)
        if profile is not None:
            unrealized = float(account.get("unrealized_pl", 0.0) or 0.0)
            threshold = profile.max_daily_loss_usd(self.risk.daily_start_equity
                                                   or equity)
            if self.risk.check_intraday_halt(equity, threshold, unrealized):
                # Alert once per halt transition
                if (self.alert_manager
                        and not getattr(self, "_alerted_intraday_halt", False)):
                    realized = (equity - unrealized) - self.risk.daily_start_equity
                    self.alert_manager.intraday_halt(realized, unrealized, threshold)
                    self._alerted_intraday_halt = True
                if self.cycle_count % 60 == 0:
                    log.warning(f"Trading halted: {self.risk.halt_reason}")
                return

        # Check risk limits
        if not self.risk.can_trade(equity):
            if self.cycle_count % 60 == 0:
                log.warning(f"Trading halted: {self.risk.halt_reason}")
            # Alert exactly once per halt transition (not every 60 cycles)
            if (self.alert_manager and self.risk.halted
                    and not getattr(self, "_alerted_drawdown_halt", False)):
                self.alert_manager.drawdown_halt(
                    self.risk.daily_drawdown, Config.DAILY_DRAWDOWN_LIMIT)
                self._alerted_drawdown_halt = True
            return
        else:
            # Reset the alert flags when we're no longer halted (e.g. after daily reset)
            if getattr(self, "_alerted_drawdown_halt", False) and not self.risk.halted:
                self._alerted_drawdown_halt = False
            if getattr(self, "_alerted_intraday_halt", False) and not self.risk.halted:
                self._alerted_intraday_halt = False

        # Determine which symbols to process this cycle
        active_symbols = list(Config.CRYPTO_SYMBOLS)
        if Config.is_market_open():
            active_symbols += Config.EQUITY_SYMBOLS

        # v3: Pre-fetch bars in parallel when enabled
        prefetched_bars = {}
        if self.data_fetcher and active_symbols:
            try:
                prefetched_bars = self.data_fetcher.fetch_recent_bars_batch(active_symbols)
            except Exception as e:
                log.error(f"Parallel fetch failed, falling back to sequential: {e}")

        # ── Process all active symbols ───────────────────────────────
        for symbol in active_symbols:
            try:
                if symbol in prefetched_bars:
                    self._process_symbol_with_data(symbol, equity, prefetched_bars[symbol])
                else:
                    self._process_symbol(symbol, equity)
            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")
                if self.cycle_count <= 3:
                    traceback.print_exc()

        if not Config.is_market_open() and self.cycle_count % 60 == 0:
            now = datetime.now(Config.MARKET_TZ)
            log.info(f"Market closed ({now.strftime('%I:%M %p ET')}) — skipping equities")

        # Sprint 8 (post-incident Apr 21): liveness heartbeat.
        # The watchdog reads this file; if ts is stale during market hours
        # the engine is zombied inside a "healthy" container — alert.
        try:
            from core.heartbeat import write_heartbeat
            last_trade_ts = self._query_last_trade_ts() if hasattr(
                self, "_query_last_trade_ts") else None
            write_heartbeat(
                cycle_count=self.cycle_count,
                positions_evaluated=len(active_symbols),
                signals_produced=0,  # room to plumb if needed later
                last_trade_ts=last_trade_ts,
                equity=equity,
                halted=self.risk.halted,
            )
        except Exception as e:
            # Never let observability break trading
            if self.cycle_count % 60 == 0:
                log.warning(f"Heartbeat write skipped: {e}")

    def _query_last_trade_ts(self) -> str | None:
        """Most recent trade timestamp from trades.db; None if DB unavailable."""
        try:
            import sqlite3
            from utils.logger import DB_PATH
            conn = sqlite3.connect(DB_PATH, timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM trades"
                ).fetchone()
            finally:
                conn.close()
            return row[0] if row and row[0] else None
        except Exception:
            return None

    def _get_total_exposure(self) -> float:
        """Get total dollar value of all open positions.

        On broker API failure, logs and tracks consecutive failures (alert at 3),
        returns 0.0 so that the caller can proceed without exposure data.
        """
        try:
            positions = self.broker.get_positions()
            self._record_broker_success("get_positions")
        except Exception as e:
            self._record_broker_failure("get_positions", e)
            return 0.0
        return sum(abs(float(p["market_value"])) for p in positions)

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is still in cooldown from its last trade."""
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        cooldown = self.COOLDOWN_CRYPTO if Config.is_crypto(symbol) else self.COOLDOWN_EQUITY
        elapsed = time.time() - last
        if elapsed < cooldown:
            remaining = int(cooldown - elapsed)
            if self.cycle_count % 5 == 0:
                log.info(f"  {symbol} on cooldown ({remaining}s remaining)")
            return True
        return False

    def _record_trade(self, symbol: str):
        """Record trade time for cooldown tracking."""
        self._last_trade_time[symbol] = time.time()
        # Update daily trade count
        today = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")
        if getattr(self, "_daily_trade_date", "") != today:
            self._daily_trade_count = {}
            self._daily_trade_date = today
        asset_class = "crypto" if Config.is_crypto(symbol) else "equity"
        counts = getattr(self, "_daily_trade_count", {})
        counts[asset_class] = counts.get(asset_class, 0) + 1
        self._daily_trade_count = counts

    def _record_broker_failure(self, method: str, error: Exception) -> None:
        """Track consecutive broker API failures — alert after 3 in a row.

        Uses getattr fallback so tests that skip __init__ (via TradingEngine.__new__)
        still work without the counter attribute.
        """
        counters = getattr(self, "_consecutive_broker_failures", None)
        if counters is None:
            counters = {}
            self._consecutive_broker_failures = counters
        counters[method] = counters.get(method, 0) + 1
        count = counters[method]
        log.warning(f"Broker API failure #{count} on {method}: {error}")
        if count >= 3 and getattr(self, "alert_manager", None):
            self.alert_manager.broker_api_failure(method, count, str(error))

    def _record_broker_success(self, method: str) -> None:
        """Reset the consecutive-failure counter for a broker method."""
        counters = getattr(self, "_consecutive_broker_failures", None)
        if counters and method in counters:
            counters[method] = 0

    def _is_daily_trade_limit_reached(self, symbol: str) -> bool:
        """Check if we've hit the daily trade limit for this asset class.

        When a UserProfile is active (set by engine.initialize()), uses the
        profile's per-tier limits. Otherwise falls back to the legacy
        Config.MAX_TRADES_PER_DAY_* values (preserves test compatibility).
        """
        today = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")
        if getattr(self, "_daily_trade_date", "") != today:
            self._daily_trade_count = {}
            self._daily_trade_date = today
            return False
        is_crypto = Config.is_crypto(symbol)
        asset_class = "crypto" if is_crypto else "equity"
        profile = getattr(Config, "_PROFILE", None)
        if profile is not None:
            limit = (profile.max_trades_per_day_crypto if is_crypto
                     else profile.max_trades_per_day_equity)
        else:
            limit = (Config.MAX_TRADES_PER_DAY_CRYPTO if is_crypto
                     else Config.MAX_TRADES_PER_DAY_EQUITY)
        count = getattr(self, "_daily_trade_count", {}).get(asset_class, 0)
        return count >= limit

    def _process_symbol_with_data(self, symbol: str, equity: float, df):
        """Process a symbol with pre-fetched data (used by parallel fetcher)."""
        if df is None or df.empty or len(df) < 30:
            return
        self._process_symbol_inner(symbol, equity, df)

    def _process_symbol(self, symbol: str, equity: float):
        """Evaluate signals and act for one symbol (fetches its own data)."""
        if self._is_on_cooldown(symbol):
            return
        # Get market data
        df = self.broker.get_recent_bars(symbol, limit=100)
        if df.empty or len(df) < 30:
            return
        self._process_symbol_inner(symbol, equity, df)

    def _process_symbol_inner(self, symbol: str, equity: float, df):
        """Core processing logic for a single symbol (shared by both paths)."""
        import math
        is_equity = not Config.is_crypto(symbol)

        # Skip if symbol is on cooldown
        if self._is_on_cooldown(symbol):
            return

        current_price = float(df["close"].iloc[-1])

        # Guard against NaN/Inf prices (bad data from broker)
        if math.isnan(current_price) or math.isinf(current_price) or current_price <= 0:
            log.warning(f"  {symbol}: Invalid price {current_price}, skipping")
            return

        # Check trailing stops for existing positions
        position = self.broker.get_position(symbol)
        if position:
            if self.risk.should_stop_loss(symbol, current_price):
                # PDT check: don't sell equity same day we bought it
                if is_equity and self._is_same_day_buy(symbol):
                    log.warning(f"Trailing stop hit for {symbol} but PDT protection blocks same-day sell")
                    return

                log.info(f"Closing {symbol} — trailing stop hit")
                # Capture entry price BEFORE unregister removes it
                stop_entry = self.risk.trailing_stops.get(symbol)
                entry_price = stop_entry.entry_price if stop_entry else current_price
                result = self.broker.close_position(symbol)
                if result:
                    pnl = position["unrealized_pl"]
                    log.trade(symbol, "sell", position["market_value"], current_price,
                              "trailing stop", pnl, strategy=get_strategy_key(symbol))
                    self.risk.unregister(symbol)
                    self._clear_buy_record(symbol)
                    if self.alert_manager:
                        market_val = abs(position["market_value"]) if position["market_value"] else 0
                        loss_pct = abs(pnl / market_val) if market_val > 0 else 0
                        self.alert_manager.stop_loss_hit(symbol, entry_price, current_price, loss_pct)
                return

        # Pre-earnings position closure (optional — closes equity positions in blackout window)
        if (Config.EARNINGS_CALENDAR_ENABLED and Config.EARNINGS_CLOSE_POSITIONS
                and position and is_equity):
            from core.signal_modifiers import is_in_earnings_blackout
            if is_in_earnings_blackout(symbol):
                if self._is_same_day_buy(symbol):
                    log.warning(f"{symbol} in earnings blackout but PDT blocks same-day sell")
                else:
                    log.info(f"Closing {symbol} — earnings blackout (pre-earnings risk reduction)")
                    result = self.broker.close_position(symbol)
                    if result:
                        pnl = position["unrealized_pl"]
                        log.trade(symbol, "sell", position["market_value"], current_price,
                                  "earnings blackout close", pnl, strategy=get_strategy_key(symbol))
                        self.risk.unregister(symbol)
                        # No cooldown for earnings close — it's defensive, not a trade signal
                        self._clear_buy_record(symbol)
                        if self.alert_manager and Config.ALERT_ON_TRADE:
                            self.alert_manager.trade_executed(
                                symbol, "sell", position["market_value"],
                                current_price, "earnings blackout close")
                    return

        # Max hold time check
        if position and self.risk.is_max_hold_exceeded(symbol):
            if is_equity and self._is_same_day_buy(symbol):
                log.warning(f"Max hold exceeded for {symbol} but PDT blocks same-day sell")
            else:
                log.info(f"Closing {symbol} — max hold time exceeded")
                result = self.broker.close_position(symbol)
                if result:
                    pnl = float(position.get("unrealized_pl", 0))
                    strategy_key = get_strategy_key(symbol)
                    log.trade(symbol, "sell", abs(float(position["market_value"])),
                              current_price, "max_hold_exceeded", pnl,
                              strategy=strategy_key)
                    self.risk.unregister(symbol)
                return

        # Compute strategy signals (routed per symbol)
        # Sprint 6C: pass current regime so the router can skip strategies that
        # are known to misfire in this vol environment. Fall back to 'normal'
        # for test engines constructed via __new__ that skip __init__.
        detector = getattr(self, "regime_detector", None)
        regime = detector.get_current_regime() if detector is not None else "normal"
        signal = route_signals(symbol, df, regime=regime)

        if signal["action"] == "buy" and not position and not (
                self.execution_manager and self.execution_manager.get_active_plans(symbol)):
            # ── Min signal strength gate ──
            if signal.get("strength", 0) < Config.MIN_SIGNAL_STRENGTH:
                return

            # ── Daily trade limit ──
            if self._is_daily_trade_limit_reached(symbol):
                if self.cycle_count % 10 == 0:
                    log.info(f"  {symbol}: Skipping buy — daily trade limit reached")
                return

            # ── Duplicate order guard ──
            if self.order_manager and self.order_manager.get_active_orders(symbol):
                if self.cycle_count % 10 == 0:
                    log.info(f"  {symbol}: Skipping buy — pending order already exists")
                return

            # ── Check total exposure cap (90% of equity) ──
            total_exposure = self._get_total_exposure()
            max_total_exposure = equity * 0.90
            remaining_budget = max_total_exposure - total_exposure

            if remaining_budget < 100:
                if self.cycle_count % 10 == 0:
                    log.info(f"  {symbol}: Skipping buy — total exposure ${total_exposure:.0f} "
                             f"near cap ${max_total_exposure:.0f}")
                return

            # v3: Correlation check (skip buy if too correlated with existing positions)
            if Config.CORRELATION_CHECK_ENABLED:
                try:
                    existing_dfs = self._get_existing_position_dfs()
                    if not self.risk.check_correlation(df, existing_dfs, Config.CORRELATION_THRESHOLD):
                        return
                except Exception:
                    pass

            # Sprint 5C: Pre-trade liquidity gate — skip if spread too wide.
            # A beginner with $800 can lose ~$10 of a $50 position to spread if they
            # market-buy a thinly-traded name — the gate prevents that.
            if Config.LIQUIDITY_GATE_ENABLED:
                try:
                    quote = self.broker.get_latest_quote(symbol)
                    if quote is not None:
                        limit_bps = (Config.MAX_SPREAD_BPS_CRYPTO if Config.is_crypto(symbol)
                                     else Config.MAX_SPREAD_BPS_EQUITY)
                        if quote["spread_bps"] > limit_bps:
                            if self.cycle_count % 10 == 0:
                                log.info(
                                    f"  {symbol}: Skipping buy — spread "
                                    f"{quote['spread_bps']:.1f} bps > {limit_bps:.0f} bps limit"
                                )
                            if self.alert_manager:
                                self.alert_manager.liquidity_skip(
                                    symbol, quote["spread_bps"], limit_bps
                                )
                            return
                except Exception as e:
                    # Never let quote-fetch failure block a trade — log and continue
                    log.warning(f"  {symbol}: Liquidity gate check failed: {e}")

            # Capital-tier-aware position sizing (see core/user_profile.py)
            # Fall back to legacy hardcoded sizing if profile not initialized (tests).
            profile = getattr(Config, "_PROFILE", None)
            if profile is not None:
                base_pct = (profile.base_position_pct_equity if is_equity
                            else profile.base_position_pct_crypto)
                kelly_fraction = profile.kelly_fraction
                max_pos_pct = profile.max_single_position_pct
            else:
                base_pct = 0.15 if is_equity else 0.20
                kelly_fraction = None  # use Config.KELLY_FRACTION default
                max_pos_pct = Config.MAX_SINGLE_POSITION_PCT

            # Sprint 5E: degraded symbols get a 70% size cut.
            # The full RL + signal pipeline still runs; we only reduce the bet size
            # because past performance flagged the strategy as drifting.
            if self.drift_detector and self.drift_detector.is_degraded(symbol):
                base_pct *= 0.3
                # Alert once per (symbol, degradation event)
                alerted = getattr(self, "_alerted_degraded", None)
                if alerted is None:
                    alerted = set()
                    self._alerted_degraded = alerted
                if symbol not in alerted:
                    metrics = self.drift_detector.get_recent_metrics(symbol)
                    if self.alert_manager and metrics:
                        # Use win rate * avg_pnl as a Sharpe-ish proxy (we don't
                        # store true Sharpe per-symbol)
                        proxy = metrics.win_rate * metrics.avg_pnl
                        self.alert_manager.symbol_degraded(
                            symbol, recent_sharpe=proxy,
                            lookback_days=self.drift_detector.lookback_days,
                        )
                    alerted.add(symbol)
                if self.cycle_count % 10 == 0:
                    log.info(f"  {symbol}: degraded — size cut to 30%")

            # v3: Portfolio optimization — Kelly or mean-variance override
            if self.portfolio_optimizer:
                if Config.KELLY_SIZING_ENABLED:
                    kelly_size = self.portfolio_optimizer.get_kelly_size(
                        symbol, equity, base_pct,
                        kelly_fraction=kelly_fraction,
                        max_pct=max_pos_pct)
                    base_pct = kelly_size / equity if equity > 0 else base_pct
                elif Config.MEAN_VARIANCE_ENABLED:
                    base_pct = self.portfolio_optimizer.get_position_pct(
                        symbol, base_pct)

            # Cap at profile's (or legacy) max_single_position_pct
            base_pct = min(base_pct, max_pos_pct)

            # v3: Volatility-adjusted sizing
            if Config.VOLATILITY_SIZING_ENABLED:
                max_size = self.risk.calculate_volatility_adjusted_size(equity, df, base_pct)
            else:
                max_size = equity * base_pct

            # Don't exceed remaining budget
            max_size = min(max_size, remaining_budget)

            size = max_size * signal["strength"]
            size = max(size, 1.0)

            # Sprint 5F: VaR-aware position cap (profile-driven).
            # Estimates the symbol's 95% one-bar VaR from recent returns, then caps
            # `size` so that worst-case loss <= profile.max_var_contribution_pct × equity.
            # Beginner: 1% of equity per trade; Hobbyist 2%; Learner 3%.
            if profile is not None:
                var_pct = self.risk.estimate_var_pct(df, confidence=0.95)
                if var_pct > 0:
                    max_position_at_var = (
                        profile.max_var_contribution_pct * equity / var_pct
                    )
                    if size > max_position_at_var:
                        if self.cycle_count % 10 == 0:
                            log.info(
                                f"  {symbol}: Capping size from ${size:.0f} to "
                                f"${max_position_at_var:.0f} (VaR cap: position × "
                                f"{var_pct:.2%} <= {profile.max_var_contribution_pct:.1%} equity)"
                            )
                        size = max(max_position_at_var, 1.0)

            # v3: Transaction cost adjustment — reduce size to account for costs
            if self.cost_model:
                size = self.cost_model.net_buy_amount(symbol, size)
                if size < 1.0:
                    return

            # v3: Buying power pre-check
            if self.order_manager:
                try:
                    buying_power = self.broker.check_buying_power()
                    if size > buying_power:
                        if self.cycle_count % 10 == 0:
                            log.info(f"  {symbol}: Skipping buy — size ${size:.0f} > buying power ${buying_power:.0f}")
                        return
                except Exception:
                    pass  # Fall through — broker will reject if insufficient

            log.info(f"BUY signal for {symbol}: {signal['reason']} (strength: {signal['strength']:.2f})")

            # v3: Route through VWAP/TWAP execution when enabled and above min notional
            if self.execution_manager and size >= Config.VWAP_TWAP_MIN_NOTIONAL:
                # Skip if an execution is already active for this symbol
                if self.execution_manager.get_active_plans(symbol):
                    return

                volume_profile = None
                if Config.VWAP_TWAP_ALGO == "vwap":
                    try:
                        from core.execution_algo import build_volume_profile
                        volume_profile = build_volume_profile(
                            self.broker, symbol, Config.VWAP_VOLUME_LOOKBACK_DAYS)
                    except Exception:
                        pass  # Falls back to uniform weights

                plan = self.execution_manager.create_plan(
                    symbol=symbol, side="buy", total_notional=size,
                    algo=Config.VWAP_TWAP_ALGO, current_price=current_price,
                    volume_profile=volume_profile,
                )
                log.info(f"  {symbol}: Created {Config.VWAP_TWAP_ALGO.upper()} execution plan "
                         f"({len(plan.children)} slices, ${size:.0f} total)")
                self._record_trade(symbol)
                if is_equity:
                    self._record_buy(symbol)
            # v3: Route through OrderManager when enabled
            elif self.order_manager:
                order = self.order_manager.submit_market_buy(symbol, size, current_price)
                if order and order.state.value == "filled":
                    # Compute and log slippage for immediately filled orders
                    # (poll_pending_orders computes it for deferred fills, but
                    #  immediate fills skip polling so we compute here)
                    if order.expected_price and order.filled_avg_price:
                        from core.order_manager import OrderManager as OM
                        slippage = OM.compute_slippage(order)
                        log.log_slippage(symbol, order.expected_price,
                                         order.filled_avg_price, slippage)
                    # Sprint 6G: plain-language explanation for the dashboard.
                    explanation = explain_trade(signal, df, symbol, size)
                    log.trade(symbol, "buy", size, current_price, signal["reason"],
                              strategy=signal.get("strategy", ""),
                              regime=regime, explanation=explanation)
                    self.risk.register_entry(symbol, current_price, df)
                    self._record_trade(symbol)
                    if is_equity:
                        self._record_buy(symbol)
                    if self.alert_manager and Config.ALERT_ON_TRADE:
                        self.alert_manager.trade_executed(symbol, "buy", size, current_price, signal.get("reason", ""))
                elif order and order.state.value == "submitted":
                    # Order submitted but not yet filled — will be handled by poll_pending_orders
                    log.info(f"  {symbol}: Market buy submitted (order {order.order_id[:8]}), awaiting fill")
            else:
                result = self.broker.buy(symbol, size)
                if result:
                    explanation = explain_trade(signal, df, symbol, size)
                    log.trade(symbol, "buy", size, current_price, signal["reason"],
                              strategy=signal.get("strategy", ""),
                              regime=regime, explanation=explanation)
                    self.risk.register_entry(symbol, current_price, df)
                    self._record_trade(symbol)
                    if is_equity:
                        self._record_buy(symbol)
                    if self.alert_manager and Config.ALERT_ON_TRADE:
                        self.alert_manager.trade_executed(symbol, "buy", size, current_price, signal.get("reason", ""))

        elif signal["action"] == "sell" and position:
            # PDT check
            if is_equity and self._is_same_day_buy(symbol):
                log.warning(f"SELL signal for {symbol} but PDT protection blocks same-day sell — holding")
                return

            log.info(f"SELL signal for {symbol}: {signal['reason']}")
            result = self.broker.close_position(symbol)
            if result:
                pnl = position["unrealized_pl"]
                explanation = explain_trade(
                    signal, df, symbol, float(position.get("market_value", 0) or 0)
                )
                log.trade(symbol, "sell", position["market_value"], current_price,
                          signal["reason"], pnl, strategy=signal.get("strategy", ""),
                          regime=regime, explanation=explanation)
                self.risk.unregister(symbol)
                self._record_trade(symbol)
                self._clear_buy_record(symbol)
                if self.alert_manager and Config.ALERT_ON_TRADE:
                    self.alert_manager.trade_executed(symbol, "sell", position["market_value"], current_price, signal.get("reason", ""))

    def _handle_filled_orders(self, newly_terminal: list):
        """Process orders that just transitioned to a terminal state."""
        from core.order_manager import OrderState
        for order in newly_terminal:
            # Sprint 5: surface rejections via AlertManager
            if order.state == OrderState.REJECTED:
                log.warning(
                    f"Order rejected: {order.side.upper()} {order.symbol} "
                    f"${(order.requested_notional or 0):.2f} — {order.error or 'unknown reason'}"
                )
                if self.alert_manager:
                    self.alert_manager.order_rejected(
                        order.symbol, order.side,
                        order.requested_notional or 0,
                        order.error or "unknown",
                    )
                continue
            if order.state != OrderState.FILLED:
                continue
            symbol = order.symbol
            is_equity = not Config.is_crypto(symbol)

            # Log slippage for all filled orders
            if order.expected_price and order.filled_avg_price:
                log.log_slippage(symbol, order.expected_price,
                                 order.filled_avg_price, order.slippage)

            if order.side == "buy":
                log.trade(symbol, "buy", order.requested_notional or 0,
                          order.filled_avg_price, "filled via poll",
                          strategy=get_strategy_key(symbol))
                # df unavailable at poll time — falls back to fixed stops (no ATR)
                self.risk.register_entry(symbol, order.filled_avg_price)
                self._record_trade(symbol)
                if is_equity:
                    self._record_buy(symbol)
                if self.alert_manager and Config.ALERT_ON_TRADE:
                    self.alert_manager.trade_executed(
                        symbol, "buy", order.requested_notional or 0,
                        order.filled_avg_price, "filled via poll")

    def _handle_completed_executions(self, completed_plans: list):
        """Handle execution plans that finished all child orders."""
        for plan in completed_plans:
            avg_price = plan.avg_fill_price
            if avg_price == 0:
                log.warning(f"  {plan.symbol}: {plan.algo.upper()} execution had no fills")
                continue
            log.info(
                f"  {plan.symbol}: {plan.algo.upper()} execution complete — "
                f"{plan.filled_children} slices, avg fill ${avg_price:.2f}, "
                f"slippage vs arrival: {plan.slippage_bps:+.1f} bps"
            )
            log.trade(plan.symbol, "buy", plan.filled_notional,
                      avg_price,
                      f"{plan.algo.upper()} execution ({plan.filled_children} slices)",
                      strategy=get_strategy_key(plan.symbol))
            self.risk.register_entry(plan.symbol, avg_price)

    def _get_existing_position_dfs(self) -> dict[str, "pd.DataFrame"]:
        """Fetch recent bars for all open positions (for correlation checks)."""
        if not hasattr(self, "_cached_position_dfs") or self._cached_position_dfs is None:
            import pandas as pd
            self._cached_position_dfs = {}
            positions = self.broker.get_positions()
            for p in positions:
                sym = p["symbol"]
                # Reverse the Alpaca symbol stripping for crypto
                for crypto_sym in Config.CRYPTO_SYMBOLS:
                    if crypto_sym.replace("/", "") == sym:
                        sym = crypto_sym
                        break
                try:
                    bars = self.broker.get_recent_bars(sym, limit=100)
                    if not bars.empty:
                        self._cached_position_dfs[sym] = bars
                except Exception:
                    pass
        return self._cached_position_dfs

    # ── Portfolio Optimization ──────────────────────────────────────────

    def _recompute_allocation(self, equity: float):
        """Recompute mean-variance allocation from recent bar data."""
        import pandas as pd

        symbols = Config.SYMBOLS
        returns_data = {}
        for sym in symbols:
            try:
                bars = self.broker.get_recent_bars(sym, limit=Config.MEAN_VARIANCE_LOOKBACK_DAYS)
                if bars is not None and len(bars) >= 10 and "close" in bars.columns:
                    rets = bars["close"].pct_change().dropna()
                    if len(rets) >= 5:
                        returns_data[sym] = rets.values[-Config.MEAN_VARIANCE_LOOKBACK_DAYS:]
            except Exception:
                pass

        if len(returns_data) < 2:
            return

        # Align lengths
        min_len = min(len(v) for v in returns_data.values())
        aligned = {sym: vals[-min_len:] for sym, vals in returns_data.items()}
        returns_df = pd.DataFrame(aligned)

        self.portfolio_optimizer.update_allocation(returns_df, self.cycle_count)

    # ── Position Reconciliation ────────────────────────────────────────

    def _run_reconciliation(self):
        """Compare engine trailing stops with broker positions and flag mismatches."""
        try:
            positions = self.broker.get_positions()
            result = self.position_reconciler.reconcile(
                trailing_stops=self.risk.trailing_stops,
                broker_positions=positions,
                tracked_symbols=Config.SYMBOLS,
            )
            if not result.ok:
                for m in result.mismatches:
                    log.warning(f"Position mismatch: {m.symbol} — {m.issue}: {m.detail}")

                # Send alert BEFORE auto-fix so it reports actual problems
                if self.alert_manager:
                    self.alert_manager.position_mismatch(
                        [{"symbol": m.symbol, "issue": f"{m.issue}: {m.detail}"}
                         for m in result.mismatches])

                    # Sprint 5: dedicated alerts for external positions (one per symbol)
                    alerted = getattr(self, "_alerted_external_positions", None)
                    if alerted is None:
                        alerted = set()
                        self._alerted_external_positions = alerted
                    for m in result.mismatches:
                        if m.issue == "external_position" and m.symbol not in alerted:
                            pos = next((p for p in positions
                                        if p.get("symbol") == m.symbol), None)
                            if pos:
                                self.alert_manager.external_position(
                                    m.symbol,
                                    float(pos.get("qty", 0)),
                                    abs(float(pos.get("market_value", 0))),
                                )
                                alerted.add(m.symbol)

                # Auto-fix orphaned stops if enabled
                if Config.RECONCILIATION_AUTO_FIX:
                    cleaned = self.position_reconciler.auto_fix_orphaned_stops(
                        self.risk.trailing_stops, result)
                    if cleaned:
                        log.info(f"Reconciler auto-fixed {len(cleaned)} orphaned stop(s)")
        except Exception as e:
            log.error(f"Position reconciliation error: {e}")

    # ── Orphan Position Detection ──────────────────────────────────

    def _detect_orphan_positions(self):
        """Detect broker positions missing trailing stops and register them."""
        try:
            positions = self.broker.get_positions()
            registered = 0
            for p in positions:
                symbol = self._resolve_symbol(p["symbol"])
                if symbol not in self.risk.trailing_stops:
                    entry_price = float(p["avg_entry_price"])
                    current_price = float(p["current_price"])
                    market_value = abs(float(p["market_value"]))
                    log.warning(
                        f"Orphan position detected: {symbol} | "
                        f"entry=${entry_price:.2f} | current=${current_price:.2f} | "
                        f"value=${market_value:.2f} — registering trailing stop"
                    )
                    df = None
                    try:
                        df = self.broker.get_recent_bars(symbol, limit=100)
                    except Exception:
                        pass
                    self.risk.register_entry(symbol, entry_price, df)
                    if current_price > entry_price and symbol in self.risk.trailing_stops:
                        self.risk.trailing_stops[symbol].highest_price = current_price
                    registered += 1
            if positions:
                log.info(f"Orphan detection complete: {registered} new stops registered "
                         f"out of {len(positions)} broker positions")
        except Exception as e:
            log.error(f"Orphan position detection failed: {e}")

    def _resolve_symbol(self, raw_symbol: str) -> str:
        """Resolve Alpaca stripped symbol back to config symbol."""
        for crypto_sym in Config.CRYPTO_SYMBOLS:
            if crypto_sym.replace("/", "") == raw_symbol:
                return crypto_sym
        return raw_symbol

    # ── PDT Protection ───────────────────────────────────────────────

    def _record_buy(self, symbol: str):
        """Track when we bought an equity for PDT protection."""
        if Config.PDT_PROTECTION:
            self._equity_buys_today[symbol] = datetime.now(Config.MARKET_TZ)

    def _is_same_day_buy(self, symbol: str) -> bool:
        """Check if we bought this equity today (would be a day trade to sell)."""
        if not Config.PDT_PROTECTION:
            return False
        buy_time = self._equity_buys_today.get(symbol)
        if not buy_time:
            return False
        now = datetime.now(Config.MARKET_TZ)
        return buy_time.date() == now.date()

    def _clear_buy_record(self, symbol: str):
        self._equity_buys_today.pop(symbol, None)

    # ── Signal Handling ─────────────────────────────────────────────

    def _register_signal_handlers(self):
        """Register SIGTERM and SIGHUP handlers (must be called from main thread)."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle_sighup)

    def _handle_sigterm(self, signum, frame):
        # Only set flag — log.info() is not async-signal-safe
        self._shutdown_requested = True

    def _handle_sighup(self, signum, frame):
        # Only set flag — log.info() is not async-signal-safe
        self._reload_requested = True

    def _interruptible_sleep(self, seconds: int):
        """Sleep in 1-second increments so shutdown signals are handled promptly."""
        for _ in range(seconds):
            if self._shutdown_requested:
                break
            time.sleep(1)

    def _handle_reload(self):
        """Perform config hot-reload (triggered by SIGHUP or polling)."""
        if not self.config_reloader:
            from core.config_reloader import ConfigReloader
            self.config_reloader = ConfigReloader()
        try:
            changed = self.config_reloader.reload()
            if changed:
                log.info(f"Config reloaded via SIGHUP: {', '.join(changed.keys())}")
                if self.alert_manager:
                    self.alert_manager.config_reloaded(list(changed.keys()))
            else:
                log.info("Config reload: no changes detected")
        except Exception as e:
            log.error(f"Config reload failed: {e}")

    # ── Run Loop ─────────────────────────────────────────────────────

    def run(self, interval_seconds: int = 60):
        """Main loop — runs until SIGTERM, SIGHUP triggers reload."""
        self.initialize()
        self._register_signal_handlers()
        log.info(f"Starting trading loop (interval: {interval_seconds}s)")
        log.info("Press Ctrl+C or send SIGTERM to stop\n")

        while not self._shutdown_requested:
            try:
                if self._reload_requested:
                    self._handle_reload()
                    self._reload_requested = False

                self.run_cycle()
                self._interruptible_sleep(interval_seconds)
            except KeyboardInterrupt:
                log.info("\nReceived interrupt, shutting down...")
                self._shutdown_requested = True
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                traceback.print_exc()
                self._interruptible_sleep(interval_seconds * 2)

        self._shutdown()

    def _shutdown(self, reason: str = "manual"):
        """Clean shutdown — save state first (critical), then print summary."""
        log.info(f"Shutdown initiated (reason: {reason})")

        # Save state first — most critical operation
        self._save_persisted_state()

        # Send shutdown alert before potentially slow broker calls
        if self.alert_manager:
            try:
                self.alert_manager.bot_shutdown(reason)
            except Exception:
                pass

        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            log.snapshot(account["equity"], account["cash"])

            log.info(f"\nFinal equity: ${account['equity']:.2f}")
            log.info(f"Final cash: ${account['cash']:.2f}")

            if positions:
                log.info("Open positions:")
                crypto_open = []
                for p in positions:
                    sym = p["symbol"]
                    is_crypto = "USD" in sym and len(sym) > 4
                    asset_tag = "[CRYPTO]" if is_crypto else "[EQUITY]"
                    log.info(f"  {asset_tag} {sym}: {p['qty']} units @ ${p['avg_entry_price']:.2f} "
                             f"(PnL: ${p['unrealized_pl']:.2f})")
                    if is_crypto:
                        crypto_open.append(sym)
                if crypto_open:
                    log.warning(
                        f"WARNING: {len(crypto_open)} crypto position(s) still open at shutdown: "
                        f"{', '.join(crypto_open)} — these trade 24/7 and will have no active "
                        f"trailing stop protection until the bot restarts"
                    )

            # Warn about pending orders
            if self.order_manager and self.order_manager.pending_count > 0:
                log.warning(f"  {self.order_manager.pending_count} orders still pending at broker")

            pnl = account["equity"] - self.risk.starting_equity
            pnl_pct = (pnl / self.risk.starting_equity * 100) if self.risk.starting_equity else 0
            log.info(f"\nTotal PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            log.print_summary()
        except Exception as e:
            log.error(f"Error during shutdown summary: {e}")
