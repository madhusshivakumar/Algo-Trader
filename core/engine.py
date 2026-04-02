"""Main trading engine — handles both crypto (24/7) and equities (market hours)."""

import signal
import time
import traceback
from datetime import datetime

from core.broker import Broker
from core.risk_manager import RiskManager
from strategies.router import compute_signals as route_signals, get_strategy, get_strategy_key
from config import Config
from utils.logger import log


class TradingEngine:
    # Minimum seconds between trades on the same symbol
    COOLDOWN_CRYPTO = 900   # 15 minutes
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

        # Alerting (optional)
        self.alert_manager = None
        if Config.ALERTING_ENABLED:
            from core.alerting import AlertManager
            self.alert_manager = AlertManager()

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

        # Restore persisted state (must come after risk.initialize)
        self._load_persisted_state()

        if Config.is_paper():
            log.warning("Running in PAPER trading mode")
        else:
            log.warning("Running in LIVE trading mode — real money!")

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
        account = self.broker.get_account()
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

        # v3: Config hot-reload every 30 cycles
        if self.config_reloader and self.cycle_count % 30 == 0:
            try:
                if self.config_reloader.check_for_changes():
                    changed = self.config_reloader.reload()
                    if changed:
                        log.info(f"Config reloaded: {', '.join(changed.keys())}")
            except Exception as e:
                log.error(f"Config reload error: {e}")

        # v3: Drift detection every 100 cycles
        if self.drift_detector and self.cycle_count % 100 == 0:
            try:
                alerts = self.drift_detector.should_alert()
                for alert in alerts:
                    log.warning(alert)
            except Exception as e:
                log.error(f"Drift detection error: {e}")

        # Snapshot equity every 10 cycles
        if self.cycle_count % 10 == 0:
            log.snapshot(equity, account["cash"])
            # v3: Save state periodically
            self._save_persisted_state()

        # Check risk limits
        if not self.risk.can_trade(equity):
            if self.cycle_count % 60 == 0:
                log.warning(f"Trading halted: {self.risk.halt_reason}")
            # Alert on first detection and every 60 cycles after
            if self.alert_manager and self.risk.halted and (self.cycle_count % 60 == 1):
                self.alert_manager.drawdown_halt(
                    self.risk.daily_drawdown, Config.DAILY_DRAWDOWN_LIMIT)
            return

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

    def _get_total_exposure(self) -> float:
        """Get total dollar value of all open positions."""
        positions = self.broker.get_positions()
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
        is_equity = not Config.is_crypto(symbol)

        # Skip if symbol is on cooldown
        if self._is_on_cooldown(symbol):
            return

        current_price = float(df["close"].iloc[-1])

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

        # Compute strategy signals (routed per symbol)
        signal = route_signals(symbol, df)

        if signal["action"] == "buy" and not position:
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

            # Per-symbol position sizing: 15% equity for equities, 35% for crypto
            if is_equity:
                base_pct = 0.15
            else:
                base_pct = 0.35

            # v3: Volatility-adjusted sizing
            if Config.VOLATILITY_SIZING_ENABLED:
                max_size = self.risk.calculate_volatility_adjusted_size(equity, df, base_pct)
            else:
                max_size = equity * base_pct

            # Don't exceed remaining budget
            max_size = min(max_size, remaining_budget)

            size = max_size * signal["strength"]
            size = max(size, 1.0)

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

            # v3: Route through OrderManager when enabled
            if self.order_manager:
                order = self.order_manager.submit_market_buy(symbol, size, current_price)
                if order and order.state.value == "filled":
                    log.trade(symbol, "buy", size, current_price, signal["reason"],
                              strategy=signal.get("strategy", ""))
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
                    log.trade(symbol, "buy", size, current_price, signal["reason"],
                              strategy=signal.get("strategy", ""))
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
                log.trade(symbol, "sell", position["market_value"], current_price,
                          signal["reason"], pnl, strategy=signal.get("strategy", ""))
                self.risk.unregister(symbol)
                self._record_trade(symbol)
                self._clear_buy_record(symbol)
                if self.alert_manager and Config.ALERT_ON_TRADE:
                    self.alert_manager.trade_executed(symbol, "sell", position["market_value"], current_price, signal.get("reason", ""))

    def _handle_filled_orders(self, newly_terminal: list):
        """Process orders that just transitioned to a terminal state."""
        from core.order_manager import OrderState
        for order in newly_terminal:
            if order.state != OrderState.FILLED:
                continue
            symbol = order.symbol
            is_equity = not Config.is_crypto(symbol)
            if order.side == "buy":
                log.trade(symbol, "buy", order.requested_notional or 0,
                          order.filled_avg_price, "filled via poll",
                          strategy=get_strategy_key(symbol))
                self.risk.register_entry(symbol, order.filled_avg_price)
                self._record_trade(symbol)
                if is_equity:
                    self._record_buy(symbol)
                if self.alert_manager and Config.ALERT_ON_TRADE:
                    self.alert_manager.trade_executed(
                        symbol, "buy", order.requested_notional or 0,
                        order.filled_avg_price, "filled via poll")

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
                for p in positions:
                    asset_tag = "[CRYPTO]" if "USD" in p["symbol"] and len(p["symbol"]) > 4 else "[EQUITY]"
                    log.info(f"  {asset_tag} {p['symbol']}: {p['qty']} units @ ${p['avg_entry_price']:.2f} "
                             f"(PnL: ${p['unrealized_pl']:.2f})")

            # Warn about pending orders
            if self.order_manager and self.order_manager.pending_count > 0:
                log.warning(f"  {self.order_manager.pending_count} orders still pending at broker")

            pnl = account["equity"] - self.risk.starting_equity
            pnl_pct = (pnl / self.risk.starting_equity * 100) if self.risk.starting_equity else 0
            log.info(f"\nTotal PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            log.print_summary()
        except Exception as e:
            log.error(f"Error during shutdown summary: {e}")
