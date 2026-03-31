"""Main trading engine — handles both crypto (24/7) and equities (market hours)."""

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
        # Track which equity positions were opened today (PDT protection)
        self._equity_buys_today: dict[str, datetime] = {}
        # Cooldown: last trade timestamp per symbol
        self._last_trade_time: dict[str, float] = {}

    def initialize(self):
        """Set up initial state."""
        account = self.broker.get_account()
        log.info(f"Account equity: ${account['equity']:.2f} | Cash: ${account['cash']:.2f}")
        self.risk.initialize(account["equity"])

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
        account = self.broker.get_account()
        equity = account["equity"]

        # Snapshot equity every 10 cycles
        if self.cycle_count % 10 == 0:
            log.snapshot(equity, account["cash"])

        # Check risk limits
        if not self.risk.can_trade(equity):
            if self.cycle_count % 60 == 0:
                log.warning(f"Trading halted: {self.risk.halt_reason}")
            return

        # ── Crypto (always runs) ─────────────────────────────────────
        for symbol in Config.CRYPTO_SYMBOLS:
            try:
                self._process_symbol(symbol, equity)
            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")
                if self.cycle_count <= 3:
                    traceback.print_exc()

        # ── Equities (only during market hours) ──────────────────────
        if Config.is_market_open():
            for symbol in Config.EQUITY_SYMBOLS:
                try:
                    self._process_symbol(symbol, equity)
                except Exception as e:
                    log.error(f"Error processing {symbol}: {e}")
                    if self.cycle_count <= 3:
                        traceback.print_exc()
        elif self.cycle_count % 60 == 0:
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

    def _process_symbol(self, symbol: str, equity: float):
        """Evaluate signals and act for one symbol."""
        is_equity = not Config.is_crypto(symbol)

        # Skip if symbol is on cooldown
        if self._is_on_cooldown(symbol):
            return

        # Get market data
        df = self.broker.get_recent_bars(symbol, limit=100)
        if df.empty or len(df) < 30:
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
                result = self.broker.close_position(symbol)
                if result:
                    pnl = position["unrealized_pl"]
                    log.trade(symbol, "sell", position["market_value"], current_price,
                              "trailing stop", pnl, strategy=get_strategy_key(symbol))
                    self.risk.unregister(symbol)
                    self._clear_buy_record(symbol)
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

            # Per-symbol position sizing: 15% equity for equities, 35% for crypto
            if is_equity:
                max_size = equity * 0.15
            else:
                max_size = equity * 0.35

            # Don't exceed remaining budget
            max_size = min(max_size, remaining_budget)

            size = max_size * signal["strength"]
            size = max(size, 1.0)

            log.info(f"BUY signal for {symbol}: {signal['reason']} (strength: {signal['strength']:.2f})")
            result = self.broker.buy(symbol, size)
            if result:
                log.trade(symbol, "buy", size, current_price, signal["reason"],
                          strategy=signal.get("strategy", ""))
                self.risk.register_entry(symbol, current_price)
                self._record_trade(symbol)
                if is_equity:
                    self._record_buy(symbol)

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

    # ── Run Loop ─────────────────────────────────────────────────────

    def run(self, interval_seconds: int = 60):
        """Main loop — runs forever."""
        self.initialize()
        log.info(f"Starting trading loop (interval: {interval_seconds}s)")
        log.info("Press Ctrl+C to stop\n")

        while True:
            try:
                self.run_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                log.info("\nStopping trading bot...")
                self._shutdown()
                break
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                traceback.print_exc()
                time.sleep(interval_seconds * 2)

    def _shutdown(self):
        """Clean shutdown — print summary."""
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

        pnl = account["equity"] - self.risk.starting_equity
        pnl_pct = (pnl / self.risk.starting_equity * 100) if self.risk.starting_equity else 0
        log.info(f"\nTotal PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        log.print_summary()
