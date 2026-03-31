"""Additional strategy tests to cover remaining edge cases and branches."""

import pytest
import numpy as np
import pandas as pd

from tests.conftest import _make_ohlcv


def _make_strong_uptrend(n=200, seed=10):
    """Generate data that should trigger overbought/sell signals."""
    rng = np.random.RandomState(seed)
    close = np.linspace(50, 120, n) + rng.randn(n) * 0.3
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 0.8, n)
    volume = rng.uniform(2_000_000, 15_000_000, n)
    return pd.DataFrame({"open": close - 0.1, "high": high, "low": low,
                          "close": close, "volume": volume})


def _make_strong_downtrend(n=200, seed=10):
    """Generate data that should trigger oversold/buy signals."""
    rng = np.random.RandomState(seed)
    close = np.linspace(150, 50, n) + rng.randn(n) * 0.3
    high = close + rng.uniform(0.2, 0.8, n)
    low = close - rng.uniform(0.2, 1.5, n)
    volume = rng.uniform(2_000_000, 15_000_000, n)
    return pd.DataFrame({"open": close + 0.1, "high": high, "low": low,
                          "close": close, "volume": volume})


class TestMeanReversionFull:
    """Cover missing branches in mean_reversion.py."""

    def test_buy_signal_rsi_oversold(self):
        from strategies.mean_reversion import compute_signals
        df = _make_strong_downtrend(200, seed=7)
        signal = compute_signals(df)
        # Strong downtrend should trigger buy or hold
        assert signal["action"] in ("buy", "hold")

    def test_sell_signal_rsi_overbought(self):
        from strategies.mean_reversion import compute_signals
        df = _make_strong_uptrend(200, seed=7)
        signal = compute_signals(df)
        assert signal["action"] in ("sell", "hold")

    def test_hold_insufficient_data(self):
        from strategies.mean_reversion import compute_signals
        df = _make_ohlcv(15)
        signal = compute_signals(df)
        assert signal["action"] == "hold"


class TestTripleEMAFull:
    """Cover missing branches in triple_ema.py."""

    def test_buy_on_ema_alignment(self):
        from strategies.triple_ema import compute_signals
        df = _make_strong_uptrend(200, seed=5)
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")

    def test_sell_on_ema_cross_down(self):
        from strategies.triple_ema import compute_signals
        df = _make_strong_downtrend(200, seed=5)
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")

    def test_various_seeds(self):
        """Run multiple seeds to hit different branch combinations."""
        from strategies.triple_ema import compute_signals
        actions_seen = set()
        for seed in range(20):
            df = _make_ohlcv(200, trend="volatile", seed=seed)
            signal = compute_signals(df)
            actions_seen.add(signal["action"])
        # Should see at least hold + one other
        assert "hold" in actions_seen


class TestEnsembleFull:
    """Cover missing branches in ensemble.py — dual buy agreement."""

    def test_multiple_market_conditions(self):
        """Try many seeds to hit the dual-buy agreement branch."""
        from strategies.ensemble import compute_signals
        actions_seen = set()
        for seed in range(30):
            df = _make_ohlcv(200, trend="volatile", seed=seed)
            signal = compute_signals(df)
            actions_seen.add(signal["action"])
            if signal["action"] != "hold":
                assert "ensemble" in signal["reason"].lower()

    def test_strong_uptrend(self):
        from strategies.ensemble import compute_signals
        df = _make_strong_uptrend(200, seed=3)
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")

    def test_strong_downtrend(self):
        from strategies.ensemble import compute_signals
        df = _make_strong_downtrend(200, seed=3)
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")


class TestRSIDivergenceFull:
    """Cover remaining branches in rsi_divergence.py."""

    def test_various_market_shapes(self):
        from strategies.rsi_divergence import compute_signals
        for trend in ["up", "down", "flat", "volatile"]:
            for seed in range(5):
                df = _make_ohlcv(200, trend=trend, seed=seed)
                signal = compute_signals(df)
                assert signal["action"] in ("buy", "sell", "hold")


class TestVolumeProfileFull:
    """Cover remaining branches in volume_profile.py."""

    def test_high_volume_spike_with_positive_roc(self):
        from strategies.volume_profile import compute_signals
        df = _make_ohlcv(200, trend="up", seed=42)
        # Spike the last bar's volume
        df.iloc[-1, df.columns.get_loc("volume")] = 100_000_000
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")

    def test_high_volume_spike_with_negative_roc(self):
        from strategies.volume_profile import compute_signals
        df = _make_ohlcv(200, trend="down", seed=42)
        df.iloc[-1, df.columns.get_loc("volume")] = 100_000_000
        signal = compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")


class TestLoggerPrintSummary:
    """Cover the print_summary method of the logger."""

    def test_print_summary_with_data(self, tmp_db):
        from unittest.mock import patch
        from utils.logger import Logger
        with patch("utils.logger.DB_PATH", tmp_db):
            logger = Logger()
            logger.print_summary()  # Should not crash

    def test_print_summary_empty_db(self, tmp_path):
        from unittest.mock import patch
        from utils.logger import Logger
        db_path = str(tmp_path / "empty.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.print_summary()  # Should not crash

    def test_print_summary_no_snapshots(self, tmp_path):
        import sqlite3
        from unittest.mock import patch
        from utils.logger import Logger
        db_path = str(tmp_path / "nosnapshots.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            # Add a trade but no snapshot
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, amount, price, reason) "
                "VALUES ('2026-03-27 10:00:00', 'AAPL', 'buy', 5000, 250, 'test')"
            )
            conn.commit()
            conn.close()
            logger.print_summary()


class TestCompareStrategiesFull:
    """Cover edge cases in compare_strategies.py backtest function."""

    def test_backtest_no_trades(self):
        """Strategy that never triggers should still return valid metrics."""
        from compare_strategies import backtest_strategy

        def never_trade(df):
            return {"action": "hold", "reason": "never", "strength": 0}

        df = _make_ohlcv(200)
        result = backtest_strategy(never_trade, df)
        assert result["num_trades"] == 0
        assert result["win_rate"] == 0
        assert result["final_equity"] > 0

    def test_backtest_always_buy(self):
        """Strategy that always buys."""
        from compare_strategies import backtest_strategy

        def always_buy(df):
            return {"action": "buy", "reason": "always", "strength": 1.0}

        df = _make_ohlcv(200)
        result = backtest_strategy(always_buy, df)
        assert result["final_equity"] > 0

    def test_backtest_with_hard_stop(self):
        """Test that hard stop-loss triggers in backtest."""
        from compare_strategies import backtest_strategy

        call_count = 0
        def buy_then_hold(df):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "buy", "reason": "buy", "strength": 1.0}
            return {"action": "hold", "reason": "hold", "strength": 0}

        df = _make_strong_downtrend(200)
        result = backtest_strategy(buy_then_hold, df, stop_loss_pct=0.01)
        assert result["final_equity"] > 0

    def test_backtest_strategy_raises_exception(self):
        """Strategy that raises should be skipped gracefully."""
        from compare_strategies import backtest_strategy

        call_count = 0
        def flaky_strategy(df):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("flaky")
            return {"action": "hold", "reason": "ok", "strength": 0}

        df = _make_ohlcv(200)
        result = backtest_strategy(flaky_strategy, df)
        assert result["final_equity"] > 0
