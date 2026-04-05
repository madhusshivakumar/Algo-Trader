"""Tests for earnings calendar agent, signal modifier, and engine integration."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from config import Config


# ── Agent Tests ─────────────────────────────────────────────────────


class TestFetchEarnings:
    """Test the earnings API fetching logic."""

    def test_fetch_earnings_success(self):
        from agents.earnings_calendar import fetch_earnings

        # Use MARKET_TZ-aware "today" to match the agent's timezone logic
        today_market = datetime.now(Config.MARKET_TZ).date()
        target_date = (today_market + timedelta(days=3)).isoformat()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"date": target_date}]

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result is not None
        assert result["next_earnings_date"] == target_date
        assert result["days_until"] == 3

    def test_fetch_earnings_no_announcements(self):
        from agents.earnings_calendar import fetch_earnings

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result is not None
        assert result["next_earnings_date"] is None
        assert result["days_until"] is None

    def test_fetch_earnings_api_error(self):
        from agents.earnings_calendar import fetch_earnings

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result is None

    def test_fetch_earnings_404_graceful(self):
        from agents.earnings_calendar import fetch_earnings

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result is not None
        assert result["next_earnings_date"] is None

    def test_fetch_earnings_network_error(self):
        import requests
        from agents.earnings_calendar import fetch_earnings

        with patch("agents.earnings_calendar.requests.get",
                   side_effect=requests.RequestException("timeout")):
            result = fetch_earnings("AAPL")

        assert result is None

    def test_fetch_earnings_multiple_dates_picks_earliest(self):
        from agents.earnings_calendar import fetch_earnings

        today = datetime.now(Config.MARKET_TZ).date()
        d1 = (today + timedelta(days=10)).isoformat()
        d2 = (today + timedelta(days=3)).isoformat()
        d3 = (today + timedelta(days=20)).isoformat()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"date": d1}, {"date": d2}, {"date": d3},
        ]

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result["next_earnings_date"] == d2
        assert result["days_until"] == 3

    def test_fetch_earnings_past_dates_ignored(self):
        from agents.earnings_calendar import fetch_earnings

        today = datetime.now(Config.MARKET_TZ).date()
        yesterday = (today - timedelta(days=1)).isoformat()
        tomorrow = (today + timedelta(days=1)).isoformat()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"date": yesterday}, {"date": tomorrow},
        ]

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result["next_earnings_date"] == tomorrow

    def test_fetch_earnings_bad_date_format_handled(self):
        from agents.earnings_calendar import fetch_earnings

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"date": "not-a-date"}]

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result is not None
        assert result["next_earnings_date"] is None

    def test_fetch_earnings_uses_record_date_fallback(self):
        from agents.earnings_calendar import fetch_earnings

        tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"record_date": tomorrow}]

        with patch("agents.earnings_calendar.requests.get", return_value=mock_response):
            result = fetch_earnings("AAPL")

        assert result["next_earnings_date"] == tomorrow


class TestRunAnalysis:
    """Test the analysis pipeline."""

    def test_filters_crypto_symbols(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {"next_earnings_date": None, "days_until": None}
            result = run_analysis(symbols=["AAPL", "BTC/USD", "TSLA"])

        # BTC/USD should be skipped
        assert "AAPL" in result["earnings"]
        assert "TSLA" in result["earnings"]
        assert "BTC/USD" not in result["earnings"]

    def test_blackout_computation_within_window(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {
                "next_earnings_date": "2026-04-03",
                "days_until": 1,
            }
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert result["earnings"]["AAPL"]["in_blackout"] is True

    def test_blackout_computation_outside_window(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {
                "next_earnings_date": "2026-04-20",
                "days_until": 19,
            }
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert result["earnings"]["AAPL"]["in_blackout"] is False

    def test_blackout_exact_boundary(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {
                "next_earnings_date": "2026-04-03",
                "days_until": 2,
            }
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert result["earnings"]["AAPL"]["in_blackout"] is True

    def test_api_error_defaults_safe(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings", return_value=None):
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert result["earnings"]["AAPL"]["in_blackout"] is False

    def test_no_earnings_date_not_in_blackout(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {"next_earnings_date": None, "days_until": None}
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert result["earnings"]["AAPL"]["in_blackout"] is False

    def test_output_structure(self):
        from agents.earnings_calendar import run_analysis

        with patch("agents.earnings_calendar.fetch_earnings") as mock_fetch:
            mock_fetch.return_value = {"next_earnings_date": None, "days_until": None}
            result = run_analysis(symbols=["AAPL"], blackout_days=2)

        assert "timestamp" in result
        assert "blackout_days" in result
        assert "earnings" in result
        assert isinstance(result["earnings"], dict)


class TestWriteOutput:
    """Test JSON file output."""

    def test_writes_json_file(self):
        from agents.earnings_calendar import write_output

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.json")
            with patch("agents.earnings_calendar._OUTPUT_FILE", output_file), \
                 patch("agents.earnings_calendar._DATA_DIR", tmpdir):
                write_output({"timestamp": "now", "earnings": {"AAPL": {}}})

            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)
            assert "AAPL" in data["earnings"]


class TestMain:
    """Test main entry point."""

    def test_main_runs_pipeline(self):
        from agents.earnings_calendar import main

        with patch("agents.earnings_calendar.run_analysis") as mock_run, \
             patch("agents.earnings_calendar.write_output") as mock_write:
            mock_run.return_value = {
                "timestamp": "now",
                "earnings": {"AAPL": {"in_blackout": True}},
            }
            result = main()

        mock_run.assert_called_once()
        mock_write.assert_called_once()
        assert result is not None


# ── Signal Modifier Tests ───────────────────────────────────────────


class TestApplyEarningsBlackout:
    """Test the earnings blackout signal modifier."""

    def setup_method(self):
        """Reset earnings cache between tests."""
        import core.signal_modifiers as sm
        sm._earnings_cache = {}
        sm._earnings_cache_mtime = 0.0

    def _write_earnings_file(self, tmpdir, earnings_data):
        """Helper to write a fake earnings JSON file."""
        filepath = os.path.join(tmpdir, "output.json")
        with open(filepath, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "earnings": earnings_data,
            }, f)
        return filepath

    def test_crypto_passthrough(self):
        from core.signal_modifiers import apply_earnings_blackout

        signal = {"action": "buy", "strength": 0.8, "reason": "test"}
        result = apply_earnings_blackout(signal, "BTC/USD")
        assert result["action"] == "buy"
        assert result["strength"] == 0.8

    def test_no_file_passthrough(self):
        from core.signal_modifiers import apply_earnings_blackout

        with patch("core.signal_modifiers._EARNINGS_FILE", "/nonexistent/path.json"):
            signal = {"action": "buy", "strength": 0.8, "reason": "test"}
            result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "buy"
        assert result["strength"] == 0.8

    def test_not_in_blackout_passthrough(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "AAPL": {"in_blackout": False, "days_until": 20},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                signal = {"action": "buy", "strength": 0.8, "reason": "test"}
                result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "buy"
        assert result["strength"] == 0.8
        assert "earnings_blackout" not in result

    def test_buy_signal_reduced(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "AAPL": {"in_blackout": True, "days_until": 1},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath), \
                 patch.object(Config, "EARNINGS_SIZE_REDUCTION", 0.5):
                signal = {"action": "buy", "strength": 0.8, "reason": "test"}
                result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "buy"
        assert result["strength"] < 0.8
        assert result["earnings_blackout"] is True
        assert result["days_to_earnings"] == 1
        assert "earnings_blackout" in result["reason"]

    def test_buy_signal_blocked_at_zero_reduction(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "AAPL": {"in_blackout": True, "days_until": 1},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath), \
                 patch.object(Config, "EARNINGS_SIZE_REDUCTION", 0.0):
                signal = {"action": "buy", "strength": 0.8, "reason": "test"}
                result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "hold"
        assert result["strength"] == 0.0
        assert "blocked" in result["reason"]

    def test_sell_signal_passthrough(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "AAPL": {"in_blackout": True, "days_until": 1},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                signal = {"action": "sell", "strength": 0.8, "reason": "test"}
                result = apply_earnings_blackout(signal, "AAPL")

        # Sell signals should not be dampened
        assert result["action"] == "sell"
        assert result["strength"] == 0.8
        assert result["earnings_blackout"] is True

    def test_hold_signal_annotated(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "AAPL": {"in_blackout": True, "days_until": 2},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                signal = {"action": "hold", "strength": 0.5, "reason": "no signal"}
                result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "hold"
        assert result["earnings_blackout"] is True
        assert "earnings_blackout" in result["reason"]

    def test_symbol_not_in_data_passthrough(self):
        from core.signal_modifiers import apply_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._write_earnings_file(tmpdir, {
                "TSLA": {"in_blackout": True, "days_until": 1},
            })
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                signal = {"action": "buy", "strength": 0.8, "reason": "test"}
                result = apply_earnings_blackout(signal, "AAPL")

        assert result["action"] == "buy"
        assert result["strength"] == 0.8


class TestIsInEarningsBlackout:
    """Test the quick-check helper used by engine."""

    def setup_method(self):
        """Reset earnings cache between tests."""
        import core.signal_modifiers as sm
        sm._earnings_cache = {}
        sm._earnings_cache_mtime = 0.0

    def test_returns_true_in_blackout(self):
        from core.signal_modifiers import is_in_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output.json")
            with open(filepath, "w") as f:
                json.dump({"earnings": {"AAPL": {"in_blackout": True}}}, f)
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                assert is_in_earnings_blackout("AAPL") is True

    def test_returns_false_not_in_blackout(self):
        from core.signal_modifiers import is_in_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output.json")
            with open(filepath, "w") as f:
                json.dump({"earnings": {"AAPL": {"in_blackout": False}}}, f)
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                assert is_in_earnings_blackout("AAPL") is False

    def test_returns_false_missing_file(self):
        from core.signal_modifiers import is_in_earnings_blackout

        with patch("core.signal_modifiers._EARNINGS_FILE", "/nonexistent.json"):
            assert is_in_earnings_blackout("AAPL") is False

    def test_returns_false_missing_symbol(self):
        from core.signal_modifiers import is_in_earnings_blackout

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output.json")
            with open(filepath, "w") as f:
                json.dump({"earnings": {"TSLA": {"in_blackout": True}}}, f)
            with patch("core.signal_modifiers._EARNINGS_FILE", filepath):
                assert is_in_earnings_blackout("AAPL") is False


# ── Engine Integration Tests ────────────────────────────────────────


class TestEngineEarningsClose:
    """Test pre-earnings position closure in the engine."""

    def _make_engine(self):
        from core.engine import TradingEngine
        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.risk = MagicMock()
        engine.risk.is_max_hold_exceeded.return_value = False
        engine.order_manager = None
        engine.data_fetcher = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.cost_model = None
        engine.alert_manager = None
        engine.db_rotator = None
        engine.state_store = None
        engine.execution_manager = None
        engine.portfolio_optimizer = None
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine._cached_position_dfs = None
        engine.cycle_count = 1
        return engine

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", True)
    @patch("core.signal_modifiers.is_in_earnings_blackout", return_value=True)
    def test_closes_equity_position_in_blackout(self, mock_blackout):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        # Set up position
        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": -100.0,
        }
        engine.broker.close_position.return_value = True

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals") as mock_route, \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.close_position.assert_called_once_with("AAPL")
        engine.risk.unregister.assert_called_once_with("AAPL")
        # Signal computation should NOT have happened (returned early)
        mock_route.assert_not_called()

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", True)
    @patch("core.signal_modifiers.is_in_earnings_blackout", return_value=True)
    @patch.object(Config, "PDT_PROTECTION", True)
    def test_pdt_blocks_earnings_close(self, mock_blackout):
        import pandas as pd
        import numpy as np
        from datetime import datetime
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": -100.0,
        }
        engine._equity_buys_today = {"AAPL": datetime.now(Config.MARKET_TZ)}

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log") as mock_log:
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.close_position.assert_not_called()

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", False)
    def test_no_close_when_close_flag_off(self):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": 100.0,
        }
        engine.risk.should_stop_loss.return_value = False

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals",
                   return_value={"action": "hold", "strength": 0.5, "reason": "test"}):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.close_position.assert_not_called()

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", True)
    @patch("core.signal_modifiers.is_in_earnings_blackout", return_value=True)
    def test_crypto_skips_earnings_close(self, mock_blackout):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": 100.0,
        }
        engine.risk.should_stop_loss.return_value = False

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals",
                   return_value={"action": "hold", "strength": 0.5, "reason": "test"}):
            engine._process_symbol_inner("BTC/USD", 10000.0, df)

        # Crypto positions should never be closed for earnings
        engine.broker.close_position.assert_not_called()
        mock_blackout.assert_not_called()

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", True)
    @patch("core.signal_modifiers.is_in_earnings_blackout", return_value=True)
    def test_no_cooldown_on_earnings_close(self, mock_blackout):
        """Earnings close should NOT set cooldown (it's defensive, not a trade signal)."""
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": -100.0,
        }
        engine.broker.close_position.return_value = True

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        # Cooldown should NOT be set for earnings close
        assert "AAPL" not in engine._last_trade_time

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False)
    @patch("core.signal_modifiers.is_in_earnings_blackout", return_value=True)
    def test_earnings_disabled_skips_close(self, mock_blackout):
        """EARNINGS_CALENDAR_ENABLED=False should skip the entire earnings block."""
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": 100.0,
        }
        engine.risk.should_stop_loss.return_value = False

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals",
                   return_value={"action": "hold", "strength": 0.5, "reason": "test"}):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.close_position.assert_not_called()
        # is_in_earnings_blackout should never be called when feature is disabled
        mock_blackout.assert_not_called()

    @patch.object(Config, "EARNINGS_CALENDAR_ENABLED", True)
    @patch.object(Config, "EARNINGS_CLOSE_POSITIONS", True)
    def test_trailing_stop_fires_before_earnings(self):
        """When trailing stop hits, it should close before earnings check runs."""
        import pandas as pd
        import numpy as np
        engine = self._make_engine()

        engine.broker.get_position.return_value = {
            "market_value": 5000.0, "unrealized_pl": -500.0,
        }
        engine.risk.should_stop_loss.return_value = True
        engine.broker.close_position.return_value = True
        engine.risk.trailing_stops = {}

        df = pd.DataFrame({
            "close": np.linspace(100, 95, 50),
            "open": np.linspace(99, 94, 50),
            "high": np.linspace(101, 96, 50),
            "low": np.linspace(98, 93, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.get_strategy_key", return_value="momentum"), \
             patch("core.signal_modifiers.is_in_earnings_blackout") as mock_blackout:
            engine._process_symbol_inner("AAPL", 10000.0, df)

        # Position closed by trailing stop
        engine.broker.close_position.assert_called_once_with("AAPL")
        # Earnings blackout check should never have been reached
        mock_blackout.assert_not_called()
