"""Tests for agents/market_scanner.py — covers utility functions and main() with mocking."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd
import numpy as np


class TestLoadOptimizerAssignments:
    def test_load_from_assignments_file(self, tmp_path):
        from agents.market_scanner import load_optimizer_assignments
        af = tmp_path / "optimizer" / "strategy_assignments.json"
        af.parent.mkdir(parents=True)
        af.write_text(json.dumps({
            "assignments": {
                "AAPL": {"strategy": "momentum", "reason": "best"},
                "BTC/USD": "mean_reversion",
            }
        }))
        with patch("agents.market_scanner.ASSIGNMENTS_FILE", str(af)), \
             patch("agents.market_scanner.AGENT_STATE_FILE", str(tmp_path / "nope.json")), \
             patch("agents.market_scanner.FALLBACK_FILE", str(tmp_path / "nope2.json")):
            result = load_optimizer_assignments()
        assert result["AAPL"] == "momentum"
        assert result["BTC/USD"] == "mean_reversion"

    def test_load_with_agent_state_success_today(self, tmp_path):
        from agents.market_scanner import load_optimizer_assignments
        sf = tmp_path / "agent_state.json"
        today = datetime.now().strftime("%Y-%m-%d")
        sf.write_text(json.dumps({
            "strategy_optimizer": {"status": "success", "last_run": f"{today}T05:30:00"}
        }))
        af = tmp_path / "assignments.json"
        af.write_text(json.dumps({"assignments": {"AAPL": {"strategy": "scalper"}}}))
        with patch("agents.market_scanner.AGENT_STATE_FILE", str(sf)), \
             patch("agents.market_scanner.ASSIGNMENTS_FILE", str(af)), \
             patch("agents.market_scanner.FALLBACK_FILE", str(tmp_path / "nope.json")):
            result = load_optimizer_assignments()
        assert result["AAPL"] == "scalper"

    def test_fallback_to_fallback_config(self, tmp_path):
        from agents.market_scanner import load_optimizer_assignments
        ff = tmp_path / "fallback.json"
        ff.write_text(json.dumps({"strategy_map": {"TSLA": "momentum"}}))
        with patch("agents.market_scanner.AGENT_STATE_FILE", str(tmp_path / "nope.json")), \
             patch("agents.market_scanner.ASSIGNMENTS_FILE", str(tmp_path / "nope2.json")), \
             patch("agents.market_scanner.FALLBACK_FILE", str(ff)):
            result = load_optimizer_assignments()
        assert result["TSLA"] == "momentum"

    def test_all_missing_returns_empty(self, tmp_path):
        from agents.market_scanner import load_optimizer_assignments
        with patch("agents.market_scanner.AGENT_STATE_FILE", str(tmp_path / "a.json")), \
             patch("agents.market_scanner.ASSIGNMENTS_FILE", str(tmp_path / "b.json")), \
             patch("agents.market_scanner.FALLBACK_FILE", str(tmp_path / "c.json")):
            assert load_optimizer_assignments() == {}

    def test_corrupt_assignments_file(self, tmp_path):
        from agents.market_scanner import load_optimizer_assignments
        af = tmp_path / "assignments.json"
        af.write_text("{bad json")
        ff = tmp_path / "fallback.json"
        ff.write_text(json.dumps({"strategy_map": {"X": "y"}}))
        with patch("agents.market_scanner.AGENT_STATE_FILE", str(tmp_path / "nope.json")), \
             patch("agents.market_scanner.ASSIGNMENTS_FILE", str(af)), \
             patch("agents.market_scanner.FALLBACK_FILE", str(ff)):
            result = load_optimizer_assignments()
        assert result == {"X": "y"}


class TestLoadLearnings:
    def test_load_existing(self, tmp_path):
        from agents.market_scanner import load_learnings
        lf = tmp_path / "learnings.json"
        lf.write_text(json.dumps({
            "entries": [{"findings": {"symbols_to_favor": ["AAPL"], "symbols_to_avoid": ["META"]}}]
        }))
        with patch("agents.market_scanner.LEARNINGS_FILE", str(lf)):
            favor, avoid = load_learnings()
        assert "AAPL" in favor
        assert "META" in avoid

    def test_load_missing(self, tmp_path):
        from agents.market_scanner import load_learnings
        with patch("agents.market_scanner.LEARNINGS_FILE", str(tmp_path / "nope.json")):
            favor, avoid = load_learnings()
        assert favor == []
        assert avoid == []

    def test_load_corrupt(self, tmp_path):
        from agents.market_scanner import load_learnings
        lf = tmp_path / "learnings.json"
        lf.write_text("{bad")
        with patch("agents.market_scanner.LEARNINGS_FILE", str(lf)):
            favor, avoid = load_learnings()
        assert favor == []
        assert avoid == []


class TestCheckDeadline:
    def test_before_deadline(self):
        from agents.market_scanner import check_deadline
        mock_now = datetime(2026, 3, 29, 6, 0, 0)
        with patch("agents.market_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert check_deadline() is False

    def test_after_deadline_morning(self):
        from agents.market_scanner import check_deadline
        mock_now = datetime(2026, 3, 29, 6, 26, 0)
        with patch("agents.market_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert check_deadline() is True

    def test_after_noon_not_enforced(self):
        from agents.market_scanner import check_deadline
        mock_now = datetime(2026, 3, 29, 14, 0, 0)
        with patch("agents.market_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert check_deadline() is False


class TestScoreCandidate:
    def _make_df(self, n=5, price=100, volume=5_000_000, atr_factor=1.0):
        rng = np.random.RandomState(42)
        close = np.full(n, price) + rng.randn(n) * 0.5
        high = close + atr_factor * 2.0
        low = close - atr_factor * 2.0
        return pd.DataFrame({
            "open": close + rng.randn(n) * 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, volume),
        })

    def test_good_candidate(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=100, volume=5_000_000, atr_factor=1.0)
        result = score_candidate("AAPL", df, 0.7, [], [])
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["composite_score"] > 0

    def test_filtered_by_price_low(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=5, volume=5_000_000)
        assert score_candidate("PENNY", df, 0.5, [], []) is None

    def test_filtered_by_price_high(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=600, volume=5_000_000)
        assert score_candidate("EXPENSIVE", df, 0.5, [], []) is None

    def test_filtered_by_low_volume(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=100, volume=100_000)
        assert score_candidate("LOW_VOL", df, 0.5, [], []) is None

    def test_insufficient_data(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=2, price=100)
        assert score_candidate("X", df, 0.5, [], []) is None

    def test_avoid_penalty(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=100, volume=5_000_000, atr_factor=1.0)
        normal = score_candidate("AAPL", df, 0.7, [], [])
        avoided = score_candidate("AAPL", df, 0.7, [], ["AAPL"])
        if normal and avoided:
            assert avoided["composite_score"] < normal["composite_score"]

    def test_favor_boost(self):
        from agents.market_scanner import score_candidate
        df = self._make_df(n=5, price=100, volume=5_000_000, atr_factor=1.0)
        normal = score_candidate("AAPL", df, 0.7, [], [])
        favored = score_candidate("AAPL", df, 0.7, ["AAPL"], [])
        if normal and favored:
            assert favored["composite_score"] > normal["composite_score"]


class TestApplySectorDiversity:
    def test_limits_per_sector(self):
        from agents.market_scanner import apply_sector_diversity
        candidates = [{"symbol": f"T{i}", "sector": "Tech", "composite_score": 1.0 - i * 0.01}
                      for i in range(10)]
        result = apply_sector_diversity(candidates, max_per_sector=4)
        assert len(result) == 4

    def test_diverse_sectors_pass_through(self):
        from agents.market_scanner import apply_sector_diversity
        candidates = [
            {"symbol": "AAPL", "sector": "Tech", "composite_score": 0.9},
            {"symbol": "JPM", "sector": "Finance", "composite_score": 0.8},
            {"symbol": "XOM", "sector": "Energy", "composite_score": 0.7},
        ]
        result = apply_sector_diversity(candidates)
        assert len(result) == 3


class TestUpdateEnvFile:
    def test_update_existing_line(self, tmp_path):
        from agents.market_scanner import update_env_file
        env = tmp_path / ".env"
        env.write_text("ALPACA_API_KEY=abc\nEQUITY_SYMBOLS=OLD,LIST\nOTHER=val\n")
        with patch("agents.market_scanner.ENV_FILE", str(env)):
            update_env_file(["AAPL", "MSFT", "TSLA"])
        content = env.read_text()
        assert "EQUITY_SYMBOLS=AAPL,MSFT,TSLA" in content
        assert "ALPACA_API_KEY=abc" in content

    def test_add_new_line(self, tmp_path):
        from agents.market_scanner import update_env_file
        env = tmp_path / ".env"
        env.write_text("ALPACA_API_KEY=abc\n")
        with patch("agents.market_scanner.ENV_FILE", str(env)):
            update_env_file(["AAPL"])
        content = env.read_text()
        assert "EQUITY_SYMBOLS=AAPL" in content

    def test_missing_env(self, tmp_path):
        from agents.market_scanner import update_env_file
        with patch("agents.market_scanner.ENV_FILE", str(tmp_path / "nope.env")):
            update_env_file(["AAPL"])  # Should not raise


class TestScannerUpdateAgentState:
    def test_write_state(self, tmp_path):
        from agents.market_scanner import update_agent_state
        sf = str(tmp_path / "agent_state.json")
        with patch("agents.market_scanner.AGENT_STATE_FILE", sf):
            update_agent_state("success", symbols_selected=15, duration=10)
        data = json.loads(open(sf).read())
        assert data["market_scanner"]["status"] == "success"
        assert data["market_scanner"]["symbols_selected"] == 15


class TestFetchStockData:
    def test_fetch_batches(self):
        from agents.market_scanner import fetch_stock_data
        mock_client = MagicMock()
        rng = np.random.RandomState(42)
        n = 5
        idx = pd.MultiIndex.from_arrays([
            ["AAPL"] * n + ["MSFT"] * n,
            list(pd.date_range("2026-03-20", periods=n, freq="1D")) * 2,
        ], names=["symbol", "timestamp"])
        df = pd.DataFrame({
            "open": rng.uniform(100, 200, n * 2),
            "high": rng.uniform(100, 200, n * 2),
            "low": rng.uniform(100, 200, n * 2),
            "close": rng.uniform(100, 200, n * 2),
            "volume": rng.randint(1_000_000, 10_000_000, n * 2),
            "trade_count": rng.randint(100, 1000, n * 2),
            "vwap": rng.uniform(100, 200, n * 2),
        }, index=idx)
        mock_bars = MagicMock()
        mock_bars.df = df
        mock_client.get_stock_bars.return_value = mock_bars
        result = fetch_stock_data(mock_client, ["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result

    def test_fetch_error_continues(self):
        from agents.market_scanner import fetch_stock_data
        mock_client = MagicMock()
        mock_client.get_stock_bars.side_effect = Exception("API error")
        result = fetch_stock_data(mock_client, ["AAPL"])
        assert result == {}


class TestScannerMain:
    def test_main_success(self, tmp_path):
        from agents import market_scanner as scan_mod

        # Setup dirs
        (tmp_path / "optimizer").mkdir()
        (tmp_path / "scanner").mkdir()
        (tmp_path / "history" / "symbols").mkdir(parents=True)
        (tmp_path / "history" / "assignments").mkdir(parents=True)

        env_file = tmp_path / ".env"
        env_file.write_text("ALPACA_API_KEY=test\nEQUITY_SYMBOLS=OLD\n")

        # Mock stock data — must produce enough candidates (>= MIN_SYMBOLS=6)
        rng = np.random.RandomState(42)
        n = 5

        def mock_fetch(client, symbols):
            result = {}
            for sym in symbols:
                base_price = 100
                close = np.full(n, base_price) + rng.randn(n) * 0.5
                result[sym] = pd.DataFrame({
                    "open": close + rng.randn(n) * 0.1,
                    "high": close + 3.0,   # ATR ~3% — passes filter
                    "low": close - 3.0,
                    "close": close,
                    "volume": np.full(n, 5_000_000),
                })
            return result

        # Mock score_candidate to return valid scores for enough symbols
        fake_candidates = [
            {"symbol": sym, "price": 100, "avg_volume": 5_000_000, "atr_pct": 0.03,
             "momentum_5d": 0.02, "vol_score": 0.5, "volatility_score": 0.5,
             "momentum_score": 0.5, "backtest_score": 0.7, "composite_score": 0.6,
             "sector": sector, "favored": False, "avoided": False}
            for sym, sector in [
                ("AAPL", "Tech"), ("MSFT", "Tech"), ("NVDA", "Tech"),
                ("JPM", "Finance"), ("BAC", "Finance"),
                ("XOM", "Energy"), ("CVX", "Energy"),
                ("TSLA", "Consumer"), ("AMZN", "Consumer"),
                ("BA", "Industrial"),
            ]
        ]
        call_count = [0]
        def mock_score(*args, **kwargs):
            if call_count[0] < len(fake_candidates):
                result = fake_candidates[call_count[0]]
                call_count[0] += 1
                return result
            return None

        with patch.object(scan_mod, "DATA_DIR", str(tmp_path)), \
             patch.object(scan_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch.object(scan_mod, "ASSIGNMENTS_FILE", str(tmp_path / "optimizer" / "strategy_assignments.json")), \
             patch.object(scan_mod, "FALLBACK_FILE", str(tmp_path / "fallback.json")), \
             patch.object(scan_mod, "LEARNINGS_FILE", str(tmp_path / "learnings.json")), \
             patch.object(scan_mod, "SELECTED_FILE", str(tmp_path / "scanner" / "selected_symbols.json")), \
             patch.object(scan_mod, "CANDIDATES_FILE", str(tmp_path / "scanner" / "candidates.json")), \
             patch.object(scan_mod, "ENV_FILE", str(env_file)), \
             patch("agents.market_scanner.Config") as mock_config, \
             patch("agents.market_scanner.StockHistoricalDataClient"), \
             patch("agents.market_scanner.fetch_stock_data", side_effect=mock_fetch), \
             patch("agents.market_scanner.score_candidate", side_effect=mock_score), \
             patch("agents.market_scanner.check_deadline", return_value=False):
            mock_config.validate.return_value = None
            mock_config.ALPACA_API_KEY = "test"
            mock_config.ALPACA_SECRET_KEY = "test"
            mock_config.CRYPTO_SYMBOLS = ["BTC/USD"]
            # Provide enough symbols in SCAN_UNIVERSE
            with patch.object(scan_mod, "SCAN_UNIVERSE", [c["symbol"] for c in fake_candidates]):
                scan_mod.main()

        # Verify outputs were written
        assert os.path.exists(str(tmp_path / "scanner" / "selected_symbols.json"))
        assert "AAPL" in env_file.read_text()

    def test_main_too_few_candidates(self, tmp_path):
        from agents import market_scanner as scan_mod

        env_file = tmp_path / ".env"
        env_file.write_text("ALPACA_API_KEY=test\n")

        with patch.object(scan_mod, "DATA_DIR", str(tmp_path)), \
             patch.object(scan_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch.object(scan_mod, "ASSIGNMENTS_FILE", str(tmp_path / "nope.json")), \
             patch.object(scan_mod, "FALLBACK_FILE", str(tmp_path / "nope2.json")), \
             patch.object(scan_mod, "LEARNINGS_FILE", str(tmp_path / "nope3.json")), \
             patch.object(scan_mod, "ENV_FILE", str(env_file)), \
             patch("agents.market_scanner.Config") as mock_config, \
             patch("agents.market_scanner.StockHistoricalDataClient"), \
             patch("agents.market_scanner.fetch_stock_data", return_value={}), \
             patch("agents.market_scanner.check_deadline", return_value=False):
            mock_config.validate.return_value = None
            mock_config.ALPACA_API_KEY = "test"
            mock_config.ALPACA_SECRET_KEY = "test"
            mock_config.CRYPTO_SYMBOLS = []
            scan_mod.main()

    def test_main_deadline_abort(self, tmp_path):
        from agents import market_scanner as scan_mod

        with patch.object(scan_mod, "DATA_DIR", str(tmp_path)), \
             patch.object(scan_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch.object(scan_mod, "ASSIGNMENTS_FILE", str(tmp_path / "nope.json")), \
             patch.object(scan_mod, "FALLBACK_FILE", str(tmp_path / "nope2.json")), \
             patch.object(scan_mod, "LEARNINGS_FILE", str(tmp_path / "nope3.json")), \
             patch("agents.market_scanner.Config") as mock_config, \
             patch("agents.market_scanner.StockHistoricalDataClient"), \
             patch("agents.market_scanner.fetch_stock_data", return_value={}), \
             patch("agents.market_scanner.check_deadline", return_value=True):
            mock_config.validate.return_value = None
            mock_config.ALPACA_API_KEY = "test"
            mock_config.ALPACA_SECRET_KEY = "test"
            mock_config.CRYPTO_SYMBOLS = []
            scan_mod.main()
