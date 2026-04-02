"""Tests for LLM Analyst v2 — big-picture macro → sector → symbol pipeline."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


# ── Helpers ─────────────────────────────────────────────────────

def _mock_llm_json(parsed_response):
    """Create a mock call_llm_json return value."""
    return {"parsed": parsed_response, "text": json.dumps(parsed_response),
            "model": "test", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001}


def _make_macro_context(**overrides):
    base = {
        "regime": "risk-on",
        "fed_stance": "neutral",
        "key_events": ["Strong tech earnings"],
        "geopolitical_risk": "low",
        "summary": "Broadly bullish market with tech leading.",
        "sector_implications": {"Tech": "bullish", "Finance": "neutral"},
    }
    base.update(overrides)
    return base


def _make_sector_context(sector="Tech", **overrides):
    base = {
        "sector": sector,
        "outlook": "bullish",
        "conviction": 0.6,
        "drivers": ["AI spending strong"],
        "cross_sector_effects": {},
        "risk_level": "low",
        "reasoning": "Tech sector strong on AI tailwinds.",
    }
    base.update(overrides)
    return base


def _make_conviction(**overrides):
    base = {
        "conviction_score": 0.5,
        "bias": "bullish",
        "confidence": 0.7,
        "key_factors": ["momentum"],
        "risk_flags": [],
        "reasoning": "Test conviction.",
    }
    base.update(overrides)
    return base


# ── Test Constants ──────────────────────────────────────────────

class TestDataStructures:
    def test_sector_etf_map_has_all_sectors(self):
        from agents.llm_analyst_v2 import SECTOR_ETF_MAP
        for sector in ["Tech", "Finance", "Energy", "Healthcare", "Consumer", "Industrial"]:
            assert sector in SECTOR_ETF_MAP

    def test_sector_dependencies_valid(self):
        from agents.llm_analyst_v2 import SECTOR_DEPENDENCIES
        for source, impacts in SECTOR_DEPENDENCIES.items():
            for target, weight in impacts.items():
                assert -1.0 <= weight <= 1.0, f"{source} -> {target}: {weight}"
                assert source != target, f"Self-dependency: {source}"

    def test_historical_patterns_not_empty(self):
        from agents.llm_analyst_v2 import HISTORICAL_PATTERNS
        assert len(HISTORICAL_PATTERNS) >= 5
        for p in HISTORICAL_PATTERNS:
            assert "trigger" in p
            assert "effects" in p

    def test_symbol_sector_map_covers_common_symbols(self):
        from agents.llm_analyst_v2 import SYMBOL_SECTOR_MAP
        for sym in ["AAPL", "TSLA", "XOM", "JPM", "BTC/USD"]:
            assert sym in SYMBOL_SECTOR_MAP


# ── Stage 1: Macro Scan ────────────────────────────────────────

class TestAnalyzeMacro:
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_success(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_macro
        macro = _make_macro_context()
        mock_llm.return_value = _mock_llm_json(macro)

        result = analyze_macro(fetch_fn=lambda sym, lookback_hours=24: ["headline1"])
        assert result["regime"] == "risk-on"
        assert result["fed_stance"] == "neutral"
        assert mock_llm.called

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_no_headlines_returns_default(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_macro
        result = analyze_macro(fetch_fn=lambda sym, lookback_hours=24: [])
        assert result["regime"] == "neutral"
        assert not mock_llm.called

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_api_failure_returns_default(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_macro
        mock_llm.side_effect = RuntimeError("API error")
        result = analyze_macro(fetch_fn=lambda sym, lookback_hours=24: ["headline"])
        assert result["regime"] == "neutral"

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_deduplicates_headlines(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_macro
        mock_llm.return_value = _mock_llm_json(_make_macro_context())

        # Same headline from multiple ETFs
        result = analyze_macro(fetch_fn=lambda sym, lookback_hours=24: ["same headline"])
        # Should deduplicate — prompt should only have 1 headline
        call_args = mock_llm.call_args
        prompt = call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
        assert prompt.count("same headline") == 1

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_fetches_from_three_etfs(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_macro
        mock_llm.return_value = _mock_llm_json(_make_macro_context())

        fetched_symbols = []
        def track_fetch(sym, lookback_hours=24):
            fetched_symbols.append(sym)
            return [f"headline for {sym}"]

        analyze_macro(fetch_fn=track_fetch)
        assert "SPY" in fetched_symbols
        assert "QQQ" in fetched_symbols
        assert "DIA" in fetched_symbols


# ── Stage 2: Sector Analysis ───────────────────────────────────

class TestAnalyzeSector:
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_success(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_sector
        sector_ctx = _make_sector_context("Energy")
        mock_llm.return_value = _mock_llm_json(sector_ctx)
        macro = _make_macro_context()

        result = analyze_sector("Energy", macro, fetch_fn=lambda sym, lookback_hours=24: ["oil up"])
        assert result["sector"] == "Energy"
        assert result["outlook"] == "bullish"

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_api_failure_returns_default(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_sector
        mock_llm.side_effect = RuntimeError("API down")
        macro = _make_macro_context()

        result = analyze_sector("Tech", macro, fetch_fn=lambda sym, lookback_hours=24: [])
        assert result["sector"] == "Tech"
        assert result["outlook"] == "neutral"

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_conviction_clamped(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_sector
        mock_llm.return_value = _mock_llm_json({"conviction": 5.0, "sector": "Tech"})
        macro = _make_macro_context()

        result = analyze_sector("Tech", macro, fetch_fn=lambda sym, lookback_hours=24: ["h"])
        assert result["conviction"] <= 1.0

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_fetches_correct_etf(self, mock_llm):
        from agents.llm_analyst_v2 import analyze_sector
        mock_llm.return_value = _mock_llm_json(_make_sector_context("Energy"))
        macro = _make_macro_context()

        fetched = []
        def track(sym, lookback_hours=24):
            fetched.append(sym)
            return []

        analyze_sector("Energy", macro, fetch_fn=track)
        assert "XLE" in fetched


class TestAnalyzeAllSectors:
    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_analyzes_relevant_sectors_only(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_all_sectors
        mock_llm.return_value = _mock_llm_json(_make_sector_context())
        macro = _make_macro_context()

        symbols = ["AAPL", "XOM"]  # Tech + Energy
        result = analyze_all_sectors(macro, symbols,
                                     fetch_fn=lambda sym, lookback_hours=24: [])
        assert "Tech" in result
        assert "Energy" in result
        assert "Finance" not in result

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=999.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_stops_on_budget(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_all_sectors
        macro = _make_macro_context()
        result = analyze_all_sectors(macro, ["AAPL", "XOM"],
                                     fetch_fn=lambda sym, lookback_hours=24: [])
        assert len(result) == 0
        assert not mock_llm.called


# ── Stage 3: Symbol Conviction ─────────────────────────────────

class TestAnalyzeSymbol:
    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_success(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        conv = _make_conviction()
        mock_llm.return_value = _mock_llm_json(conv)
        macro = _make_macro_context()
        sectors = {"Tech": _make_sector_context("Tech")}

        result = analyze_symbol("AAPL", macro, sectors,
                                headlines=["Apple earnings beat"],
                                indicators={"RSI": 45})
        assert result["conviction_score"] == 0.5
        assert result["bias"] == "bullish"

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=999.0)
    def test_budget_exceeded_returns_none(self, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        result = analyze_symbol("AAPL", {}, {})
        assert result is None

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_api_failure_returns_neutral(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        mock_llm.side_effect = RuntimeError("fail")
        result = analyze_symbol("AAPL", {}, {}, headlines=[])
        assert result["conviction_score"] == 0
        assert "analysis_failed" in result["risk_flags"]

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_conviction_clamped(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        mock_llm.return_value = _mock_llm_json({"conviction_score": 3.0})
        result = analyze_symbol("AAPL", {}, {}, headlines=[])
        assert result["conviction_score"] <= 1.0

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_fetches_headlines_when_not_provided(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        mock_llm.return_value = _mock_llm_json(_make_conviction())

        fetched = []
        def track(sym, lookback_hours=24):
            fetched.append(sym)
            return ["headline"]

        analyze_symbol("AAPL", {}, {}, fetch_fn=track)
        assert "AAPL" in fetched

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_no_sector_context_still_works(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import analyze_symbol
        mock_llm.return_value = _mock_llm_json(_make_conviction())
        result = analyze_symbol("UNKNOWN", {}, {}, headlines=[])
        assert result["conviction_score"] == 0.5


# ── Stage 4: Portfolio Synthesis ───────────────────────────────

class TestSynthesizePortfolio:
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_success(self, mock_llm):
        from agents.llm_analyst_v2 import synthesize_portfolio
        adjusted = {
            "adjusted_convictions": {"AAPL": 0.4, "XOM": -0.2},
            "portfolio_risk": "medium",
            "concentration_warnings": ["Tech heavy"],
            "recommendations": ["Reduce tech exposure"],
            "overall_bias": "bullish",
            "reasoning": "Portfolio is tech-heavy.",
        }
        mock_llm.return_value = _mock_llm_json(adjusted)

        macro = _make_macro_context()
        sectors = {"Tech": _make_sector_context()}
        convictions = {
            "AAPL": _make_conviction(conviction_score=0.6),
            "XOM": _make_conviction(conviction_score=-0.3),
        }

        result = synthesize_portfolio(macro, sectors, convictions)
        assert result["adjusted_convictions"]["AAPL"] == 0.4
        assert result["portfolio_risk"] == "medium"

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_api_failure_uses_raw_scores(self, mock_llm):
        from agents.llm_analyst_v2 import synthesize_portfolio
        mock_llm.side_effect = RuntimeError("fail")

        convictions = {
            "AAPL": _make_conviction(conviction_score=0.6),
        }
        result = synthesize_portfolio({}, {}, convictions)
        assert result["adjusted_convictions"]["AAPL"] == 0.6

    def test_empty_convictions(self):
        from agents.llm_analyst_v2 import synthesize_portfolio
        result = synthesize_portfolio({}, {}, {})
        assert result["adjusted_convictions"] == {}
        assert result["portfolio_risk"] == "medium"

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_missing_symbol_in_adjustment_uses_raw(self, mock_llm):
        from agents.llm_analyst_v2 import synthesize_portfolio
        # Sonnet only adjusts AAPL, not XOM
        mock_llm.return_value = _mock_llm_json({
            "adjusted_convictions": {"AAPL": 0.3},
            "portfolio_risk": "low",
        })
        convictions = {
            "AAPL": _make_conviction(conviction_score=0.5),
            "XOM": _make_conviction(conviction_score=0.4),
        }
        result = synthesize_portfolio({}, {}, convictions)
        assert result["adjusted_convictions"]["AAPL"] == 0.3
        assert result["adjusted_convictions"]["XOM"] == 0.4  # fallback to raw

    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_adjusted_scores_clamped(self, mock_llm):
        from agents.llm_analyst_v2 import synthesize_portfolio
        mock_llm.return_value = _mock_llm_json({
            "adjusted_convictions": {"AAPL": 5.0},
        })
        convictions = {"AAPL": _make_conviction()}
        result = synthesize_portfolio({}, {}, convictions)
        assert result["adjusted_convictions"]["AAPL"] <= 1.0


# ── Prompt Building ────────────────────────────────────────────

class TestPromptBuilding:
    def test_sector_prompt_includes_macro(self):
        from agents.llm_analyst_v2 import _build_sector_prompt
        macro = _make_macro_context(summary="Fed is dovish")
        prompt = _build_sector_prompt("Tech", macro, ["AI rally"])
        assert "Fed is dovish" in prompt
        assert "AI rally" in prompt

    def test_sector_prompt_includes_dependencies(self):
        from agents.llm_analyst_v2 import _build_sector_prompt
        macro = _make_macro_context()
        # Tech is affected by Finance (rate sensitivity)
        prompt = _build_sector_prompt("Tech", macro, [])
        assert "Finance" in prompt or "Cross-Sector" in prompt

    def test_sector_prompt_includes_historical_patterns(self):
        from agents.llm_analyst_v2 import _build_sector_prompt
        macro = _make_macro_context()
        prompt = _build_sector_prompt("Tech", macro, [])
        assert "Historical" in prompt or "When" in prompt

    def test_symbol_prompt_includes_all_context(self):
        from agents.llm_analyst_v2 import _build_symbol_prompt
        macro = _make_macro_context(summary="Risk-on environment")
        sector = _make_sector_context("Tech", reasoning="AI boom")
        prompt = _build_symbol_prompt("AAPL", macro, sector,
                                      ["Apple earnings"], {"RSI": 35})
        assert "Risk-on environment" in prompt
        assert "AI boom" in prompt
        assert "Apple earnings" in prompt
        assert "RSI" in prompt

    def test_symbol_prompt_no_sector_context(self):
        from agents.llm_analyst_v2 import _build_symbol_prompt
        macro = _make_macro_context()
        prompt = _build_symbol_prompt("AAPL", macro, None, [], {})
        assert "AAPL" in prompt
        assert "No symbol-specific news" in prompt

    def test_symbol_prompt_no_headlines(self):
        from agents.llm_analyst_v2 import _build_symbol_prompt
        macro = _make_macro_context()
        prompt = _build_symbol_prompt("AAPL", macro, None, [], {})
        assert "No symbol-specific news" in prompt


# ── Orchestrator ───────────────────────────────────────────────

class TestRunAnalysis:
    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_full_pipeline(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import run_analysis

        call_count = [0]
        def mock_response(*args, **kwargs):
            call_count[0] += 1
            system = kwargs.get("system", "")
            if "macro strategist" in system:
                return _mock_llm_json(_make_macro_context())
            elif "sector analyst" in system:
                return _mock_llm_json(_make_sector_context())
            elif "equity analyst" in system:
                return _mock_llm_json(_make_conviction())
            elif "portfolio risk" in system:
                return _mock_llm_json({
                    "adjusted_convictions": {"AAPL": 0.4},
                    "portfolio_risk": "low",
                    "overall_bias": "bullish",
                })
            return _mock_llm_json({})

        mock_llm.side_effect = mock_response

        result = run_analysis(
            symbols=["AAPL"],
            fetch_fn=lambda sym, lookback_hours=24: ["test headline"],
        )

        assert "convictions" in result
        assert "AAPL" in result["convictions"]
        assert result["version"] == "v2"
        assert "portfolio_risk" in result
        # macro(1) + sector(1 for Tech) + symbol(1) + portfolio(1) = 4 calls
        assert call_count[0] == 4

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_output_format_compatible_with_signal_modifiers(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import run_analysis

        mock_llm.return_value = _mock_llm_json(_make_macro_context())
        # Override for each call type
        def mock_response(*args, **kwargs):
            system = kwargs.get("system", "")
            if "macro" in system:
                return _mock_llm_json(_make_macro_context())
            elif "sector" in system:
                return _mock_llm_json(_make_sector_context())
            elif "equity" in system:
                return _mock_llm_json(_make_conviction(conviction_score=0.7))
            elif "portfolio" in system:
                return _mock_llm_json({
                    "adjusted_convictions": {"AAPL": 0.65},
                    "portfolio_risk": "low",
                })
            return _mock_llm_json({})

        mock_llm.side_effect = mock_response

        result = run_analysis(
            symbols=["AAPL"],
            fetch_fn=lambda sym, lookback_hours=24: [],
        )

        # signal_modifiers.py reads: data["convictions"][symbol]["score"]
        assert "score" in result["convictions"]["AAPL"]
        assert isinstance(result["convictions"]["AAPL"]["score"], (int, float))

    @patch("agents.llm_analyst_v2.get_daily_spend", return_value=0.0)
    @patch("agents.llm_analyst_v2.call_llm_json")
    def test_includes_raw_and_adjusted_scores(self, mock_llm, mock_spend):
        from agents.llm_analyst_v2 import run_analysis

        def mock_response(*args, **kwargs):
            system = kwargs.get("system", "")
            if "macro strategist" in system:
                return _mock_llm_json(_make_macro_context())
            elif "sector analyst" in system:
                return _mock_llm_json(_make_sector_context())
            elif "equity analyst" in system:
                return _mock_llm_json(_make_conviction(conviction_score=0.8))
            elif "portfolio risk" in system:
                return _mock_llm_json({
                    "adjusted_convictions": {"AAPL": 0.5},
                    "portfolio_risk": "medium",
                })
            return _mock_llm_json({})

        mock_llm.side_effect = mock_response

        result = run_analysis(
            symbols=["AAPL"],
            fetch_fn=lambda sym, lookback_hours=24: ["test headline"],
        )

        aapl = result["convictions"]["AAPL"]
        assert aapl["raw_score"] == 0.8
        assert aapl["score"] == 0.5  # adjusted by portfolio synthesis


# ── Write Outputs ──────────────────────────────────────────────

class TestWriteOutputs:
    def test_writes_all_files(self, tmp_path):
        from agents.llm_analyst_v2 import write_outputs
        with patch("agents.llm_analyst_v2._DATA_DIR", str(tmp_path)), \
             patch("agents.llm_analyst_v2._CONVICTIONS_FILE", str(tmp_path / "convictions.json")), \
             patch("agents.llm_analyst_v2._MACRO_FILE", str(tmp_path / "macro_report.json")), \
             patch("agents.llm_analyst_v2._SECTOR_FILE", str(tmp_path / "sector_report.json")):

            data = {"convictions": {"AAPL": {"score": 0.5}}, "timestamp": "now"}
            macro = _make_macro_context()
            sectors = {"Tech": _make_sector_context()}

            write_outputs(data, macro, sectors)

            assert os.path.exists(tmp_path / "convictions.json")
            assert os.path.exists(tmp_path / "macro_report.json")
            assert os.path.exists(tmp_path / "sector_report.json")

            with open(tmp_path / "convictions.json") as f:
                loaded = json.load(f)
            assert loaded["convictions"]["AAPL"]["score"] == 0.5

            with open(tmp_path / "macro_report.json") as f:
                loaded = json.load(f)
            assert loaded["regime"] == "risk-on"

            with open(tmp_path / "sector_report.json") as f:
                loaded = json.load(f)
            assert "Tech" in loaded["sectors"]


# ── Feature Flag Delegation ────────────────────────────────────

class TestFeatureFlagDelegation:
    @patch("agents.llm_analyst.Config")
    def test_v2_flag_delegates(self, mock_config):
        mock_config.LLM_ANALYST_V2_ENABLED = True

        with patch("agents.llm_analyst_v2.main") as mock_v2_main:
            mock_v2_main.return_value = {"convictions": {}}
            from agents.llm_analyst import main
            result = main()
            mock_v2_main.assert_called_once()

    @patch("agents.llm_analyst.Config")
    def test_v1_runs_when_v2_disabled(self, mock_config):
        mock_config.LLM_ANALYST_V2_ENABLED = False
        mock_config.LLM_ANALYST_ENABLED = True
        mock_config.ANTHROPIC_API_KEY = ""  # will exit early

        from agents.llm_analyst import main
        result = main()
        assert result is None  # exits because no API key
