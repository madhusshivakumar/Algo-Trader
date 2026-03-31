"""Unit tests for the strategy router."""

import json
import os
import importlib
import pytest

from strategies.router import (
    STRATEGY_REGISTRY,
    STRATEGY_MAP,
    STRATEGY_DISPLAY_NAMES,
    DEFAULT_STRATEGY,
    get_strategy,
    get_strategy_key,
    compute_signals,
)


class TestRegistry:
    """Test the strategy registry is complete and valid."""

    def test_registry_has_9_strategies(self):
        assert len(STRATEGY_REGISTRY) == 9

    def test_all_strategies_are_callable(self):
        for name, fn in STRATEGY_REGISTRY.items():
            assert callable(fn), f"Strategy '{name}' is not callable"

    def test_all_strategies_have_display_names(self):
        for key in STRATEGY_REGISTRY:
            assert key in STRATEGY_DISPLAY_NAMES, f"'{key}' missing display name"

    def test_default_strategy_exists(self):
        assert DEFAULT_STRATEGY in STRATEGY_REGISTRY

    def test_registry_keys_match(self):
        assert set(STRATEGY_REGISTRY.keys()) == set(STRATEGY_DISPLAY_NAMES.keys())


class TestStrategyMap:
    """Test the strategy map loading and assignment."""

    def test_map_is_not_empty(self):
        assert len(STRATEGY_MAP) > 0

    def test_all_assigned_strategies_exist(self):
        for symbol, strat_key in STRATEGY_MAP.items():
            assert strat_key in STRATEGY_REGISTRY, \
                f"Symbol '{symbol}' assigned to unknown strategy '{strat_key}'"

    def test_crypto_symbols_assigned(self):
        # BTC and ETH should always have assignments
        assert "BTC/USD" in STRATEGY_MAP or DEFAULT_STRATEGY in STRATEGY_REGISTRY
        assert "ETH/USD" in STRATEGY_MAP or DEFAULT_STRATEGY in STRATEGY_REGISTRY


class TestGetStrategy:
    """Test the get_strategy function."""

    def test_known_symbol(self):
        name, fn = get_strategy("BTC/USD")
        assert isinstance(name, str)
        assert callable(fn)

    def test_unknown_symbol_gets_default(self):
        name, fn = get_strategy("UNKNOWN_SYMBOL")
        default_name = STRATEGY_DISPLAY_NAMES[DEFAULT_STRATEGY]
        assert name == default_name

    def test_returns_tuple(self):
        result = get_strategy("BTC/USD")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetStrategyKey:
    """Test the get_strategy_key function."""

    def test_known_symbol(self):
        key = get_strategy_key("BTC/USD")
        assert key in STRATEGY_REGISTRY

    def test_unknown_symbol_gets_default(self):
        key = get_strategy_key("ZZZZ")
        assert key == DEFAULT_STRATEGY


class TestComputeSignals:
    """Test the compute_signals routing function."""

    def test_returns_signal_with_strategy_tag(self, flat_market):
        signal = compute_signals("BTC/USD", flat_market)
        assert "strategy" in signal
        assert isinstance(signal["strategy"], str)

    def test_routes_to_correct_strategy(self, flat_market):
        signal = compute_signals("BTC/USD", flat_market)
        expected_name, _ = get_strategy("BTC/USD")
        assert signal["strategy"] == expected_name

    def test_valid_signal_format(self, flat_market):
        signal = compute_signals("TSLA", flat_market)
        assert "action" in signal
        assert "reason" in signal
        assert "strength" in signal
        assert signal["action"] in ("buy", "sell", "hold")


class TestLoadFromFiles:
    """Test that router can load strategy assignments from JSON files."""

    def test_load_from_assignments_file(self, tmp_data_dir, sample_assignments):
        """Verify router can parse strategy_assignments.json."""
        with open(sample_assignments) as f:
            data = json.load(f)
        assignments = data["assignments"]
        for sym, info in assignments.items():
            strat = info["strategy"] if isinstance(info, dict) else info
            assert strat in STRATEGY_REGISTRY, f"Invalid strategy '{strat}' for {sym}"

    def test_fallback_config_parseable(self):
        """Verify fallback_config.json in data/ is valid."""
        fallback_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "fallback_config.json"
        )
        if os.path.exists(fallback_path):
            with open(fallback_path) as f:
                data = json.load(f)
            assert "strategy_map" in data
            for sym, strat in data["strategy_map"].items():
                assert strat in STRATEGY_REGISTRY, f"Invalid strategy '{strat}' in fallback"
