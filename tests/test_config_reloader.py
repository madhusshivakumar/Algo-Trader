"""Tests for core/config_reloader.py — config hot-reload."""

import os
import time
import pytest
from unittest.mock import patch

from config import Config
from core.config_reloader import ConfigReloader, RELOADABLE_KEYS


@pytest.fixture
def env_file(tmp_path):
    env_path = str(tmp_path / ".env")
    with open(env_path, "w") as f:
        f.write("MAX_POSITION_PCT=0.50\n")
        f.write("STOP_LOSS_PCT=0.025\n")
        f.write("SENTIMENT_ENABLED=false\n")
        f.write("ALPACA_API_KEY=secret123\n")  # Not reloadable
    return env_path


@pytest.fixture
def reloader(env_file):
    return ConfigReloader(env_path=env_file)


class TestCheckForChanges:
    def test_no_changes(self, reloader):
        assert reloader.check_for_changes() is False

    def test_detects_change(self, reloader, env_file):
        # Modify the file
        time.sleep(0.1)  # Ensure mtime differs
        with open(env_file, "a") as f:
            f.write("DAILY_DRAWDOWN_LIMIT=0.15\n")
        assert reloader.check_for_changes() is True

    def test_nonexistent_file(self, tmp_path):
        reloader = ConfigReloader(env_path=str(tmp_path / "nonexistent"))
        assert reloader.check_for_changes() is False


class TestReload:
    def test_reload_changes_config(self, reloader, env_file):
        with open(env_file, "w") as f:
            f.write("MAX_POSITION_PCT=0.30\n")
            f.write("STOP_LOSS_PCT=0.05\n")

        original_max = Config.MAX_POSITION_PCT
        original_stop = Config.STOP_LOSS_PCT
        try:
            changed = reloader.reload()
            assert "MAX_POSITION_PCT" in changed
            assert "STOP_LOSS_PCT" in changed
            assert Config.MAX_POSITION_PCT == 0.30
        finally:
            Config.MAX_POSITION_PCT = original_max
            Config.STOP_LOSS_PCT = original_stop

    def test_reload_ignores_non_reloadable(self, reloader, env_file):
        with open(env_file, "w") as f:
            f.write("ALPACA_API_KEY=new_key\n")
            f.write("MAX_POSITION_PCT=0.30\n")

        original = Config.MAX_POSITION_PCT
        try:
            changed = reloader.reload()
            assert "ALPACA_API_KEY" not in changed
        finally:
            Config.MAX_POSITION_PCT = original

    def test_reload_handles_bool(self, reloader, env_file):
        with open(env_file, "w") as f:
            f.write("SENTIMENT_ENABLED=true\n")

        original = Config.SENTIMENT_ENABLED
        try:
            changed = reloader.reload()
            if not original:  # only expect change if was false
                assert "SENTIMENT_ENABLED" in changed
        finally:
            Config.SENTIMENT_ENABLED = original

    def test_reload_no_changes(self, reloader, env_file):
        # Write current values to .env
        with open(env_file, "w") as f:
            f.write(f"MAX_POSITION_PCT={Config.MAX_POSITION_PCT}\n")

        changed = reloader.reload()
        assert "MAX_POSITION_PCT" not in changed

    def test_reload_nonexistent_file(self, tmp_path):
        reloader = ConfigReloader(env_path=str(tmp_path / "nonexistent"))
        changed = reloader.reload()
        assert changed == {}

    def test_reload_updates_last_reload_time(self, reloader):
        assert reloader.last_reload_time is None
        reloader.reload()
        assert reloader.last_reload_time is not None

    def test_reload_skips_comments_and_blanks(self, reloader, env_file):
        with open(env_file, "w") as f:
            f.write("# Comment line\n")
            f.write("\n")
            f.write("MAX_POSITION_PCT=0.99\n")
            f.write("  \n")

        original = Config.MAX_POSITION_PCT
        try:
            changed = reloader.reload()
            assert "MAX_POSITION_PCT" in changed
        finally:
            Config.MAX_POSITION_PCT = original

    def test_reload_handles_invalid_value(self, reloader, env_file):
        with open(env_file, "w") as f:
            f.write("MAX_POSITION_PCT=not_a_number\n")

        original = Config.MAX_POSITION_PCT
        try:
            changed = reloader.reload()
            assert "MAX_POSITION_PCT" not in changed
        finally:
            Config.MAX_POSITION_PCT = original


class TestGetReloadableKeys:
    def test_returns_sorted_list(self, reloader):
        keys = reloader.get_reloadable_keys()
        assert isinstance(keys, list)
        assert keys == sorted(keys)
        assert len(keys) > 0

    def test_api_keys_not_reloadable(self):
        assert "ALPACA_API_KEY" not in RELOADABLE_KEYS
        assert "ALPACA_SECRET_KEY" not in RELOADABLE_KEYS
        assert "ANTHROPIC_API_KEY" not in RELOADABLE_KEYS

    def test_risk_params_reloadable(self):
        assert "MAX_POSITION_PCT" in RELOADABLE_KEYS
        assert "STOP_LOSS_PCT" in RELOADABLE_KEYS
        assert "DAILY_DRAWDOWN_LIMIT" in RELOADABLE_KEYS
