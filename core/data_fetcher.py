"""Parallel data fetcher — fetches bars for multiple symbols concurrently.

Uses ThreadPoolExecutor since the Alpaca SDK is synchronous.
Default 5 workers keeps us under Alpaca's rate limits (~200 req/min).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from utils.logger import log


class DataFetcher:
    def __init__(self, broker, max_workers: int = 5):
        self.broker = broker
        self.max_workers = max_workers

    def fetch_recent_bars_batch(self, symbols: list[str],
                                limit: int = 100) -> dict[str, pd.DataFrame]:
        """Fetch recent bars for all symbols in parallel.

        Returns {symbol: DataFrame}. Failed fetches return empty DataFrame.
        """
        if not symbols:
            return {}

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_recent, symbol, limit): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    log.error(f"Failed to fetch bars for {symbol}: {e}")
                    results[symbol] = pd.DataFrame()

        return results

    def fetch_historical_bars_batch(self, symbols: list[str],
                                    days: int = 30) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for all symbols in parallel.

        Returns {symbol: DataFrame}. Failed fetches return empty DataFrame.
        """
        if not symbols:
            return {}

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_historical, symbol, days): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    log.error(f"Failed to fetch historical bars for {symbol}: {e}")
                    results[symbol] = pd.DataFrame()

        return results

    def _fetch_recent(self, symbol: str, limit: int) -> pd.DataFrame:
        return self.broker.get_recent_bars(symbol, limit=limit)

    def _fetch_historical(self, symbol: str, days: int) -> pd.DataFrame:
        return self.broker.get_historical_bars(symbol, days=days)
