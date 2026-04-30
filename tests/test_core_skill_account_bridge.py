import unittest
from datetime import datetime, timezone

import pandas as pd

from services.core_skill_account_bridge import build_core_skill_block


class CoreSkillAccountBridgeTests(unittest.TestCase):
    def test_build_core_skill_block_groups_actions_and_scores_for_all_positions(self):
        workspace = {
            "account_state": {"nav": 100000, "cash": 20000},
            "positions": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            "candidates": [],
        }

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28", "2026-04-29"]),
                "Open": [100, 101],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
            }
        )

        fetch_calls = []

        def fetch_market_data(symbol, interval, start_dt, end_dt, market_data_source=None):
            fetch_calls.append((symbol, interval, market_data_source))
            return market_df

        def analyze_position(ticker, df, workspace_payload):
            return {
                "single_name_score": {
                    "decision": "持有",
                    "recommended_action": "观察",
                    "trend_score": 70,
                    "entry_score": 55,
                    "volatility_score": 45,
                    "risk_reward_ratio": "1.5:1",
                    "suggested_position_range": "3%-5%",
                    "justification": f"{ticker} 结构正常",
                }
            }

        block = build_core_skill_block(
            workspace_name="default",
            workspace=workspace,
            fetch_market_data=fetch_market_data,
            analyze_position=analyze_position,
            market_data_source="yfinance",
            now=datetime(2026, 4, 29, tzinfo=timezone.utc),
        )

        self.assertEqual(block["summary"]["total_positions"], 2)
        self.assertEqual(block["summary"]["processed_positions"], 2)
        self.assertEqual(block["summary"]["timeframe"], "1d")
        self.assertEqual(block["summary"]["lookback_days"], 90)
        self.assertEqual(len(block["score_table"]), 2)
        self.assertEqual(fetch_calls[0][1], "1d")

    def test_build_core_skill_block_raises_when_any_position_fails(self):
        workspace = {"account_state": {}, "positions": [{"ticker": "AAPL"}], "candidates": []}

        def fetch_market_data(symbol, interval, start_dt, end_dt, market_data_source=None):
            return pd.DataFrame()

        def analyze_position(ticker, df, workspace_payload):
            return {}

        with self.assertRaises(ValueError):
            build_core_skill_block(
                workspace_name="default",
                workspace=workspace,
                fetch_market_data=fetch_market_data,
                analyze_position=analyze_position,
                market_data_source="yfinance",
            )


if __name__ == "__main__":
    unittest.main()
