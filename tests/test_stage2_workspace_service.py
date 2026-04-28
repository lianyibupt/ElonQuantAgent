import unittest
from datetime import datetime, timezone

from services.stage2_workspace import (
    build_default_workspace,
    normalize_workspace_payload,
    recalculate_account_state,
    refresh_position_after_analysis,
)


class BuildDefaultWorkspaceTests(unittest.TestCase):
    def test_build_default_workspace_returns_expected_shape(self):
        workspace = build_default_workspace()

        self.assertEqual(workspace["positions"], [])
        self.assertEqual(workspace["account_state"]["nav"], 0.0)
        self.assertEqual(workspace["account_state"]["cash"], 0.0)
        self.assertEqual(workspace["account_state"]["current_drawdown"], 0.0)
        self.assertEqual(workspace["account_state"]["total_market_value"], 0.0)
        self.assertEqual(workspace["account_state"]["cash_pct"], 0.0)
        self.assertEqual(workspace["account_state"]["gross_exposure"], 0.0)
        self.assertEqual(workspace["account_state"]["core_exposure"], 0.0)
        self.assertEqual(workspace["account_state"]["tactical_exposure"], 0.0)
        self.assertEqual(workspace["account_state"]["largest_position"], "")
        self.assertEqual(workspace["account_state"]["largest_position_weight"], 0.0)
        self.assertEqual(workspace["account_state"]["position_count"], 0)
        self.assertIsNone(workspace["account_state"]["last_aggregated_at"])


class NormalizeWorkspacePayloadTests(unittest.TestCase):
    def test_normalize_workspace_payload_uppercases_ticker_and_sets_missing_derived_fields(self):
        workspace = normalize_workspace_payload(
            {
                "account_state": {"nav": 1000, "cash": 400, "current_drawdown": 12.0},
                "positions": [
                    {
                        "ticker": " msft ",
                        "book_type": "core",
                        "cost_basis": 250.5,
                        "shares": 2,
                        "tracking_status": "active",
                        "factor_tags": "growth, ai , quality",
                        "notes": "watch earnings",
                    }
                ],
            }
        )

        position = workspace["positions"][0]
        self.assertEqual(position["ticker"], "MSFT")
        self.assertEqual(position["factor_tags"], ["growth", "ai", "quality"])
        self.assertEqual(position["cost_basis"], 250.5)
        self.assertEqual(position["shares"], 2.0)
        self.assertEqual(position["latest_price"], 0.0)
        self.assertEqual(position["market_value"], 0.0)
        self.assertEqual(position["position_weight"], 0.0)
        self.assertEqual(position["unrealized_pnl"], 0.0)
        self.assertEqual(position["unrealized_pnl_pct"], 0.0)
        self.assertEqual(position["price_source"], "")
        self.assertIsNone(position["last_price_update_at"])
        self.assertIsNone(position["last_analyzed_at"])
        self.assertEqual(position["last_analysis_summary"], "")

    def test_normalize_workspace_payload_defensively_handles_malformed_position_items(self):
        workspace = normalize_workspace_payload(
            {
                "positions": [
                    "bad item",
                    123,
                    None,
                    {"ticker": " aapl ", "shares": "5"},
                ]
            }
        )

        self.assertEqual(len(workspace["positions"]), 4)
        self.assertEqual(workspace["positions"][0]["ticker"], "")
        self.assertEqual(workspace["positions"][1]["shares"], 0.0)
        self.assertIsNone(workspace["positions"][2]["last_price_update_at"])
        self.assertEqual(workspace["positions"][3]["ticker"], "AAPL")
        self.assertEqual(workspace["positions"][3]["shares"], 5.0)

    def test_normalize_workspace_payload_converts_timestamp_fields_to_string_or_none(self):
        timestamp = datetime(2026, 4, 28, 10, 15, tzinfo=timezone.utc)
        workspace = normalize_workspace_payload(
            {
                "positions": [
                    {
                        "ticker": "NVDA",
                        "last_price_update_at": timestamp,
                        "last_analyzed_at": 1714299300,
                    }
                ]
            }
        )

        position = workspace["positions"][0]
        self.assertEqual(position["last_price_update_at"], "2026-04-28 10:15:00+00:00")
        self.assertEqual(position["last_analyzed_at"], "1714299300")


class RecalculateAccountStateTests(unittest.TestCase):
    def test_recalculate_account_state_computes_position_and_account_aggregates_in_percent(self):
        workspace = normalize_workspace_payload(
            {
                "account_state": {"nav": 1000, "cash": 250, "current_drawdown": 8.0},
                "positions": [
                    {
                        "ticker": "AAPL",
                        "book_type": "core",
                        "cost_basis": 100,
                        "shares": 3,
                        "latest_price": 110,
                    },
                    {
                        "ticker": "TSLA",
                        "book_type": "tactical",
                        "cost_basis": 50,
                        "shares": 2,
                        "latest_price": 40,
                    },
                ],
            }
        )

        recalculated = recalculate_account_state(workspace)
        aapl, tsla = recalculated["positions"]
        account_state = recalculated["account_state"]

        self.assertEqual(aapl["market_value"], 330.0)
        self.assertEqual(aapl["unrealized_pnl"], 30.0)
        self.assertEqual(aapl["unrealized_pnl_pct"], 10.0)
        self.assertEqual(aapl["position_weight"], 33.0)
        self.assertEqual(tsla["market_value"], 80.0)
        self.assertEqual(tsla["unrealized_pnl"], -20.0)
        self.assertEqual(tsla["unrealized_pnl_pct"], -20.0)
        self.assertEqual(tsla["position_weight"], 8.0)
        self.assertEqual(account_state["total_market_value"], 410.0)
        self.assertEqual(account_state["cash_pct"], 25.0)
        self.assertEqual(account_state["gross_exposure"], 41.0)
        self.assertEqual(account_state["core_exposure"], 33.0)
        self.assertEqual(account_state["tactical_exposure"], 8.0)
        self.assertEqual(account_state["largest_position"], "AAPL")
        self.assertEqual(account_state["largest_position_weight"], 33.0)
        self.assertEqual(account_state["position_count"], 2)
        self.assertIsNotNone(account_state["last_aggregated_at"])


class RefreshPositionAfterAnalysisTests(unittest.TestCase):
    def test_refresh_position_after_analysis_updates_only_matching_position_and_returns_metadata(self):
        workspace = recalculate_account_state(
            normalize_workspace_payload(
                {
                    "account_state": {"nav": 1000, "cash": 700, "current_drawdown": 3.0},
                    "positions": [
                        {"ticker": "MSFT", "book_type": "core", "cost_basis": 100, "shares": 1},
                        {"ticker": "NVDA", "book_type": "tactical", "cost_basis": 200, "shares": 1},
                    ],
                }
            )
        )

        updated_workspace, metadata = refresh_position_after_analysis(
            workspace,
            asset="nvda",
            latest_price=220,
            analysis_summary="Momentum remains strong.",
            updated_at="2026-04-28T10:15:00Z",
        )

        msft, nvda = updated_workspace["positions"]
        self.assertEqual(msft["latest_price"], 0.0)
        self.assertEqual(msft["last_analysis_summary"], "")
        self.assertEqual(nvda["latest_price"], 220.0)
        self.assertEqual(nvda["market_value"], 220.0)
        self.assertEqual(nvda["position_weight"], 22.0)
        self.assertEqual(nvda["unrealized_pnl"], 20.0)
        self.assertEqual(nvda["unrealized_pnl_pct"], 10.0)
        self.assertEqual(nvda["price_source"], "analysis_refresh")
        self.assertEqual(nvda["last_price_update_at"], "2026-04-28T10:15:00Z")
        self.assertEqual(nvda["last_analyzed_at"], "2026-04-28T10:15:00Z")
        self.assertEqual(nvda["last_analysis_summary"], "Momentum remains strong.")
        self.assertEqual(metadata["updated_ticker"], "NVDA")
        self.assertEqual(metadata["updated_at"], "2026-04-28T10:15:00Z")
        self.assertEqual(metadata["position_index"], 1)

    def test_refresh_position_after_analysis_is_no_op_when_position_missing(self):
        workspace = recalculate_account_state(
            normalize_workspace_payload(
                {
                    "account_state": {"nav": 1000, "cash": 700, "current_drawdown": 3.0},
                    "positions": [
                        {
                            "ticker": "MSFT",
                            "book_type": "core",
                            "cost_basis": 100,
                            "shares": 1,
                            "latest_price": 110,
                        }
                    ],
                }
            )
        )
        original_aggregated_at = workspace["account_state"]["last_aggregated_at"]

        updated_workspace, metadata = refresh_position_after_analysis(
            workspace,
            asset="nvda",
            latest_price=220,
            analysis_summary="Momentum remains strong.",
            updated_at=datetime(2026, 4, 28, 10, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(updated_workspace, workspace)
        self.assertEqual(updated_workspace["account_state"]["last_aggregated_at"], original_aggregated_at)
        self.assertEqual(metadata["updated_ticker"], "NVDA")
        self.assertEqual(metadata["updated_at"], "2026-04-28 10:15:00+00:00")
        self.assertIsNone(metadata["position_index"])


if __name__ == "__main__":
    unittest.main()
