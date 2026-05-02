import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services.database import DatabaseManager
import services.database as database_module
import web.web_interface_new as web_interface_new


class AccountAnalysisApiTests(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.test_db_manager = DatabaseManager(self.db_path)

        self.original_web_db_manager = web_interface_new.db_manager
        self.original_database_singleton = database_module._db_manager
        self.original_account_analysis_dir = web_interface_new.ACCOUNT_ANALYSIS_DIR

        self.temp_artifact_dir = Path(tempfile.mkdtemp()) / "account-analysis"

        web_interface_new.db_manager = self.test_db_manager
        database_module._db_manager = self.test_db_manager
        web_interface_new.ACCOUNT_ANALYSIS_DIR = self.temp_artifact_dir

        web_interface_new.app.config["TESTING"] = True
        self.client = web_interface_new.app.test_client()

    def tearDown(self):
        web_interface_new.db_manager = self.original_web_db_manager
        database_module._db_manager = self.original_database_singleton
        web_interface_new.ACCOUNT_ANALYSIS_DIR = self.original_account_analysis_dir
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        shutil.rmtree(self.temp_artifact_dir.parent, ignore_errors=True)

    def test_post_account_analysis_refreshes_prices_runs_llm_and_saves_artifacts(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 400, "current_drawdown": 3.5},
                "positions": [
                    {
                        "ticker": "AAPL",
                        "book_type": "core",
                        "cost_basis": 100,
                        "shares": 2,
                        "tracking_status": "active",
                    },
                    {
                        "ticker": "MSFT",
                        "book_type": "tactical",
                        "cost_basis": 50,
                        "shares": 4,
                        "tracking_status": "watch",
                    },
                ],
                "candidates": [{"ticker": "NVDA", "factor_tags": ["ai"], "notes": "watch"}],
                "notes": "focus on concentration",
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-27", "2026-04-28"]),
                "Open": [118.0, 119.0],
                "High": [121.0, 123.0],
                "Low": [117.0, 118.0],
                "Close": [120.0, 125.0],
                "Volume": [1000, 1200],
            }
        )

        llm_payload = {
            "summary": "Portfolio remains constructive but concentration is rising.",
            "portfolio_health_score": 78,
            "holding_health": [{"ticker": "AAPL", "status": "healthy", "summary": "Trend intact."}],
            "pnl_breakdown": {"winners": ["AAPL"], "losers": []},
            "concentration_risks": ["AAPL exceeds target weight"],
            "crowded_exposures": ["AI megacaps"],
            "manager_actions": [{"action": "Trim AAPL", "reason": "Weight is above plan."}],
        }

        fake_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": json.dumps(llm_payload, ensure_ascii=False)})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {
                "total_positions": 2,
                "processed_positions": 2,
                "timeframe": "1d",
                "lookback_days": 90,
            },
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [
                {"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"},
                {"ticker": "MSFT", "recommended_action": "观察", "trend_score": 68, "entry_score": 58, "volatility_score": 42, "risk_reward_ratio": "1.4:1", "suggested_position_range": "2%-4%", "decision": "持有", "justification": "趋势稳定"},
            ],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.return_value = fake_response

            response = self.client.post(
                "/api/account-analysis",
                json={"workspace_name": "default", "market_data_source": "yfinance"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["workspace_name"], "default")
        self.assertEqual(payload["analysis"]["summary"], llm_payload["summary"])
        self.assertEqual(payload["analysis"]["portfolio_health_score"], 78.0)
        self.assertIn("core_skill_block", payload)
        self.assertEqual(payload["core_skill_block"]["summary"]["timeframe"], "1d")
        self.assertEqual(payload["core_skill_block"]["summary"]["lookback_days"], 90)
        self.assertEqual(payload["core_skill_block"]["summary"]["processed_positions"], 2)
        self.assertIn("score_table", payload["core_skill_block"])
        request_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertIn("中文", request_kwargs["messages"][0]["content"])
        self.assertEqual(payload["price_refresh"]["updated_count"], 2)
        self.assertEqual(len(payload["price_refresh"]["updated_positions"]), 2)
        self.assertEqual(payload["artifacts"]["workspace_name"], "default")
        self.assertEqual(payload["artifacts"]["summary"], llm_payload["summary"])
        self.assertIn("# 账户分析", payload["analysis_markdown"])
        self.assertIn("Trim AAPL", payload["analysis_markdown"])

        with open(payload["artifacts"]["json_path"], "r", encoding="utf-8") as handle:
            saved_payload = json.load(handle)
        self.assertEqual(saved_payload["summary"], llm_payload["summary"])
        self.assertEqual(saved_payload["workspace_name"], "default")

        saved_workspace = self.test_db_manager.get_stage2_workspace("default")
        self.assertEqual(saved_workspace["positions"][0]["latest_price"], 125.0)
        self.assertEqual(saved_workspace["positions"][0]["market_value"], 250.0)
        self.assertEqual(saved_workspace["positions"][0]["unrealized_pnl"], 50.0)
        self.assertEqual(saved_workspace["positions"][1]["latest_price"], 125.0)
        self.assertEqual(saved_workspace["account_state"]["position_count"], 2)
        self.assertEqual(saved_workspace["account_state"]["largest_position"], "MSFT")

    def test_post_account_analysis_returns_404_when_workspace_missing(self):
        response = self.client.post("/api/account-analysis", json={"workspace_name": "missing"})

        self.assertEqual(response.status_code, 404)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("Workspace not found", payload["error"])

    def test_post_account_analysis_recovers_when_llm_output_has_single_json_comma_error(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 500},
                "positions": [{"ticker": "AAPL", "book_type": "core", "cost_basis": 100, "shares": 1}],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [119.0],
                "High": [123.0],
                "Low": [118.0],
                "Close": [125.0],
                "Volume": [1200],
            }
        )

        malformed_but_recoverable_json = """{
  \"summary\": \"Portfolio check\",
  \"portfolio_health_score\": 80,
  \"holding_health\": [
    {\"ticker\": \"AAPL\", \"status\": \"healthy\"}
    {\"ticker\": \"MSFT\", \"status\": \"watch\"}
  ],
  \"pnl_breakdown\": {},
  \"concentration_risks\": [],
  \"crowded_exposures\": [],
  \"manager_actions\": []
}"""

        bad_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": malformed_but_recoverable_json})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {"total_positions": 1, "processed_positions": 1, "timeframe": "1d", "lookback_days": 90},
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [{"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"}],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.return_value = bad_response

            response = self.client.post("/api/account-analysis", json={"workspace_name": "default"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["analysis"]["summary"], "Portfolio check")
        self.assertEqual(payload["analysis"]["portfolio_health_score"], 80.0)

    def test_post_account_analysis_recovers_when_llm_output_has_multiple_json_comma_errors(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 500},
                "positions": [{"ticker": "AAPL", "book_type": "core", "cost_basis": 100, "shares": 1}],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [119.0],
                "High": [123.0],
                "Low": [118.0],
                "Close": [125.0],
                "Volume": [1200],
            }
        )

        malformed_with_multiple_missing_commas = """{
  \"summary\": \"Portfolio check\",
  \"portfolio_health_score\": 80,
  \"holding_health\": [
    {\"ticker\": \"AAPL\", \"status\": \"healthy\"}
    {\"ticker\": \"MSFT\", \"status\": \"watch\"}
  ],
  \"pnl_breakdown\": {}
  \"concentration_risks\": [],
  \"crowded_exposures\": [],
  \"manager_actions\": []
}"""

        bad_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": malformed_with_multiple_missing_commas})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {"total_positions": 1, "processed_positions": 1, "timeframe": "1d", "lookback_days": 90},
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [{"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"}],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.return_value = bad_response

            response = self.client.post("/api/account-analysis", json={"workspace_name": "default"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["analysis"]["summary"], "Portfolio check")
        self.assertEqual(payload["analysis"]["portfolio_health_score"], 80.0)

    def test_post_account_analysis_falls_back_when_llm_output_has_no_json(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 500, "current_drawdown": 6.2},
                "positions": [{"ticker": "AAPL", "book_type": "core", "cost_basis": 100, "shares": 1}],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [119.0],
                "High": [123.0],
                "Low": [118.0],
                "Close": [125.0],
                "Volume": [1200],
            }
        )

        non_json_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": "组合风险偏高，建议降低集中度并控制回撤。"})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {"total_positions": 1, "processed_positions": 1, "timeframe": "1d", "lookback_days": 90},
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [{"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"}],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.return_value = non_json_response

            response = self.client.post("/api/account-analysis", json={"workspace_name": "default"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn("模型输出未返回JSON", payload["analysis"]["summary"])
        self.assertEqual(payload["analysis"]["portfolio_health_score"], 0.0)

    def test_post_account_analysis_repairs_truncated_json_with_second_llm_pass(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 500, "current_drawdown": 6.2},
                "positions": [{"ticker": "AAPL", "book_type": "core", "cost_basis": 100, "shares": 1}],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [119.0],
                "High": [123.0],
                "Low": [118.0],
                "Close": [125.0],
                "Volume": [1200],
            }
        )

        truncated_json_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": '{"summary":"组合偏弱","portfolio_health_score":45,"holding_health":[]'})()},
                    )
                ]
            },
        )()

        repaired_json_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": json.dumps({
                            "summary": "修复后结构化输出",
                            "portfolio_health_score": 45,
                            "holding_health": [],
                            "pnl_breakdown": {},
                            "concentration_risks": [],
                            "crowded_exposures": [],
                            "manager_actions": []
                        }, ensure_ascii=False)})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {"total_positions": 1, "processed_positions": 1, "timeframe": "1d", "lookback_days": 90},
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [{"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"}],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.side_effect = [truncated_json_response, repaired_json_response]

            response = self.client.post("/api/account-analysis", json={"workspace_name": "default"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["analysis"]["summary"], "修复后结构化输出")
        self.assertEqual(payload["analysis"]["portfolio_health_score"], 45.0)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_get_account_analysis_history_returns_latest_entries(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 500},
                "positions": [{"ticker": "AAPL", "book_type": "core", "cost_basis": 100, "shares": 1}],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [119.0],
                "High": [123.0],
                "Low": [118.0],
                "Close": [125.0],
                "Volume": [1200],
            }
        )

        first_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": json.dumps({
                            "summary": "First snapshot",
                            "portfolio_health_score": 70,
                            "holding_health": [],
                            "pnl_breakdown": {},
                            "concentration_risks": [],
                            "crowded_exposures": [],
                            "manager_actions": []
                        })})()},
                    )
                ]
            },
        )()
        second_response = type(
            "FakeResponse",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMessage", (), {"content": json.dumps({
                            "summary": "Second snapshot",
                            "portfolio_health_score": 82,
                            "holding_health": [],
                            "pnl_breakdown": {},
                            "concentration_risks": [],
                            "crowded_exposures": [],
                            "manager_actions": []
                        })})()},
                    )
                ]
            },
        )()

        fake_core_skill_block = {
            "enabled": True,
            "summary": {"total_positions": 1, "processed_positions": 1, "timeframe": "1d", "lookback_days": 90},
            "actions": {"add": [], "trim": [], "hold": []},
            "score_table": [{"ticker": "AAPL", "recommended_action": "观察", "trend_score": 70, "entry_score": 60, "volatility_score": 40, "risk_reward_ratio": "1.5:1", "suggested_position_range": "3%-5%", "decision": "持有", "justification": "结构正常"}],
        }

        with patch.object(web_interface_new.analyzer, "fetch_market_data", return_value=market_df), \
             patch.object(web_interface_new.analyzer.llm_provider, "get_client") as mock_get_client, \
             patch.object(web_interface_new, "_run_core_skill_account_block", return_value=fake_core_skill_block, create=True), \
             patch.object(web_interface_new, "_utc_now_iso", side_effect=[
                 "2026-04-28T10:15:00Z",
                 "2026-04-28T10:15:01Z",
                 "2026-04-28T10:16:00Z",
                 "2026-04-28T10:16:01Z",
             ]):
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.side_effect = [first_response, second_response]

            self.client.post("/api/account-analysis", json={"workspace_name": "default"})
            self.client.post("/api/account-analysis", json={"workspace_name": "default"})

        response = self.client.get("/api/account-analysis/history?workspace_name=default")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual(payload["history"][0]["summary"], "Second snapshot")
        self.assertEqual(payload["history"][1]["summary"], "First snapshot")
        self.assertEqual(payload["history"][0]["workspace_name"], "default")
        self.assertIn("core_skill_block", payload["history"][0])

    def test_get_account_analysis_history_filters_workspace_name(self):
        for workspace_name, summary in (("alpha", "Alpha snapshot"), ("beta", "Beta snapshot")):
            workspace_dir = self.temp_artifact_dir / workspace_name
            json_path = workspace_dir / "2026-04-28T10-10-00Z.json"
            markdown_path = workspace_dir / "2026-04-28T10-10-00Z.md"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "created_at": f"2026-04-28T10:1{0 if workspace_name == 'alpha' else 1}:00Z",
                        "workspace_name": workspace_name,
                        "summary": summary,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            markdown_path.write_text(summary, encoding="utf-8")

        response = self.client.get("/api/account-analysis/history?workspace_name=beta")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["history"]), 1)
        self.assertEqual(payload["history"][0]["workspace_name"], "beta")
        self.assertEqual(payload["history"][0]["summary"], "Beta snapshot")

    def test_get_account_analysis_history_without_workspace_name_aggregates_workspaces(self):
        for workspace_name, created_at in (("alpha", "2026-04-28T10:10:00Z"), ("beta", "2026-04-28T10:11:00Z")):
            workspace_dir = self.temp_artifact_dir / workspace_name
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / f"{created_at.replace(':', '-')}.json").write_text(
                json.dumps(
                    {
                        "created_at": created_at,
                        "workspace_name": workspace_name,
                        "summary": f"{workspace_name} summary",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (workspace_dir / f"{created_at.replace(':', '-')}.md").write_text("summary", encoding="utf-8")

        response = self.client.get("/api/account-analysis/history")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual(payload["history"][0]["workspace_name"], "beta")
        self.assertEqual(payload["history"][1]["workspace_name"], "alpha")


if __name__ == "__main__":
    unittest.main()
