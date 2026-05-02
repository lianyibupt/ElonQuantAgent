import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from services.database import DatabaseManager
import services.database as database_module
import web.web_interface_new as web_interface_new


class Stage2WritebackApiTests(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.test_db_manager = DatabaseManager(self.db_path)

        self.original_web_db_manager = web_interface_new.db_manager
        self.original_database_singleton = database_module._db_manager

        web_interface_new.db_manager = self.test_db_manager
        database_module._db_manager = self.test_db_manager

        web_interface_new.app.config["TESTING"] = True
        self.client = web_interface_new.app.test_client()

    def tearDown(self):
        web_interface_new.db_manager = self.original_web_db_manager
        database_module._db_manager = self.original_database_singleton
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_post_analyze_updates_matching_saved_workspace_holding(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 750, "current_drawdown": 2.5},
                "positions": [
                    {
                        "ticker": "AAPL",
                        "book_type": "core",
                        "cost_basis": 100,
                        "shares": 2,
                        "tracking_status": "active",
                    }
                ],
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

        fake_results = {
            "success": True,
            "asset_name": "Apple Inc.",
            "timeframe": "1day",
            "data_length": 2,
            "final_state": {},
        }
        fake_formatted = {
            "success": True,
            "asset_name": "Apple Inc.",
            "timeframe": "1day",
            "data_length": 2,
            "technical_indicators": "RSI stable",
            "pattern_analysis": "Constructive base",
            "trend_analysis": "Uptrend intact",
            "pattern_chart": "",
            "trend_chart": "",
            "pattern_image_filename": "",
            "trend_image_filename": "",
            "final_decision": {
                "decision": "HOLD",
                "risk_reward_ratio": "2:1",
                "forecast_horizon": "2 weeks",
                "justification": "Trend remains supportive while risk stays contained."
            },
        }

        with patch.object(web_interface_new.db_manager, 'check_existing_analysis', return_value=None), \
             patch.object(web_interface_new.analyzer, 'fetch_market_data', return_value=market_df), \
             patch.object(web_interface_new.analyzer, 'run_analysis', return_value=fake_results), \
             patch.object(web_interface_new.analyzer, 'extract_analysis_results', side_effect=lambda results, workspace_writeback=None: {**fake_formatted, **({"workspace_writeback": workspace_writeback} if workspace_writeback is not None else {})}):
            response = self.client.post(
                '/api/analyze',
                json={
                    'asset': 'AAPL',
                    'timeframe': '1d',
                    'start_date': '2026-04-01',
                    'end_date': '2026-04-28',
                    'trading_strategy': 'high_frequency',
                }
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertTrue(payload['workspace_writeback']['updated'])
        self.assertEqual(payload['workspace_writeback']['updated_ticker'], 'AAPL')
        self.assertEqual(payload['workspace_writeback']['analysis_summary'], 'Trend remains supportive while risk stays contained.')

        saved_workspace = self.test_db_manager.get_stage2_workspace('default')
        position = saved_workspace['positions'][0]
        account_state = saved_workspace['account_state']

        self.assertEqual(position['latest_price'], 125.0)
        self.assertEqual(position['unrealized_pnl'], 50.0)
        self.assertEqual(position['market_value'], 250.0)
        self.assertEqual(position['last_analysis_summary'], 'Trend remains supportive while risk stays contained.')
        self.assertEqual(account_state['total_market_value'], 250.0)
        self.assertEqual(account_state['gross_exposure'], 25.0)
        self.assertEqual(account_state['largest_position'], 'AAPL')

    def test_post_analyze_returns_not_updated_when_workspace_has_no_matching_holding(self):
        self.test_db_manager.save_stage2_workspace(
            {
                "account_state": {"nav": 1000, "cash": 800, "current_drawdown": 1.0},
                "positions": [
                    {
                        "ticker": "MSFT",
                        "book_type": "core",
                        "cost_basis": 100,
                        "shares": 1,
                    }
                ],
            },
            workspace_name="default",
        )

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28"]),
                "Open": [199.0],
                "High": [202.0],
                "Low": [198.0],
                "Close": [200.0],
                "Volume": [900],
            }
        )

        fake_results = {
            "success": True,
            "asset_name": "Apple Inc.",
            "timeframe": "1day",
            "data_length": 1,
            "final_state": {},
        }
        fake_formatted = {
            "success": True,
            "asset_name": "Apple Inc.",
            "timeframe": "1day",
            "data_length": 1,
            "technical_indicators": "RSI stable",
            "pattern_analysis": "",
            "trend_analysis": "",
            "pattern_chart": "",
            "trend_chart": "",
            "pattern_image_filename": "",
            "trend_image_filename": "",
            "final_decision": {"justification": "No matching position should be updated."},
        }

        with patch.object(web_interface_new.db_manager, 'check_existing_analysis', return_value=None), \
             patch.object(web_interface_new.analyzer, 'fetch_market_data', return_value=market_df), \
             patch.object(web_interface_new.analyzer, 'run_analysis', return_value=fake_results), \
             patch.object(web_interface_new.analyzer, 'extract_analysis_results', side_effect=lambda results, workspace_writeback=None: {**fake_formatted, **({"workspace_writeback": workspace_writeback} if workspace_writeback is not None else {})}):
            response = self.client.post(
                '/api/analyze',
                json={
                    'asset': 'AAPL',
                    'timeframe': '1d',
                    'start_date': '2026-04-01',
                    'end_date': '2026-04-28',
                    'trading_strategy': 'high_frequency',
                }
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertFalse(payload['workspace_writeback']['updated'])
        self.assertIn('No saved workspace holding matched AAPL', payload['workspace_writeback']['message'])

        saved_workspace = self.test_db_manager.get_stage2_workspace('default')
        position = saved_workspace['positions'][0]
        self.assertEqual(position['ticker'], 'MSFT')
        self.assertEqual(position['latest_price'], 0.0)


    def test_save_analysis_history_persists_record_for_output_lookup(self):
        history_id = self.test_db_manager.save_analysis_history(
            asset='AAPL',
            timeframe='1d',
            start_date='2026-04-01',
            end_date='2026-04-28',
            trading_strategy='high_frequency',
            status='completed',
            result_summary='AAPL 1d 分析结果',
            result_details={'success': True, 'asset_name': 'Apple Inc.', 'timeframe': '1day'}
        )

        loaded = self.test_db_manager.get_analysis_history_by_id(history_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['asset'], 'AAPL')
        self.assertEqual(loaded['timeframe'], '1d')

    def test_output_route_returns_error_state_when_result_id_missing(self):
        response = self.client.get('/output?id=999999')

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Analysis record not found for id 999999', html)
        self.assertNotIn('No analysis data available', html)
        self.assertNotIn('>1h<', html)


if __name__ == '__main__':
    unittest.main()
