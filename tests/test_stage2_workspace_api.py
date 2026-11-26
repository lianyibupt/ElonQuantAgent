import os
import tempfile
import unittest

from services.database import DatabaseManager
import services.database as database_module
import web.web_interface_new as web_interface_new


class Stage2WorkspaceApiTests(unittest.TestCase):
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

    def test_get_stage2_workspace_returns_default_workspace_when_nothing_stored(self):
        response = self.client.get('/api/stage2-workspace')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        workspace = payload['workspace']

        self.assertTrue(payload['success'])
        self.assertEqual(workspace['workspace_name'], 'default')
        self.assertEqual(workspace['positions'], [])
        self.assertEqual(workspace['candidates'], [])
        self.assertEqual(workspace['notes'], '')
        self.assertEqual(workspace['account_state']['nav'], 0.0)
        self.assertEqual(workspace['account_state']['total_market_value'], 0.0)
        self.assertEqual(workspace['account_state']['position_count'], 0)
        self.assertIsNone(workspace['account_state']['last_aggregated_at'])
        self.assertIsNone(workspace['created_at'])
        self.assertIsNone(workspace['updated_at'])

    def test_portfolio_journal_links_trade_to_immutable_holding_snapshot(self):
        position = {
            "ticker": "00700.HK", "account": "Example", "currency": "HKD",
            "shares": 2.5, "cost_basis": 80, "latest_price": 100, "fx_to_usd": 0.125,
        }
        response = self.client.post("/api/stage2-workspace", json={
            "positions": [position], "account_state": {"nav": 1000}
        })
        saved = response.get_json()["workspace"]["positions"][0]
        self.assertEqual(saved["market_value"], 31.25)
        self.assertEqual(saved["unrealized_pnl"], 6.25)
        self.assertEqual(saved["position_weight"], 3.125)

        advice = self.client.post("/api/portfolio-journal/advice", json={
            "date": "2026-11-26", "title": "组合风险", "content": "核对估值假设。"
        })
        self.assertEqual(advice.status_code, 201)
        advice_id = advice.get_json()["id"]
        trade = {
            "date": "2026-11-27", "account": "Example", "ticker": "00700.HK",
            "side": "sell", "quantity": "0.5", "price": "100", "fee": "1.25", "currency": "HKD",
        }
        self.assertEqual(self.client.post(
            "/api/portfolio-journal/trade", json={**trade, "advice_id": advice_id + 1}
        ).status_code, 400)
        self.assertEqual(self.client.post(
            "/api/portfolio-journal/trade", json={**trade, "advice_id": advice_id}
        ).status_code, 201)
        self.assertEqual(self.client.get("/api/stage2-workspace").get_json()["workspace"]["positions"][0]["shares"], 2.5)

        self.client.post("/api/stage2-workspace", json={"positions": [{**position, "shares": 2}]})
        trade_entry, advice_entry = self.client.get("/api/portfolio-journal").get_json()["entries"]
        self.assertEqual(trade_entry["advice_id"], advice_entry["id"])
        self.assertEqual(trade_entry["cash_flow"], "48.75")
        self.assertEqual(advice_entry["snapshot"]["positions"][0]["shares"], 2.5)

    def test_post_stage2_workspace_normalizes_and_round_trips_workspace(self):
        response = self.client.post(
            '/api/stage2-workspace',
            json={
                'workspace_name': 'swing-book',
                'account_state': {
                    'nav': 10000,
                    'cash': 2500,
                    'current_drawdown': '4.5',
                },
                'positions': [
                    {
                        'ticker': ' msft ',
                        'book_type': 'core',
                        'cost_basis': '410.25',
                        'shares': '3',
                        'tracking_status': 'active',
                        'factor_tags': ['ai', 'quality'],
                        'notes': 'buy on dips',
                    }
                ],
                'candidates': [{'ticker': ' nvda ', 'factor_tags': 'ai, momentum', 'notes': 'watch'}],
                'notes': 'monitor earnings calendar',
            }
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        workspace = payload['workspace']

        self.assertTrue(payload['success'])
        self.assertEqual(workspace['workspace_name'], 'swing-book')
        self.assertEqual(workspace['notes'], 'monitor earnings calendar')
        self.assertEqual(workspace['candidates'], [{'ticker': 'NVDA', 'factor_tags': ['ai', 'momentum'], 'notes': 'watch', 'last_screened_at': None}])
        self.assertIsNotNone(workspace['created_at'])
        self.assertIsNotNone(workspace['updated_at'])

        position = workspace['positions'][0]
        self.assertEqual(position['ticker'], 'MSFT')
        self.assertEqual(position['cost_basis'], 410.25)
        self.assertEqual(position['shares'], 3.0)

        get_response = self.client.get('/api/stage2-workspace?workspace_name=swing-book')
        self.assertEqual(get_response.status_code, 200)
        get_payload = get_response.get_json()
        stored_workspace = get_payload['workspace']

        self.assertTrue(get_payload['success'])
        self.assertEqual(stored_workspace['workspace_name'], 'swing-book')
        self.assertEqual(stored_workspace['notes'], 'monitor earnings calendar')
        self.assertEqual(stored_workspace['candidates'], [{'ticker': 'NVDA', 'factor_tags': ['ai', 'momentum'], 'notes': 'watch', 'last_screened_at': None}])
        self.assertEqual(stored_workspace['positions'][0]['ticker'], 'MSFT')
        self.assertEqual(stored_workspace['positions'][0]['cost_basis'], 410.25)
        self.assertEqual(stored_workspace['positions'][0]['shares'], 3.0)

    def test_get_stage2_workspace_normalizes_partial_stored_workspace_shape(self):
        self.test_db_manager.save_stage2_workspace(
            {
                'account_state': {'nav': 5000},
                'positions': [],
                'candidates': 'bad payload',
                'notes': None,
            },
            workspace_name='legacy-book',
        )

        response = self.client.get('/api/stage2-workspace?workspace_name=legacy-book')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        workspace = payload['workspace']

        self.assertTrue(payload['success'])
        self.assertEqual(workspace['workspace_name'], 'legacy-book')
        self.assertEqual(workspace['account_state']['nav'], 5000.0)
        self.assertEqual(workspace['account_state']['total_market_value'], 0.0)
        self.assertEqual(workspace['account_state']['position_count'], 0)
        self.assertEqual(workspace['candidates'], [])
        self.assertEqual(workspace['notes'], '')


if __name__ == '__main__':
    unittest.main()
