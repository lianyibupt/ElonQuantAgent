import unittest
import sys
import types

langchain_openai = types.ModuleType("langchain_openai")
langchain_openai.ChatOpenAI = lambda *args, **kwargs: types.SimpleNamespace()
sys.modules.setdefault("langchain_openai", langchain_openai)

langchain_core = types.ModuleType("langchain_core")
tools_module = types.ModuleType("langchain_core.tools")
tools_module.tool = lambda func=None, *args, **kwargs: func if func is not None else (lambda f: f)
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.tools", tools_module)

mplfinance = types.ModuleType("mplfinance")
mplfinance.make_addplot = lambda *args, **kwargs: None
mplfinance.make_marketcolors = lambda *args, **kwargs: {}
mplfinance.make_mpf_style = lambda *args, **kwargs: {}
mplfinance.plot = lambda *args, **kwargs: (None, [])
sys.modules.setdefault("mplfinance", mplfinance)
sys.modules.setdefault("talib", types.ModuleType("talib"))
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv)

from core.trading_graph import TradingGraph


class Stage25ActionQueueTests(unittest.TestCase):
    def test_evaluate_portfolio_outputs_action_queue_from_risk_to_stop(self):
        graph = TradingGraph.__new__(TradingGraph)
        state = {
            "asset_symbol": "AAPL",
            "account_state": {
                "nav": 10000,
                "cash": 2000,
                "gross_exposure": 80,
                "current_drawdown": 4,
                "total_risk_to_stop": 700,
                "total_risk_to_stop_pct_nav": 7,
            },
            "positions": [
                {
                    "ticker": "AAPL",
                    "market_value": 3000,
                    "book_type": "core",
                    "factor_tags": ["ai"],
                    "risk_to_stop": 650,
                    "risk_to_stop_pct_nav": 6.5,
                    "stop_price": 108,
                    "latest_price": 120,
                    "planned_action": "hold",
                }
            ],
            "candidates": [{"ticker": "AAPL", "factor_tags": ["ai"]}],
        }
        score = {
            "decision": "买入",
            "recommended_action": "买入",
            "recommended_book": "核心仓",
            "suggested_position_range": "2% - 4%",
        }

        portfolio_directive, dashboard_payload = graph._evaluate_portfolio(state, score)

        self.assertIn("action_queue", portfolio_directive)
        self.assertIn("action_queue", dashboard_payload)
        queue = dashboard_payload["action_queue"]
        self.assertTrue(any(item["action_type"] == "stop_review" for item in queue))
        self.assertTrue(any(item["priority"] == "high" for item in queue))
        self.assertTrue(any(item["ticker"] == "AAPL" for item in queue))
        self.assertIn("total_risk_to_stop_pct_nav", dashboard_payload["account_summary"])


if __name__ == "__main__":
    unittest.main()
