import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


class SimplePrompt:
    def __init__(self, _messages):
        pass

    @classmethod
    def from_messages(cls, messages):
        return cls(messages)

    def __or__(self, llm):
        return llm


langchain_core = types.ModuleType("langchain_core")
prompts = types.ModuleType("langchain_core.prompts")
prompts.ChatPromptTemplate = SimplePrompt
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.prompts", prompts)

module_path = Path(__file__).resolve().parents[1] / "agents" / "decision_agent.py"
spec = importlib.util.spec_from_file_location("decision_agent_under_test", module_path)
decision_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision_agent_module)
create_decision_agent = decision_agent_module.create_decision_agent


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, _messages):
        return FakeResponse(json.dumps({
            "decision": "持有",
            "confidence": "低",
            "risk_reward_ratio": "0:1",
            "forecast_horizon": "2天-1个月",
            "justification": "LLM解释层保守，但规则草案更强。",
            "recommended_book": "观察",
            "recommended_action": "观察",
            "trend_score": 1,
            "entry_score": 1,
            "valuation_stretch_score": 1,
            "catalyst_score": 1,
            "volatility_score": 1,
            "suggested_position_range": "0% - 0%",
            "invalidation_price": "待确认"
        }, ensure_ascii=False))


class DecisionAgentRuleLayerTests(unittest.TestCase):
    def test_decision_agent_keeps_rule_decision_and_adds_llm_explanation(self):
        agent = create_decision_agent(FakeLLM(), [])
        state = {
            "time_frame": "1d",
            "stock_name": "AAPL",
            "trading_strategy": "high_frequency",
            "indicator_report": "{}",
            "pattern_report": "{}",
            "trend_report": "{}",
            "structured_signal_bundle": {
                "trend": {"trend_score": 82, "direction": "up"},
                "momentum": {"momentum_score": 72},
                "entry": {"entry_score": 68},
                "volatility": {"volatility_score": 64, "atr": 2, "atr_pct": 1.5},
                "levels": {"nearest_support": 120.0, "nearest_resistance": 150.0},
                "price": {"last_close": 132.0},
                "evidence": ["trend up"],
                "contradictions": [],
            },
            "messages": [],
        }

        result = agent(state)
        payload = json.loads(result["final_trade_decision"])

        self.assertEqual(payload["decision"], "买入")
        self.assertEqual(payload["recommended_action"], "买入")
        self.assertEqual(payload["recommended_book"], "战术仓")
        self.assertEqual(payload["invalidation_price"], "120.00")
        self.assertIn("llm_explanation", payload)
        self.assertIn("rule_score", payload)


if __name__ == "__main__":
    unittest.main()
