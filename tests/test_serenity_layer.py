import unittest

from core.serenity_layer import build_serenity_research_lens, blend_serenity_into_decision


class SerenityLayerTests(unittest.TestCase):
    def test_build_serenity_research_lens_returns_us_equity_research_schema(self):
        lens = build_serenity_research_lens(
            asset_symbol="TSLA",
            structured_signal_bundle={
                "trend": {"trend_score": 72, "direction": "up"},
                "entry": {"entry_score": 58},
                "volatility": {"volatility_score": 62},
                "evidence": ["趋势向上"],
                "contradictions": [],
            },
            candidate_context={"factor_tags": ["EV", "AI", "Energy Storage"]},
        )

        self.assertEqual(lens["scope"], "us_equity")
        self.assertEqual(lens["asset_symbol"], "TSLA")
        self.assertIn("market_story", lens)
        self.assertIn("system_change", lens)
        self.assertIn("value_chain_position", lens)
        self.assertIn("scarcity_score", lens)
        self.assertIn("evidence_quality_score", lens)
        self.assertIn("market_misread", lens)
        self.assertIn("repricing_triggers", lens)
        self.assertIn("failure_conditions", lens)
        self.assertIn("serenity_score", lens)
        self.assertIn("research_summary", lens)
        self.assertGreaterEqual(lens["serenity_score"], 0)
        self.assertLessEqual(lens["serenity_score"], 100)

    def test_blend_serenity_into_decision_changes_long_term_more_than_short_term(self):
        base_decision = {
            "decision": "持有",
            "recommended_action": "观察",
            "recommended_book": "观察",
            "rule_score": 64,
            "trend_score": 68,
            "entry_score": 52,
            "catalyst_score": 50,
            "volatility_score": 60,
            "justification": "技术面中性偏强。",
        }
        serenity_lens = {
            "serenity_score": 88,
            "scarcity_score": 90,
            "evidence_quality_score": 82,
            "research_summary": "公司处在难以快速复制的供应链位置。",
            "failure_conditions": ["客户需求验证失败"],
        }

        short_result = blend_serenity_into_decision(base_decision, serenity_lens, "high_frequency")
        long_result = blend_serenity_into_decision(base_decision, serenity_lens, "low_frequency")

        self.assertEqual(short_result["serenity_weight"], 0.25)
        self.assertEqual(long_result["serenity_weight"], 0.55)
        self.assertGreater(long_result["rule_score"], short_result["rule_score"])
        self.assertIn("serenity_score", long_result)
        self.assertIn("Serenity", long_result["justification"])


if __name__ == "__main__":
    unittest.main()
