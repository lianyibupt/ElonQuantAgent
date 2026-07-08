import unittest

from core.signal_layer import (
    build_structured_signal_bundle,
    build_rule_based_decision,
)


class SignalLayerTests(unittest.TestCase):
    def test_build_structured_signal_bundle_exposes_machine_readable_features(self):
        kline_data = {
            "Datetime": [f"2026-01-{day:02d}" for day in range(1, 41)],
            "Open": [100 + day for day in range(40)],
            "High": [101 + day for day in range(40)],
            "Low": [99 + day for day in range(40)],
            "Close": [100 + day for day in range(40)],
            "Volume": [1000 + day * 10 for day in range(40)],
        }

        bundle = build_structured_signal_bundle(kline_data)

        self.assertEqual(bundle["data_quality"]["bar_count"], 40)
        self.assertEqual(bundle["price"]["last_close"], 139.0)
        self.assertEqual(bundle["trend"]["direction"], "up")
        self.assertGreater(bundle["trend"]["trend_score"], 70)
        self.assertGreater(bundle["momentum"]["momentum_score"], 60)
        self.assertIn("atr_pct", bundle["volatility"])
        self.assertIn("nearest_support", bundle["levels"])
        self.assertIn("evidence", bundle)
        self.assertIn("contradictions", bundle)

    def test_rule_based_decision_uses_scores_before_llm_explanation(self):
        signal_bundle = {
            "trend": {"trend_score": 82, "direction": "up"},
            "momentum": {"momentum_score": 72},
            "entry": {"entry_score": 68},
            "volatility": {"volatility_score": 64, "atr_pct": 2.0},
            "levels": {"nearest_support": 120.0, "nearest_resistance": 150.0},
            "price": {"last_close": 132.0},
            "evidence": ["trend up", "momentum positive"],
            "contradictions": [],
        }

        decision = build_rule_based_decision(signal_bundle, trading_strategy="high_frequency")

        self.assertEqual(decision["decision"], "买入")
        self.assertEqual(decision["recommended_action"], "买入")
        self.assertEqual(decision["recommended_book"], "战术仓")
        self.assertGreaterEqual(decision["trend_score"], 80)
        self.assertGreaterEqual(decision["entry_score"], 60)
        self.assertEqual(decision["invalidation_price"], "120.00")
        self.assertIn("rule_score", decision)
        self.assertIn("decision_path", decision)


if __name__ == "__main__":
    unittest.main()

class StrategyAwareSignalLayerTests(unittest.TestCase):
    def test_structured_signal_bundle_uses_different_profiles_by_strategy(self):
        kline_data = {
            "Datetime": [f"2026-02-{(day % 28) + 1:02d}" for day in range(80)],
            "Open": [100 + day * 0.3 for day in range(80)],
            "High": [101 + day * 0.3 for day in range(80)],
            "Low": [99 + day * 0.3 for day in range(80)],
            "Close": [100 + day * 0.3 + (3 if day > 70 else 0) for day in range(80)],
            "Volume": [1000 + day * 5 for day in range(80)],
        }

        short_bundle = build_structured_signal_bundle(kline_data, trading_strategy="high_frequency")
        long_bundle = build_structured_signal_bundle(kline_data, trading_strategy="low_frequency")

        self.assertEqual(short_bundle["profile"]["name"], "短期节奏")
        self.assertEqual(long_bundle["profile"]["name"], "长期趋势")
        self.assertLess(short_bundle["profile"]["trend_lookback"], long_bundle["profile"]["trend_lookback"])
        self.assertNotEqual(short_bundle["price"].get("return_profile_pct"), long_bundle["price"].get("return_profile_pct"))

    def test_rule_decision_uses_different_weights_by_strategy(self):
        signal_bundle = {
            "trend": {"trend_score": 78, "direction": "up"},
            "momentum": {"momentum_score": 52},
            "entry": {"entry_score": 46},
            "volatility": {"volatility_score": 74, "atr": 2, "atr_pct": 1.5},
            "levels": {"nearest_support": 120.0, "nearest_resistance": 150.0},
            "price": {"last_close": 132.0},
            "evidence": ["长期趋势向上"],
            "contradictions": [],
        }

        short_decision = build_rule_based_decision(signal_bundle, trading_strategy="high_frequency")
        long_decision = build_rule_based_decision(signal_bundle, trading_strategy="low_frequency")

        self.assertLess(short_decision["rule_score"], long_decision["rule_score"])
        self.assertEqual(short_decision["strategy_profile"], "短期节奏")
        self.assertEqual(long_decision["strategy_profile"], "长期趋势")
