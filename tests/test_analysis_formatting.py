import json
import unittest

from services.analysis_formatting import (
    format_agent_report_for_display,
    format_structured_signal_bundle_for_display,
)


class AnalysisFormattingTests(unittest.TestCase):
    def test_format_agent_report_turns_json_protocol_into_readable_summary(self):
        raw = json.dumps({
            "direction": "up",
            "strength_score": 72,
            "evidence": ["短期均线位于中期均线上方", "MACD柱线为正"],
            "contradictions": ["价格距离支撑较远"],
            "key_levels": {"support": 120, "resistance": 150},
            "invalid_if": "跌破120",
            "summary": "趋势偏多但不宜追高。",
        }, ensure_ascii=False)

        formatted = format_agent_report_for_display(raw, "指标分析")

        self.assertIn("### 指标分析", formatted)
        self.assertIn("- 方向：up", formatted)
        self.assertIn("- 强度评分：72", formatted)
        self.assertIn("短期均线位于中期均线上方", formatted)
        self.assertNotIn('{"direction"', formatted)

    def test_format_structured_signal_bundle_adds_indicator_snapshot(self):
        formatted = format_structured_signal_bundle_for_display({
            "data_quality": {"bar_count": 40, "status": "ok"},
            "price": {"last_close": 139, "return_20_pct": 18.2},
            "trend": {"direction": "up", "trend_score": 86, "ema_8": 135, "ema_21": 128},
            "momentum": {"momentum_score": 75, "rsi_14": 64, "macd_hist": 1.2},
            "entry": {"entry_score": 62, "distance_to_support_pct": 3.4},
            "volatility": {"volatility_score": 70, "atr_pct": 2.1},
            "levels": {"nearest_support": 120, "nearest_resistance": 150},
            "evidence": ["RSI处于可持续区间"],
            "contradictions": [],
        })

        self.assertIn("### 结构化指标快照", formatted)
        self.assertIn("趋势：up / 86", formatted)
        self.assertIn("RSI：64", formatted)
        self.assertIn("支撑 / 阻力：120 / 150", formatted)


if __name__ == "__main__":
    unittest.main()

class AnalysisFormattingIntegrationTests(unittest.TestCase):
    def test_web_extraction_can_display_signal_snapshot_and_formatted_agent_json(self):
        signal_bundle = {
            "data_quality": {"bar_count": 40, "status": "ok"},
            "price": {"last_close": 139, "return_20_pct": 18.2},
            "trend": {"direction": "up", "trend_score": 86, "ema_8": 135, "ema_21": 128},
            "momentum": {"momentum_score": 75, "rsi_14": 64, "macd_hist": 1.2},
            "entry": {"entry_score": 62, "distance_to_support_pct": 3.4, "distance_to_resistance_pct": 7.9},
            "volatility": {"volatility_score": 70, "atr_pct": 2.1},
            "levels": {"nearest_support": 120, "nearest_resistance": 150},
            "evidence": ["RSI处于可持续区间"],
            "contradictions": [],
        }
        raw_indicator = json.dumps({
            "direction": "up",
            "strength_score": 72,
            "evidence": ["MACD柱线为正"],
            "summary": "指标偏多。",
        }, ensure_ascii=False)

        technical_indicators = "\n\n".join([
            format_structured_signal_bundle_for_display(signal_bundle),
            format_agent_report_for_display(raw_indicator, "指标分析"),
        ])

        self.assertIn("结构化指标快照", technical_indicators)
        self.assertIn("RSI：64", technical_indicators)
        self.assertIn("### 指标分析", technical_indicators)
        self.assertIn("指标偏多", technical_indicators)
        self.assertNotIn('{"direction"', technical_indicators)

class SerenityFormattingTests(unittest.TestCase):
    def test_format_serenity_lens_for_display_is_readable(self):
        from services.analysis_formatting import format_serenity_lens_for_display

        formatted = format_serenity_lens_for_display({
            "asset_symbol": "TSLA",
            "market_story": "市场按AI和储能定价。",
            "system_change": "电动车和能源基础设施需求变化。",
            "value_chain_position": "电动车、储能与自动驾驶应用层",
            "scarcity_score": 76,
            "evidence_quality_score": 68,
            "serenity_score": 72,
            "market_misread": "市场可能低估能源业务。",
            "repricing_triggers": ["储能订单提升"],
            "failure_conditions": ["毛利率继续下滑"],
            "research_summary": "研究重点是确认稀缺位置。",
        })

        self.assertIn("Serenity Research Lens", formatted)
        self.assertIn("TSLA", formatted)
        self.assertIn("卡住的环节", formatted)
        self.assertIn("研究优先级分：72", formatted)
        self.assertIn("储能订单提升", formatted)
        self.assertNotIn('{"asset_symbol"', formatted)
