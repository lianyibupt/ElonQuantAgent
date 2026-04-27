"""TradingGraph: simplified orchestrator for the multi-agent trading system."""

import json
import os
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from config.default_config import DEFAULT_CONFIG
from utils.graph_util import TechnicalTools


def safe_str(obj):
    """Safely convert object to string, handling encoding issues."""
    try:
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, str):
            return obj.encode("utf-8", errors="replace").decode("utf-8")
        return str(obj).encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "Error converting to string"


class TradingGraph:
    """Simplified orchestrator for the multi-agent trading system."""

    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.agent_llm = self._create_llm(
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )
        self.toolkit = TechnicalTools()

    def _create_llm(self, model="gpt-4o-mini", temperature=0.1):
        """Create LLM instance with proper provider support."""
        try:
            llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")

            if llm_provider == "deepseek":
                deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
                if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
                    deepseek_model = "deepseek-v4-flash" if "gpt" in model.lower() else model
                    return ChatOpenAI(
                        model=deepseek_model,
                        temperature=temperature,
                        api_key=deepseek_key,
                        base_url="https://api.deepseek.com/v1",
                    )

            if llm_provider == "volcengine":
                volcengine_key = os.environ.get("VOLCENGINE_API_KEY")
                if volcengine_key and volcengine_key != "your-volcengine-api-key-here":
                    volcengine_model = "ep-20250519162223-96wj4" if "gpt" in model.lower() else model
                    return ChatOpenAI(
                        model=volcengine_model,
                        temperature=temperature,
                        api_key=volcengine_key,
                        base_url="https://ark-cn-beijing.bytedance.net/api/v3",
                    )

            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key-here":
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=openai_key,
                )

            print(f"Warning: No valid API key found for LLM provider: {llm_provider}")
            return ChatOpenAI(model=model, temperature=temperature)
        except Exception as e:
            error_msg = safe_str(e)
            print(f"Error creating LLM: {error_msg}")
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    def refresh_llms(self):
        """Refresh the LLM objects with the current API key from environment."""
        try:
            self.agent_llm = self._create_llm(
                model=self.config.get("agent_llm_model", "gpt-4o-mini"),
                temperature=self.config.get("agent_llm_temperature", 0.1),
            )
            self.graph_llm = self._create_llm(
                model=self.config.get("graph_llm_model", "gpt-4o"),
                temperature=self.config.get("graph_llm_temperature", 0.1),
            )
        except Exception as e:
            error_msg = safe_str(e)
            print(f"Error refreshing LLMs: {error_msg}")

    def _build_initial_state(
        self,
        data,
        asset_symbol: str,
        time_frame: str,
        trading_strategy: str,
        account_state: Optional[Dict[str, Any]] = None,
        positions: Optional[list] = None,
        candidates: Optional[list] = None,
    ) -> Dict[str, Any]:
        return {
            "kline_data": data,
            "data": data,
            "asset_symbol": asset_symbol,
            "time_frame": time_frame,
            "stock_name": asset_symbol,
            "messages": [],
            "indicator_report": "",
            "pattern_report": "",
            "trend_report": "",
            "decision_report": "",
            "trading_strategy": trading_strategy,
            "account_state": account_state or {},
            "positions": positions or [],
            "candidates": candidates or [],
            "decision_payload": {},
            "single_name_score": {},
            "portfolio_directive": {},
            "dashboard_payload": {},
        }

    def _create_agents(self, text_only: bool = False):
        from agents.decision_agent import create_decision_agent
        from agents.indicator_agent import create_indicator_agent
        from agents.pattern_agent import create_pattern_agent, create_pattern_agent_text_only
        from agents.trend_agent import create_trend_agent, create_trend_agent_text_only

        indicator_agent = create_indicator_agent(self.agent_llm, self.toolkit)
        pattern_agent = (
            create_pattern_agent_text_only(self.agent_llm, self.toolkit.get_pattern_tools())
            if text_only
            else create_pattern_agent(self.agent_llm, self.toolkit.get_pattern_tools())
        )
        trend_agent = (
            create_trend_agent_text_only(self.agent_llm, self.toolkit.get_trend_tools())
            if text_only
            else create_trend_agent(self.agent_llm, self.toolkit.get_trend_tools())
        )
        decision_agent = create_decision_agent(self.graph_llm, self.toolkit.get_decision_tools())
        return indicator_agent, pattern_agent, trend_agent, decision_agent

    def _parse_decision_payload(self, raw_decision: Any) -> Dict[str, Any]:
        decision_text = safe_str(raw_decision)
        if not decision_text:
            return {}

        try:
            return json.loads(decision_text)
        except json.JSONDecodeError:
            start = decision_text.find("{")
            end = decision_text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(decision_text[start:end])
                except json.JSONDecodeError:
                    return {}
        return {}

    def _normalize_scorecard(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def clamp_score(value: Any) -> int:
            try:
                numeric = int(float(value))
            except (TypeError, ValueError):
                numeric = 0
            return max(0, min(100, numeric))

        return {
            "decision": safe_str(payload.get("decision", "持有")),
            "confidence": safe_str(payload.get("confidence", "低")),
            "risk_reward_ratio": safe_str(payload.get("risk_reward_ratio", "N/A")),
            "forecast_horizon": safe_str(payload.get("forecast_horizon", "未知")),
            "justification": safe_str(payload.get("justification", "")),
            "recommended_book": safe_str(payload.get("recommended_book", "观察")),
            "recommended_action": safe_str(payload.get("recommended_action", "观察")),
            "trend_score": clamp_score(payload.get("trend_score", 0)),
            "entry_score": clamp_score(payload.get("entry_score", 0)),
            "valuation_stretch_score": clamp_score(payload.get("valuation_stretch_score", 0)),
            "catalyst_score": clamp_score(payload.get("catalyst_score", 0)),
            "volatility_score": clamp_score(payload.get("volatility_score", 0)),
            "suggested_position_range": safe_str(payload.get("suggested_position_range", "0% - 0%")),
            "invalidation_price": safe_str(payload.get("invalidation_price", "待确认")),
        }

    def _build_final_state(self, state: Dict[str, Any], text_only: bool = False) -> Dict[str, Any]:
        raw_decision = state.get("final_trade_decision", "")
        decision_payload = self._parse_decision_payload(raw_decision)
        single_name_score = self._normalize_scorecard(decision_payload)
        state["decision_payload"] = decision_payload
        state["single_name_score"] = single_name_score

        return {
            "indicator_report": state.get("indicator_report", ""),
            "pattern_report": state.get("pattern_report", ""),
            "trend_report": state.get("trend_report", ""),
            "final_trade_decision": raw_decision,
            "decision_payload": decision_payload,
            "single_name_score": single_name_score,
            "account_state": state.get("account_state", {}),
            "positions": state.get("positions", []),
            "candidates": state.get("candidates", []),
            "portfolio_directive": state.get("portfolio_directive", {}),
            "dashboard_payload": state.get("dashboard_payload", {}),
            "pattern_image": "" if text_only else state.get("pattern_image", ""),
            "trend_image": "" if text_only else state.get("trend_image", ""),
            "pattern_image_filename": "" if text_only else state.get("pattern_image_filename", ""),
            "trend_image_filename": "" if text_only else state.get("trend_image_filename", ""),
        }

    def _run_pipeline(
        self,
        data,
        asset_symbol="BTC",
        time_frame="1d",
        trading_strategy="high_frequency",
        generate_charts: bool = True,
        account_state: Optional[Dict[str, Any]] = None,
        positions: Optional[list] = None,
        candidates: Optional[list] = None,
    ):
        try:
            indicator_agent, pattern_agent, trend_agent, decision_agent = self._create_agents(
                text_only=not generate_charts
            )
            state = self._build_initial_state(
                data=data,
                asset_symbol=asset_symbol,
                time_frame=time_frame,
                trading_strategy=trading_strategy,
                account_state=account_state,
                positions=positions,
                candidates=candidates,
            )

            mode_label = "分析" if generate_charts else "文本分析"
            print(f"🔍 [TradingGraph] 开始{mode_label} {asset_symbol}，时间框架: {time_frame}")
            if isinstance(data, dict):
                print(f"  数据格式: 字典，包含键: {list(data.keys())}")
                if "Datetime" in data:
                    print(f"  数据长度: {len(data['Datetime']) if hasattr(data['Datetime'], '__len__') else 'N/A'}")
            else:
                print(f"  数据格式: {type(data)}")
                print(f"  数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")

            print("📊 [TradingGraph] 运行指标分析...")
            state = indicator_agent(state)
            print(f"  指标分析结果长度: {len(state.get('indicator_report', ''))}")

            pattern_label = "运行形态分析..." if generate_charts else "运行形态分析(文本模式)..."
            print(f"📊 [TradingGraph] {pattern_label}")
            state = pattern_agent(state)
            print(f"  形态分析结果长度: {len(state.get('pattern_report', ''))}")
            if generate_charts:
                print(f"  形态图像: {'有' if state.get('pattern_image') else '无'}")

            trend_label = "运行趋势分析..." if generate_charts else "运行趋势分析(文本模式)..."
            print(f"📊 [TradingGraph] {trend_label}")
            state = trend_agent(state)
            print(f"  趋势分析结果长度: {len(state.get('trend_report', ''))}")
            if generate_charts:
                print(f"  趋势图像: {'有' if state.get('trend_image') else '无'}")

            print("📊 [TradingGraph] 运行决策分析...")
            state = decision_agent(state)
            print(f"  最终决策: {safe_str(state.get('final_trade_decision', '无'))[:100]}...")

            final_state = self._build_final_state(state, text_only=not generate_charts)
            final_result = {"success": True, "final_state": final_state}

            print(f"✅ [TradingGraph] {mode_label}完成!")
            print(f"  指标报告长度: {len(final_state['indicator_report'])}")
            print(f"  形态报告长度: {len(final_state['pattern_report'])}")
            print(f"  趋势报告长度: {len(final_state['trend_report'])}")
            print(f"  决策长度: {len(final_state['final_trade_decision'])}")
            return final_result
        except Exception as e:
            error_msg = f"Analysis failed: {safe_str(e)}"
            print(f"TradingGraph analysis error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "final_state": {
                    "indicator_report": f"指标分析失败: {error_msg}",
                    "pattern_report": f"形态分析失败: {error_msg}",
                    "trend_report": f"趋势分析失败: {error_msg}",
                    "final_trade_decision": f"决策分析失败: {error_msg}",
                    "decision_payload": {},
                    "single_name_score": {},
                    "account_state": account_state or {},
                    "positions": positions or [],
                    "candidates": candidates or [],
                    "portfolio_directive": {},
                    "dashboard_payload": {},
                    "pattern_image": "",
                    "trend_image": "",
                    "pattern_image_filename": "",
                    "trend_image_filename": "",
                },
            }

    def analyze(
        self,
        data,
        asset_symbol="BTC",
        time_frame="1d",
        trading_strategy="high_frequency",
        account_state: Optional[Dict[str, Any]] = None,
        positions: Optional[list] = None,
        candidates: Optional[list] = None,
    ):
        return self._run_pipeline(
            data=data,
            asset_symbol=asset_symbol,
            time_frame=time_frame,
            trading_strategy=trading_strategy,
            generate_charts=True,
            account_state=account_state,
            positions=positions,
            candidates=candidates,
        )

    def analyze_text_only(
        self,
        data,
        asset_symbol="BTC",
        time_frame="1d",
        trading_strategy="high_frequency",
        account_state: Optional[Dict[str, Any]] = None,
        positions: Optional[list] = None,
        candidates: Optional[list] = None,
    ):
        return self._run_pipeline(
            data=data,
            asset_symbol=asset_symbol,
            time_frame=time_frame,
            trading_strategy=trading_strategy,
            generate_charts=False,
            account_state=account_state,
            positions=positions,
            candidates=candidates,
        )
