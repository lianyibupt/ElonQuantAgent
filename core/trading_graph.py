"""TradingGraph: simplified orchestrator for the multi-agent trading system."""

import json
import os
from collections import Counter
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from config.default_config import DEFAULT_CONFIG
from core.serenity_layer import build_serenity_research_lens
from core.signal_layer import build_structured_signal_bundle
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
        structured_signal_bundle = build_structured_signal_bundle(
            data,
            trading_strategy=trading_strategy,
        )
        serenity_lens = self._build_serenity_lens(asset_symbol, structured_signal_bundle, candidates or [])
        return {
            "kline_data": data,
            "data": data,
            "structured_signal_bundle": structured_signal_bundle,
            "serenity_lens": serenity_lens,
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
            "rule_score": clamp_score(payload.get("rule_score", 0)),
            "strategy_profile": safe_str(payload.get("strategy_profile", "")),
            "strategy_weights": payload.get("strategy_weights", {}) if isinstance(payload.get("strategy_weights", {}), dict) else {},
            "serenity_score": clamp_score(payload.get("serenity_score", 0)),
            "serenity_weight": payload.get("serenity_weight", 0),
            "scarcity_score": clamp_score(payload.get("scarcity_score", 0)),
            "evidence_quality_score": clamp_score(payload.get("evidence_quality_score", 0)),
            "serenity_summary": safe_str(payload.get("serenity_summary", "")),
        }

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, "", "N/A"):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _format_pct(self, value: float) -> str:
        return f"{value:.1f}%"

    def _format_amount(self, value: float) -> str:
        return f"{value:,.2f}"

    def _normalize_ticker(self, value: Any) -> str:
        return safe_str(value or "").strip().upper()

    def _normalize_tags(self, item: Dict[str, Any]) -> list:
        if not isinstance(item, dict):
            return []
        tags = item.get("factor_tags") or item.get("tags") or []
        if isinstance(tags, str):
            return [safe_str(tags).strip()] if safe_str(tags).strip() else []
        if isinstance(tags, list):
            normalized = []
            for tag in tags:
                tag_text = safe_str(tag).strip()
                if tag_text:
                    normalized.append(tag_text)
            return normalized
        return []

    def _find_candidate_context(self, asset_symbol: str, candidates: list) -> Dict[str, Any]:
        normalized_symbol = self._normalize_ticker(asset_symbol)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_symbol = self._normalize_ticker(
                candidate.get("ticker") or candidate.get("symbol") or candidate.get("asset_symbol")
            )
            if candidate_symbol == normalized_symbol:
                return candidate
        return {}

    def _build_serenity_lens(
        self,
        asset_symbol: str,
        structured_signal_bundle: Dict[str, Any],
        candidates: list,
    ) -> Dict[str, Any]:
        candidate_context = self._find_candidate_context(asset_symbol, candidates or [])
        return build_serenity_research_lens(
            asset_symbol=asset_symbol,
            structured_signal_bundle=structured_signal_bundle,
            candidate_context=candidate_context,
        )

    def _portfolio_targets(self, current_drawdown: float) -> Dict[str, float]:
        if current_drawdown >= 15:
            return {
                "market_regime": "防守",
                "gross_min": 20.0,
                "gross_max": 40.0,
                "core_min": 15.0,
                "core_max": 30.0,
                "tactical_min": 0.0,
                "tactical_max": 10.0,
            }
        if current_drawdown >= 10:
            return {
                "market_regime": "收缩",
                "gross_min": 40.0,
                "gross_max": 60.0,
                "core_min": 25.0,
                "core_max": 40.0,
                "tactical_min": 5.0,
                "tactical_max": 15.0,
            }
        return {
            "market_regime": "进攻",
            "gross_min": 60.0,
            "gross_max": 85.0,
            "core_min": 35.0,
            "core_max": 60.0,
            "tactical_min": 10.0,
            "tactical_max": 25.0,
        }

    def _evaluate_portfolio(self, state: Dict[str, Any], single_name_score: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        account_state = state.get("account_state", {}) or {}
        positions = state.get("positions", []) or account_state.get("positions", []) or []
        candidates = state.get("candidates", []) or []
        asset_symbol = safe_str(state.get("asset_symbol", "")).strip()

        if not account_state and not positions and not candidates:
            return {}, {}

        normalized_positions = [position for position in positions if isinstance(position, dict)]
        candidate_context = self._find_candidate_context(asset_symbol, candidates)
        candidate_tags = self._normalize_tags(candidate_context)
        position_values = [self._coerce_float(position.get("market_value")) for position in normalized_positions]
        total_market_value = sum(position_values)
        nav = self._coerce_float(account_state.get("nav"))
        cash = self._coerce_float(account_state.get("cash"))
        current_drawdown = self._coerce_float(account_state.get("current_drawdown"))

        if nav <= 0 and (cash > 0 or total_market_value > 0):
            nav = cash + total_market_value

        gross_exposure = self._coerce_float(account_state.get("gross_exposure"))
        if gross_exposure <= 0 and nav > 0 and total_market_value > 0:
            gross_exposure = (total_market_value / nav) * 100

        core_value = sum(
            self._coerce_float(position.get("market_value"))
            for position in normalized_positions
            if safe_str(position.get("book_type", "")).strip() == "核心仓"
        )
        tactical_value = sum(
            self._coerce_float(position.get("market_value"))
            for position in normalized_positions
            if safe_str(position.get("book_type", "")).strip() == "战术仓"
        )

        core_exposure = self._coerce_float(account_state.get("core_exposure"))
        tactical_exposure = self._coerce_float(account_state.get("tactical_exposure"))
        if core_exposure <= 0 and nav > 0 and core_value > 0:
            core_exposure = (core_value / nav) * 100
        if tactical_exposure <= 0 and nav > 0 and tactical_value > 0:
            tactical_exposure = (tactical_value / nav) * 100

        cash_pct = (cash / nav) * 100 if nav > 0 else 0.0
        targets = self._portfolio_targets(current_drawdown)
        remaining_risk_budget = max(targets["gross_max"] - gross_exposure, 0.0)

        position_tickers = [
            self._normalize_ticker(position.get("ticker") or position.get("symbol"))
            for position in normalized_positions
        ]
        existing_position = self._normalize_ticker(asset_symbol) in position_tickers

        factor_counter = Counter()
        for position in normalized_positions:
            factor_counter.update(self._normalize_tags(position))
        crowded_exposures = [tag for tag, count in factor_counter.items() if count >= 2]
        crowded_candidate_tags = [tag for tag in candidate_tags if tag in crowded_exposures]

        recommended_action = safe_str(single_name_score.get("recommended_action", "观察"))
        recommended_book = safe_str(single_name_score.get("recommended_book", "观察"))
        decision = safe_str(single_name_score.get("decision", "持有"))
        is_add_risk = recommended_action in {"买入", "加仓"} or decision == "买入"
        is_trim_risk = recommended_action in {"减仓", "退出"} or decision == "卖出"

        portfolio_checks = []
        blocked_reasons = []

        def record_check(name: str, passed: bool, reason: str):
            portfolio_checks.append({
                "name": name,
                "status": "pass" if passed else "block",
                "reason": reason,
            })
            if not passed:
                blocked_reasons.append(reason)

        if is_add_risk:
            record_check("cash_buffer", cash > 0, "现金不足，无法继续新增风险。")
            record_check(
                "gross_exposure",
                gross_exposure < targets["gross_max"],
                f"总仓位 {self._format_pct(gross_exposure)} 已达到或超过目标上限 {self._format_pct(targets['gross_max'])}。",
            )
            if recommended_book == "核心仓":
                record_check(
                    "core_book_limit",
                    core_exposure < targets["core_max"],
                    f"核心仓暴露 {self._format_pct(core_exposure)} 已达到或超过目标上限 {self._format_pct(targets['core_max'])}。",
                )
            if recommended_book == "战术仓":
                record_check(
                    "tactical_book_limit",
                    tactical_exposure < targets["tactical_max"],
                    f"战术仓暴露 {self._format_pct(tactical_exposure)} 已达到或超过目标上限 {self._format_pct(targets['tactical_max'])}。",
                )
            record_check(
                "drawdown_limit",
                current_drawdown < 15.0,
                f"当前回撤 {self._format_pct(current_drawdown)} 已进入防守阈值，不宜继续加风险。",
            )
            record_check(
                "duplicate_ticker",
                not existing_position,
                f"{asset_symbol} 已存在持仓，新增前需要先确认是否属于加仓而非重复建仓。",
            )
            record_check(
                "crowded_theme",
                len(crowded_candidate_tags) == 0,
                f"候选标的与已拥挤暴露重合: {', '.join(crowded_candidate_tags)}。",
            )

        manager_actions = []
        if blocked_reasons:
            manager_actions.append(f"暂停新增 {asset_symbol} 风险：{'；'.join(blocked_reasons)}")
        elif is_add_risk:
            manager_actions.append(
                f"可按 {single_name_score.get('suggested_position_range', '0% - 0%')} 评估 {asset_symbol} 的新增仓位。"
            )
        if is_trim_risk:
            manager_actions.append(f"{asset_symbol} 进入减仓/退出观察列表，优先核对失效价与仓位来源。")
        if current_drawdown >= 15:
            manager_actions.append("账户处于防守模式，优先降波动与保留现金。")
        elif current_drawdown >= 10:
            manager_actions.append("账户处于收缩模式，新仓只保留高置信度机会。")
        if crowded_exposures:
            manager_actions.append(f"当前拥挤主题: {', '.join(crowded_exposures)}，避免进一步集中。")
        if not manager_actions:
            manager_actions.append("当前组合约束中性，可继续观察信号演进。")

        largest_position = ""
        largest_position_value = 0.0
        for position in normalized_positions:
            market_value = self._coerce_float(position.get("market_value"))
            if market_value > largest_position_value:
                largest_position_value = market_value
                largest_position = self._normalize_ticker(position.get("ticker") or position.get("symbol"))

        portfolio_directive = {
            "market_regime": targets["market_regime"],
            "target_gross_exposure": f"{self._format_pct(targets['gross_min'])} - {self._format_pct(targets['gross_max'])}",
            "target_core_exposure": f"{self._format_pct(targets['core_min'])} - {self._format_pct(targets['core_max'])}",
            "target_tactical_exposure": f"{self._format_pct(targets['tactical_min'])} - {self._format_pct(targets['tactical_max'])}",
            "remaining_risk_budget": self._format_pct(remaining_risk_budget),
            "crowded_exposures": crowded_exposures,
            "add_candidates": [asset_symbol] if is_add_risk and not blocked_reasons else [],
            "trim_candidates": [asset_symbol] if is_trim_risk else [],
            "blocked_candidates": [f"{asset_symbol}: {'；'.join(blocked_reasons)}"] if blocked_reasons else [],
            "manager_actions": manager_actions,
        }

        dashboard_payload = {
            "account_summary": {
                "nav": nav,
                "cash": cash,
                "cash_pct": round(cash_pct, 2),
                "gross_exposure": round(gross_exposure, 2),
                "core_exposure": round(core_exposure, 2),
                "tactical_exposure": round(tactical_exposure, 2),
                "current_drawdown": round(current_drawdown, 2),
                "position_count": len(normalized_positions),
                "largest_position": largest_position or "N/A",
                "largest_position_value": round(largest_position_value, 2),
            },
            "portfolio_checks": portfolio_checks,
            "factor_exposure_summary": dict(sorted(factor_counter.items())),
            "candidate_summary": {
                "ticker": asset_symbol,
                "recommended_action": recommended_action,
                "recommended_book": recommended_book,
                "suggested_position_range": safe_str(single_name_score.get("suggested_position_range", "N/A")),
                "existing_position": existing_position,
                "candidate_factor_tags": candidate_tags,
                "blocked_reasons": blocked_reasons,
            },
            "manager_actions": manager_actions,
        }
        return portfolio_directive, dashboard_payload

    def _build_final_state(self, state: Dict[str, Any], text_only: bool = False) -> Dict[str, Any]:
        raw_decision = state.get("final_trade_decision", "")
        decision_payload = self._parse_decision_payload(raw_decision)
        single_name_score = self._normalize_scorecard(decision_payload)
        portfolio_directive, dashboard_payload = self._evaluate_portfolio(state, single_name_score)
        state["decision_payload"] = decision_payload
        state["single_name_score"] = single_name_score
        state["portfolio_directive"] = portfolio_directive
        state["dashboard_payload"] = dashboard_payload

        return {
            "structured_signal_bundle": state.get("structured_signal_bundle", {}),
            "serenity_lens": state.get("serenity_lens", {}),
            "indicator_report": state.get("indicator_report", ""),
            "pattern_report": state.get("pattern_report", ""),
            "trend_report": state.get("trend_report", ""),
            "final_trade_decision": raw_decision,
            "decision_payload": decision_payload,
            "single_name_score": single_name_score,
            "account_state": state.get("account_state", {}),
            "positions": state.get("positions", []),
            "candidates": state.get("candidates", []),
            "portfolio_directive": portfolio_directive,
            "dashboard_payload": dashboard_payload,
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
