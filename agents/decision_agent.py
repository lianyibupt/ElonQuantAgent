"""
Agent for final trading decision synthesis.
Combines indicator, pattern, and trend analysis into a structured scorecard.
"""

import json

from langchain_core.prompts import ChatPromptTemplate


DEFAULT_SCORECARD = {
    "decision": "持有",
    "confidence": "低",
    "risk_reward_ratio": "1:1",
    "forecast_horizon": "未知",
    "justification": "模型未能生成稳定结论，默认保持观望。",
    "recommended_book": "观察",
    "recommended_action": "观察",
    "trend_score": 50,
    "entry_score": 50,
    "valuation_stretch_score": 50,
    "catalyst_score": 50,
    "volatility_score": 50,
    "suggested_position_range": "0% - 0%",
    "invalidation_price": "待确认",
    "checklist_summary": {
        "trend_checks_passed": 0,
        "entry_checks_passed": 0,
        "risk_checks_passed": 0,
        "trend_quality": "低",
        "entry_quality": "低",
        "risk_pass": False,
    },
}


def create_decision_agent(llm, tools):
    """Create a decision synthesis agent node."""

    def decision_agent_node(state):
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]
        indicator_report = state.get("indicator_report", "No indicator analysis available")
        pattern_report = state.get("pattern_report", "No pattern analysis available")
        trend_report = state.get("trend_report", "No trend analysis available")
        trading_strategy = state.get("trading_strategy", "high_frequency")

        if trading_strategy == "low_frequency":
            strategy_context = "你是一位资深的中线持仓交易决策专家，持有周期以月为单位，最长可达半年。"
            horizon_hint = "1-6个月"
            checklist_emphasis = "趋势延续 > 入场时机，结构逻辑 > 短期波动"
        else:
            strategy_context = "你是一位资深的摆动交易(Swing Trading)决策专家，最短持有2天，最长持有1个月。"
            horizon_hint = "2天-1个月"
            checklist_emphasis = "入场质量 > 趋势评分，波动控制 > 收益最大化"

        system_prompt = (
            f"{strategy_context}\n"
            "你不是在\"打分\"，而是在执行逐项判定的决策清单。请用中文输出。\n\n"
            "股票代码: {stock_name}\n"
            "时间周期: {time_frame}\n"
            f"决策优先级: {checklist_emphasis}\n\n"
            "=== 三项分析报告 ===\n\n"
            "技术指标分析(结构化JSON):\n{indicator_report}\n\n"
            "形态/价格行为分析:\n{pattern_report}\n\n"
            "趋势分析:\n{trend_report}\n\n"
            "=== 决策清单（逐条判定，不可跳过）===\n\n"
            "## A. 趋势维度\n"
            "- [ ] 从指标报告中提取 trend_state：是否为\"趋势上涨\"？\n"
            "- [ ] ADX 是否 > 25？（趋势市而非震荡市）\n"
            "- [ ] EMA 是否为多头排列？（EMA20 > EMA50 > EMA200）\n"
            "- [ ] 趋势报告中是否确认了上升趋势结构？\n"
            "→ 满足 ≥3 项 = 趋势质量高，2 项 = 中，≤1 项 = 低\n\n"
            "## B. 入场维度\n"
            "- [ ] RSI 不在超买区（不 > 70），有入场空间？\n"
            "- [ ] 价格距关键支撑位 < 1.5 ATR（不追高）？\n"
            "- [ ] MACD 是否金叉或柱状图转正？\n"
            "- [ ] 形态分析是否报告了结构破坏(MSB)向上或突破确认？\n"
            "→ 满足 ≥3 项 = 入场质量高，2 项 = 中，≤1 项 = 低\n\n"
            "## C. 风险维度\n"
            "- [ ] 基于 ATR 的止损距离是否给出 ≥1:2 的盈亏比？\n"
            "- [ ] 当前价格距上方阻力是否仍有空间？\n"
            "- [ ] invalidation_price 是否在关键支撑下方（给噪音留余地）？\n"
            "→ 3 项全满足 = 风控通过，否则 = 风控不通过\n\n"
            "## D. 催化剂维度（仅在低频策略中影响评分）\n"
            "- [ ] 是否有财报、重大新闻、行业政策等明确催化事件？\n"
            "- [ ] 若无明确催化剂，catalyst_score 默认为 50\n\n"
            "=== 决策规则（严格按规则输出，不可自由发挥）===\n\n"
            "| 趋势 | 入场 | 风控 | 决策 | 仓位类型 |\n"
            "|------|------|------|------|----------|\n"
            "| 高 | 高 | 通过 | 买入 | 核心仓 |\n"
            "| 高 | 中 | 通过 | 买入 | 战术仓 |\n"
            "| 中 | 高 | 通过 | 买入 | 战术仓 |\n"
            "| 高/中 | 低 | 通过 | 持有(已有)/观察(新建) | 观察 |\n"
            "| 任意 | 任意 | 不通过 | 观察 | 观察 |\n"
            "| 低 | 低 | 任意 | 持有(已有考虑减仓)/观察(新建) | 观察 |\n"
            "| 已有持仓 + 趋势转低 + invalidation触发 | 卖出 | — |\n\n"
            "=== invalidation_price 计算规则 ===\n"
            "- 多头入场: 最近明显摆动低点 - 0.3 * ATR\n"
            "- 已有持仓: 最近一个结构性低点 - 0.5 * ATR\n"
            "- 必须报具体数值，禁止用\"待确认\"\n\n"
            "=== 输出格式（仅 JSON，无其他文字）===\n"
            "{{\n"
            '  "decision": "买入/卖出/持有",\n'
            '  "confidence": "高/中/低",\n'
            '  "risk_reward_ratio": "X:Y",\n'
            '  "forecast_horizon": "{horizon_hint}",\n'
            '  "justification": "基于清单判定的理由，引用具体指标数值",\n'
            '  "recommended_book": "核心仓/战术仓/观察",\n'
            '  "recommended_action": "买入/加仓/持有/减仓/退出/观察",\n'
            '  "trend_score": 0,\n'
            '  "entry_score": 0,\n'
            '  "valuation_stretch_score": 0,\n'
            '  "catalyst_score": 0,\n'
            '  "volatility_score": 0,\n'
            '  "suggested_position_range": "例如 3% - 5%",\n'
            '  "invalidation_price": "必须填写具体数值，例如 125.50",\n'
            '  "checklist_summary": {{\n'
            '    "trend_checks_passed": 0,\n'
            '    "entry_checks_passed": 0,\n'
            '    "risk_checks_passed": 0,\n'
            '    "trend_quality": "高/中/低",\n'
            '    "entry_quality": "高/中/低",\n'
            '    "risk_pass": true\n'
            '  }}\n'
            "}}\n\n"
            f'forecast_horizon 应填写 "{horizon_hint}"'
        )

        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        try:
            final_response = (decision_prompt | llm).invoke({
                "stock_name": stock_name,
                "time_frame": time_frame,
                "horizon_hint": horizon_hint,
                "indicator_report": indicator_report,
                "pattern_report": pattern_report,
                "trend_report": trend_report,
            })
            decision_content = (
                final_response.content if hasattr(final_response, "content") else str(final_response)
            )
        except Exception as e:
            try:
                error_msg = str(e)
                if isinstance(error_msg, bytes):
                    error_msg = error_msg.decode("utf-8", errors="replace")
                else:
                    error_msg = error_msg.encode("utf-8", errors="replace").decode("utf-8")
            except Exception:
                error_msg = "Unknown encoding error"

            fallback_payload = DEFAULT_SCORECARD.copy()
            fallback_payload["justification"] = f"Error generating decision: {error_msg}"
            decision_content = json.dumps(fallback_payload, indent=2, ensure_ascii=False)

        state.update({
            "messages": state.get("messages", []),
            "final_trade_decision": decision_content,
        })
        return state

    return decision_agent_node
