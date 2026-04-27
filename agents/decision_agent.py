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
            strategy_context = "你是一位资深的低频交易决策专家，持有周期以月为单位，最长可达半年。"
            horizon_hint = "1-6个月"
            focus_hint = "更重视中期趋势延续、结构性逻辑和较低换手。"
        else:
            strategy_context = "你是一位资深的高频交易决策专家，最短持有2天，最长持有1个月。"
            horizon_hint = "2天-1个月"
            focus_hint = "更重视节奏、入场质量、波动控制和短中期趋势延续。"

        system_prompt = (
            f"{strategy_context}"
            "基于以下综合分析报告，做出最终的交易决策。请用中文回答。\n\n"
            "股票代码: {stock_name}\n"
            "时间周期: {time_frame}\n\n"
            "技术指标分析:\n{indicator_report}\n\n"
            "形态分析:\n{pattern_report}\n\n"
            "趋势分析:\n{trend_report}\n\n"
            f"请输出严格合法的 JSON，不要输出 JSON 之外的任何文字。时间周期预期可参考：{horizon_hint}。{focus_hint}\n"
            "分数字段统一使用 0-100 的整数。\n"
            "字段含义：trend_score=趋势质量，entry_score=入场质量，valuation_stretch_score=估值或拉伸状态，"
            "catalyst_score=催化质量，volatility_score=波动认知。\n"
            "recommended_book 只能填写：核心仓、战术仓、观察。\n"
            "recommended_action 只能填写：买入、加仓、持有、减仓、退出、观察。\n"
            "decision 只能填写：买入、卖出、持有。\n"
            "confidence 只能填写：高、中、低。\n"
            "请按照以下 JSON 格式输出：\n"
            "{{\n"
            '  "decision": "买入/卖出/持有",\n'
            '  "confidence": "高/中/低",\n'
            '  "risk_reward_ratio": "X:Y",\n'
            '  "forecast_horizon": "预测时间段",\n'
            '  "justification": "详细的中文理由说明",\n'
            '  "recommended_book": "核心仓/战术仓/观察",\n'
            '  "recommended_action": "买入/加仓/持有/减仓/退出/观察",\n'
            '  "trend_score": 0,\n'
            '  "entry_score": 0,\n'
            '  "valuation_stretch_score": 0,\n'
            '  "catalyst_score": 0,\n'
            '  "volatility_score": 0,\n'
            '  "suggested_position_range": "例如 3% - 5%",\n'
            '  "invalidation_price": "例如 125.50"\n'
            "}}"
        )

        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        try:
            final_response = (decision_prompt | llm).invoke({
                "stock_name": stock_name,
                "time_frame": time_frame,
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
