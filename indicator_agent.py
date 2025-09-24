"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
import json

def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """
    def indicator_agent_node(state):
        time_frame = state['time_frame']
        
        # --- Step 1: 直接调用所有技术指标工具 ---
        messages = state["messages"]
        tool_results = []
        kline_data = state["kline_data"]
        
        try:
            # 调用所有技术指标工具
            macd_result = toolkit.compute_macd.invoke({"kline_data": kline_data})
            rsi_result = toolkit.compute_rsi.invoke({"kline_data": kline_data})
            roc_result = toolkit.compute_roc.invoke({"kline_data": kline_data})
            stoch_result = toolkit.compute_stoch.invoke({"kline_data": kline_data})
            willr_result = toolkit.compute_willr.invoke({"kline_data": kline_data})
            
            tool_results = [
                f"MACD Analysis: {macd_result}",
                f"RSI Analysis: {rsi_result}",
                f"ROC Analysis: {roc_result}",
                f"Stochastic Analysis: {stoch_result}",
                f"Williams %R Analysis: {willr_result}"
            ]
            
        except Exception as e:
            tool_results = [f"Error computing indicators: {str(e)}"]

        # --- Step 2: 生成分析报告 ---
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一位在时间敏感条件下运作的高频交易(HFT)分析助手。"
                "基于以下技术指标结果，提供一份全面的中文分析报告。"
                "总结MACD、RSI、ROC、随机指标和威廉指标的关键发现。"
                "为高频交易决策提供可操作的中文见解。\n\n"
                f"OHLC数据来自{time_frame}间隔，反映了最近的市场行为。\n\n"
                "技术指标结果:\n{tool_results}\n\n"
                "请用中文详细分析每个指标的含义和交易信号。"
            )
        ])
        
        try:
            final_response = (analysis_prompt | llm).invoke({
                "tool_results": "\n".join(tool_results)
            })
            
            indicator_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
        except Exception as e:
            indicator_report = f"Error generating indicator analysis: {str(e)}\n\nRaw results:\n" + "\n".join(tool_results)
        
        # 更新state并返回
        state.update({
            "messages": messages,
            "indicator_report": indicator_report,
        })
        
        return state

    return indicator_agent_node