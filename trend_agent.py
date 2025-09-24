from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
import json
import time
import copy
from openai import RateLimitError

def create_trend_agent(llm, tools):
    """
    Create a trend analysis agent node for support/resistance and trend line analysis.
    The agent uses an LLM and a trend chart generation tool to identify trend patterns.
    """
    def trend_agent_node(state):
        time_frame = state['time_frame']
        
        # --- Step 1: 直接调用趋势图生成工具 ---
        messages = state.get("messages", [])
        kline_data = state["kline_data"]
        
        try:
            # 直接调用趋势图生成工具
            from graph_util import TechnicalTools
            toolkit = TechnicalTools()
            trend_result = toolkit.generate_trend_image.invoke({"kline_data": kline_data})
            
            trend_image = trend_result.get("trend_image", "")
            trend_image_filename = trend_result.get("trend_image_filename", "")
            
        except Exception as e:
            print(f"Error generating trend chart: {str(e)}")
            trend_image = ""
            trend_image_filename = ""
            trend_result = {"error": str(e)}

        # --- Step 2: 生成趋势分析报告 ---
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是高频交易的趋势分析专家。请用中文回答。"
                f"趋势图表是基于{time_frame}间隔数据生成的。\n\n"
                "图表生成结果: {trend_result}\n\n"
                "分析趋势数据并提供全面的中文报告，包括:\n"
                "1. 整体趋势方向（看涨、看跌或横盘）\n"
                "2. 关键支撑和阻力位\n"
                "3. 趋势强度和动量\n"
                "4. 潜在突破或跌破点\n"
                "5. 基于趋势分析的交易建议\n\n"
                "专注于为高频交易决策提供可操作的中文见解。"
            )
        ])
        
        try:
            final_response = (analysis_prompt | graph_llm).invoke({
                "trend_result": json.dumps(trend_result, indent=2)
            })
            
            trend_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
        except Exception as e:
            trend_report = f"Error generating trend analysis: {str(e)}\n\nTrend result: {json.dumps(trend_result, indent=2)}"

        # 更新state并返回
        state.update({
            "messages": messages,
            "trend_report": trend_report,
            "trend_image": trend_image,
            "trend_image_filename": trend_image_filename,
        })
        
        return state

    return trend_agent_node