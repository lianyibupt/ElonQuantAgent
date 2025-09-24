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
            print("📊 [TrendAgent] 调用趋势图生成工具...")
            trend_result = toolkit.generate_trend_image.invoke({"kline_data": kline_data})
            
            trend_image = trend_result.get("trend_image", "")
            trend_image_filename = trend_result.get("trend_image_filename", "")
            
            print(f"✅ [TrendAgent] 趋势图生成完成")
            print(f"  图像数据长度: {len(trend_image) if trend_image else 0}")
            print(f"  图像描述: {trend_result.get('trend_image_description', '无描述')}")
            
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
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
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
            trend_description = trend_result.get("trend_image_description", "Trend-enhanced chart generated successfully")
            print(f"🤖 [TrendAgent] 调用LLM进行趋势分析，图表描述长度: {len(trend_description)}")
            
            final_response = (analysis_prompt | llm).invoke({
                "trend_result": trend_description
            })
            
            trend_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            print(f"✅ [TrendAgent] LLM趋势分析完成，报告长度: {len(trend_report)}")
            
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


def create_trend_agent_text_only(llm, tools):
    """
    Create a trend analysis agent node for text-only support/resistance and trend line analysis.
    The agent uses an LLM to identify trend patterns without generating charts.
    """
    def trend_agent_node(state):
        time_frame = state['time_frame']
        
        # --- Step 1: 准备K线数据用于文本趋势分析 ---
        messages = state.get("messages", [])
        kline_data = state["kline_data"]
        
        # 提取价格数据用于趋势分析
        price_data = {
            "open_prices": kline_data.get("Open", []),
            "high_prices": kline_data.get("High", []),
            "low_prices": kline_data.get("Low", []),
            "close_prices": kline_data.get("Close", []),
            "datetimes": kline_data.get("Datetime", [])
        }
        
        # 计算趋势相关统计信息
        recent_closes = price_data["close_prices"][-20:] if len(price_data["close_prices"]) > 20 else price_data["close_prices"]
        recent_highs = price_data["high_prices"][-20:] if len(price_data["high_prices"]) > 20 else price_data["high_prices"]
        recent_lows = price_data["low_prices"][-20:] if len(price_data["low_prices"]) > 20 else price_data["low_prices"]
        
        # 计算简单移动平均线
        sma_short = sum(recent_closes[-5:]) / 5 if len(recent_closes) >= 5 else None
        sma_long = sum(recent_closes[-20:]) / 20 if len(recent_closes) >= 20 else None
        
        # 计算价格变化
        price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100) if recent_closes and recent_closes[0] != 0 else 0
        
        # 计算支撑阻力位（简化版本）
        support_level = min(recent_lows) if recent_lows else None
        resistance_level = max(recent_highs) if recent_highs else None
        
        print(f"📊 [TrendAgent-Text] 准备进行文本趋势分析，数据长度: {len(price_data['close_prices'])}")

        # --- Step 2: 生成趋势分析报告（文本模式）---
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是高频交易的趋势分析专家。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"时间框架: {time_frame}\n\n"
                "基于以下价格数据进行趋势分析:\n"
                "- 开盘价: {open_prices}\n"
                "- 最高价: {high_prices}\n"
                "- 最低价: {low_prices}\n"
                "- 收盘价: {close_prices}\n"
                "- 时间戳: {datetimes}\n\n"
                "技术统计信息:\n"
                "- 近期价格变化: {price_change:.2f}%\n"
                "- 短期均线(SMA5): {sma_short:.2f}\n"
                "- 长期均线(SMA20): {sma_long:.2f}\n"
                "- 支撑位: {support_level:.2f}\n"
                "- 阻力位: {resistance_level:.2f}\n\n"
                "请提供全面的中文趋势分析报告，包括:\n"
                "1. 整体趋势方向（看涨、看跌或横盘）\n"
                "2. 关键支撑和阻力位分析\n"
                "3. 趋势强度和动量评估\n"
                "4. 潜在突破或跌破点\n"
                "5. 基于趋势分析的交易建议\n\n"
                "专注于为高频交易决策提供可操作的中文见解。"
            )
        ])
        
        try:
            print(f"🤖 [TrendAgent-Text] 调用LLM进行文本趋势分析...")
            
            final_response = (analysis_prompt | llm).invoke({
                "open_prices": str(price_data["open_prices"][-20:]),  # 显示最近20个数据点
                "high_prices": str(price_data["high_prices"][-20:]),
                "low_prices": str(price_data["low_prices"][-20:]),
                "close_prices": str(price_data["close_prices"][-20:]),
                "datetimes": str(price_data["datetimes"][-20:]),
                "price_change": price_change,
                "sma_short": sma_short if sma_short is not None else "N/A",
                "sma_long": sma_long if sma_long is not None else "N/A",
                "support_level": support_level if support_level is not None else "N/A",
                "resistance_level": resistance_level if resistance_level is not None else "N/A"
            })
            
            trend_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            print(f"✅ [TrendAgent-Text] LLM趋势分析完成，报告长度: {len(trend_report)}")
            
        except Exception as e:
            trend_report = f"Error generating trend analysis: {str(e)}"
            print(f"❌ [TrendAgent-Text] 趋势分析失败: {str(e)}")

        # 更新state并返回（不包含图像数据）
        state.update({
            "messages": messages,
            "trend_report": trend_report,
            "trend_image": "",
            "trend_image_filename": "",
        })
        
        return state

    return trend_agent_node