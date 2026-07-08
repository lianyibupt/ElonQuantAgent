import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import RateLimitError


def invoke_tool_with_retry(tool_fn, tool_args, retries=3, wait_sec=4):
    """
    Invoke a tool function with retries if the result is missing an image.
    """
    for attempt in range(retries):
        result = tool_fn.invoke(tool_args)
        img_b64 = result.get("pattern_image")
        if img_b64:
            return result
        print(
            f"Tool returned no image, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_sec)
    raise RuntimeError("Tool failed to generate image after multiple retries")


def create_pattern_agent(llm, tools):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses precomputed images from state or falls back to tool generation.
    """

    def pattern_agent_node(state):
        time_frame = state['time_frame']
        pattern_text = """
        请参考以下经典K线形态：

        1. 倒头肩形态：三个低点，中间最低，结构对称，通常预示即将上涨。
        2. 双底形态：两个相似的低点，中间有反弹，形成'W'形。
        3. 圆弧底：价格逐渐下跌后逐渐上升，形成'U'形。
        4. 潜伏底：水平整理后突然向上突破。
        5. 下降楔形：价格向下收窄，通常向上突破。
        6. 上升楔形：价格缓慢上升但收敛，经常向下突破。
        7. 上升三角形：上升支撑线配合水平阻力线，突破通常向上。
        8. 下降三角形：下降阻力线配合水平支撑线，通常向下突破。
        9. 看涨旗形：急涨后短暂向下整理，然后继续上涨。
        10. 看跌旗形：急跌后短暂向上整理，然后继续下跌。
        11. 矩形：价格在水平支撑和阻力之间波动。
        12. 岛形反转：两个相反方向的价格缺口形成孤立的价格岛。
        13. V形反转：急跌后急涨，或相反。
        14. 圆顶/圆底：逐渐见顶或见底，形成弧形形态。
        15. 扩散三角形：高点和低点越来越宽，表示波动加剧。
        16. 对称三角形：高点和低点向顶点收敛，通常伴随突破。
        """

        # --- Step 1: 直接调用图表生成工具 ---
        messages = state.get("messages", [])
        kline_data = state["kline_data"]
        
        try:
            # 直接调用图表生成工具 - 使用tools参数中的第一个工具
            from utils.graph_util import TechnicalTools
            toolkit = TechnicalTools()
            print("🖼️  [PatternAgent] 调用图表生成工具...")
            chart_result = invoke_tool_with_retry(
                toolkit.generate_kline_image, 
                {"kline_data": kline_data}
            )
            
            pattern_image = chart_result.get("pattern_image", "")
            pattern_image_filename = chart_result.get("pattern_image_filename", "")
            
            print(f"✅ [PatternAgent] 图表生成完成")
            print(f"  图像数据长度: {len(pattern_image) if pattern_image else 0}")
            print(f"  图像描述: {chart_result.get('pattern_image_description', '无描述')}")
            
        except Exception as e:
            # 更健壮的错误处理
            try:
                error_msg = str(e)
                if isinstance(error_msg, bytes):
                    error_msg = error_msg.decode('utf-8', errors='replace')
                else:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
            except:
                error_msg = "Unknown encoding error"
                
            print(f"Error generating pattern chart: {error_msg}")
            pattern_image = ""
            pattern_image_filename = ""
            chart_result = {"error": error_msg}

        # --- Step 2: 根据交易策略生成形态分析报告 ---
        trading_strategy = state.get('trading_strategy', 'high_frequency')
        structured_signal_bundle = state.get('structured_signal_bundle', {}) or {}
        pattern_signal_view = {
            "price": structured_signal_bundle.get("price", {}),
            "entry": structured_signal_bundle.get("entry", {}),
            "levels": structured_signal_bundle.get("levels", {}),
            "volatility": structured_signal_bundle.get("volatility", {}),
            "evidence": structured_signal_bundle.get("evidence", []),
            "contradictions": structured_signal_bundle.get("contradictions", []),
        }
        
        # 从kline_data提取实际价格范围,确保LLM使用真实数据
        kline_data = state["kline_data"]
        close_prices = kline_data.get("Close", [])
        if close_prices:
            price_min = min(close_prices)
            price_max = max(close_prices)
            price_current = close_prices[-1]
            price_range_info = f"\n\n**重要: 实际价格数据**\n- 价格范围: ${price_min:.2f} - ${price_max:.2f}\n- 当前价格: ${price_current:.2f}\n- 数据点数: {len(close_prices)}\n"
        else:
            price_range_info = ""
        
        if trading_strategy == 'low_frequency':
            # 低频交易策略提示词
            system_prompt = (
                "你是低频交易的K线形态识别专家,专注于长期形态和价格模式分析。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"K线图表是基于{time_frame}间隔数据生成的。\n"
                f"{price_range_info}\n"
                "结构化信号层摘要: {pattern_signal_view}\n"
                "图表生成结果: {chart_result}\n\n"
                "分析K线形态并只输出JSON，字段包括：pattern、completion、direction、strength_score、target_zone、evidence、contradictions、invalid_if、summary。\n\n"
                "**注意: 请基于上述实际价格数据进行分析,不要仅凭图表视觉推断价格。**\n"
                "专注于为低频交易决策提供可操作证据,重点关注长期形态。"
            )
        else:
            # 高频交易策略提示词
            system_prompt = (
                "你是高频交易的K线形态识别专家。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"K线图表是基于{time_frame}间隔数据生成的。\n"
                f"{price_range_info}\n"
                "结构化信号层摘要: {pattern_signal_view}\n"
                "图表生成结果: {chart_result}\n\n"
                "分析K线形态并只输出JSON，字段包括：pattern、completion、direction、strength_score、target_zone、evidence、contradictions、invalid_if、summary。\n\n"
                "**注意: 请基于上述实际价格数据进行分析,不要仅凭图表视觉推断价格。**\n"
                "专注于为高频交易决策提供可操作证据。"
            )
            
        # 创建提示词模板
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_prompt
            )
        ])
        
        try:
            chart_description = chart_result.get("pattern_image_description", "Candlestick chart generated successfully")
            # 确保图表描述使用UTF-8编码
            if isinstance(chart_description, str):
                chart_description = chart_description.encode('utf-8', errors='replace').decode('utf-8')
            elif isinstance(chart_description, bytes):
                chart_description = chart_description.decode('utf-8', errors='replace')
                
            print(f"🤖 [PatternAgent] 调用LLM进行形态分析，图表描述长度: {len(chart_description)}")
            
            final_response = (analysis_prompt | llm).invoke({
                "chart_result": chart_description,
                "pattern_signal_view": json.dumps(pattern_signal_view, ensure_ascii=False),
                "pattern_descriptions": pattern_text
            })
            
            pattern_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            # 确保报告使用UTF-8编码
            if isinstance(pattern_report, str):
                pattern_report = pattern_report.encode('utf-8', errors='replace').decode('utf-8')
            elif isinstance(pattern_report, bytes):
                pattern_report = pattern_report.decode('utf-8', errors='replace')
                
            print(f"✅ [PatternAgent] LLM形态分析完成，报告长度: {len(pattern_report)}")
            
        except Exception as e:
            # 更健壮的错误处理
            try:
                error_msg = str(e)
                if isinstance(error_msg, bytes):
                    error_msg = error_msg.decode('utf-8', errors='replace')
                else:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
            except:
                error_msg = "Unknown encoding error"
                
            # 安全地处理chart_result
            try:
                # 创建副本以避免修改原始数据
                safe_chart_result = chart_result.copy() if isinstance(chart_result, dict) else {}
                # 移除base64图片数据
                if "pattern_image" in safe_chart_result:
                    safe_chart_result["pattern_image"] = "<base64_image_removed>"
                
                chart_result_str = json.dumps(safe_chart_result, indent=2, ensure_ascii=False)
                chart_result_str = chart_result_str.encode('utf-8', errors='replace').decode('utf-8')
            except:
                chart_result_str = "Unable to display chart result"
                
            pattern_report = f"Error generating pattern analysis: {error_msg}\n\nChart result: {chart_result_str}"

        # 更新state并返回
        state.update({
            "messages": messages,
            "pattern_report": pattern_report,
            "pattern_image": pattern_image,
            "pattern_image_filename": pattern_image_filename,
        })
        
        return state

    return pattern_agent_node


def create_pattern_agent_text_only(llm, tools):
    """
    Create a pattern recognition agent node for text-only candlestick pattern analysis.
    The agent uses an LLM to identify classic trading patterns without generating charts.
    """
    def pattern_agent_node(state):
        time_frame = state['time_frame']
        pattern_text = """
        请参考以下经典K线形态：

        1. 倒头肩形态：三个低点，中间最低，结构对称，通常预示即将上涨。
        2. 双底形态：两个相似的低点，中间有反弹，形成'W'形。
        3. 圆弧底：价格逐渐下跌后逐渐上升，形成'U'形。
        4. 潜伏底：水平整理后突然向上突破。
        5. 下降楔形：价格向下收窄，通常向上突破。
        6. 上升楔形：价格缓慢上升但收敛，经常向下突破。
        7. 上升三角形：上升支撑线配合水平阻力线，突破通常向上。
        8. 下降三角形：下降阻力线配合水平支撑线，通常向下突破。
        9. 看涨旗形：急涨后短暂向下整理，然后继续上涨。
        10. 看跌旗形：急跌后短暂向上整理，然后继续下跌。
        11. 矩形：价格在水平支撑和阻力之间波动。
        12. 岛形反转：两个相反方向的价格缺口形成孤立的价格岛。
        13. V形反转：急跌后急涨，或相反。
        14. 圆顶/圆底：逐渐见顶或见底，形成弧形形态。
        15. 扩散三角形：高点和低点越来越宽，表示波动加剧。
        16. 对称三角形：高点和低点向顶点收敛，通常伴随突破。
        """

        # --- Step 1: 准备K线数据用于文本分析 ---
        messages = state.get("messages", [])
        kline_data = state["kline_data"]
        
        # 提取价格数据用于文本分析
        price_data = {
            "open_prices": kline_data.get("Open", []),
            "high_prices": kline_data.get("High", []),
            "low_prices": kline_data.get("Low", []),
            "close_prices": kline_data.get("Close", []),
            "datetimes": kline_data.get("Datetime", [])
        }
        
        # 计算一些基本统计信息
        recent_closes = price_data["close_prices"][-10:] if len(price_data["close_prices"]) > 10 else price_data["close_prices"]
        price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100) if recent_closes else 0
        
        print(f"📊 [PatternAgent-Text] 准备进行文本形态分析，数据长度: {len(price_data['close_prices'])}")

        # --- Step 2: 根据交易策略生成模式分析报告（文本模式）---
        trading_strategy = state.get('trading_strategy', 'high_frequency')
        
        if trading_strategy == 'low_frequency':
            # 低频交易策略提示词
            system_prompt = (
                "你是一位专业的低频交易形态识别助手，专注于长期趋势和价格行为分析。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"时间框架: {time_frame}\n\n"
                "基于以下价格数据进行形态分析:\n"
                "- 开盘价: {open_prices}\n"
                "- 最高价: {high_prices}\n"
                "- 最低价: {low_prices}\n"
                "- 收盘价: {close_prices}\n"
                "- 时间戳: {datetimes}\n\n"
                "近期价格变化: {price_change:.2f}%\n\n"
                "请参考以下经典形态描述:\n\n"
                "{pattern_descriptions}\n\n"
                "请提供详细的中文形态分析报告，包括:\n"
                "1. 识别的形态（如有）\n"
                "2. 形态可靠性和强度\n"
                "3. 长期交易含义\n"
                "4. 长期关键支撑/阻力位\n"
                "5. 基于价格数据的分析推理\n"
                "6. 形态对未来1-6个月价格走势的影响"
            )
        else:
            # 高频交易策略提示词
            system_prompt = (
                "你是一位专门识别经典高频交易形态的交易形态识别助手。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"时间框架: {time_frame}\n\n"
                "基于以下价格数据进行形态分析:\n"
                "- 开盘价: {open_prices}\n"
                "- 最高价: {high_prices}\n"
                "- 最低价: {low_prices}\n"
                "- 收盘价: {close_prices}\n"
                "- 时间戳: {datetimes}\n\n"
                "近期价格变化: {price_change:.2f}%\n\n"
                "请参考以下经典形态描述:\n\n"
                "{pattern_descriptions}\n\n"
                "请提供详细的中文形态分析报告，包括:\n"
                "1. 识别的形态（如有）\n"
                "2. 形态可靠性和强度\n"
                "3. 交易含义\n"
                "4. 关键支撑/阻力位\n"
                "5. 基于价格数据的分析推理"
            )
            
        # 创建提示词模板
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_prompt
            )
        ])
        
        try:
            print(f"🤖 [PatternAgent-Text] 调用LLM进行文本形态分析...")
            
            # 确保所有字符串参数使用UTF-8编码
            open_prices_str = str(price_data["open_prices"][-20:]).encode('utf-8', errors='replace').decode('utf-8')
            high_prices_str = str(price_data["high_prices"][-20:]).encode('utf-8', errors='replace').decode('utf-8')
            low_prices_str = str(price_data["low_prices"][-20:]).encode('utf-8', errors='replace').decode('utf-8')
            close_prices_str = str(price_data["close_prices"][-20:]).encode('utf-8', errors='replace').decode('utf-8')
            datetimes_str = str(price_data["datetimes"][-20:]).encode('utf-8', errors='replace').decode('utf-8')
            
            final_response = (analysis_prompt | llm).invoke({
                "open_prices": open_prices_str,
                "high_prices": high_prices_str,
                "low_prices": low_prices_str,
                "close_prices": close_prices_str,
                "datetimes": datetimes_str,
                "price_change": price_change,
                "pattern_descriptions": pattern_text
            })
            
            pattern_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            # 确保报告使用UTF-8编码
            if isinstance(pattern_report, str):
                pattern_report = pattern_report.encode('utf-8', errors='replace').decode('utf-8')
            print(f"✅ [PatternAgent-Text] LLM形态分析完成，报告长度: {len(pattern_report)}")
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            pattern_report = f"Error generating pattern analysis: {error_msg}"
            print(f"❌ [PatternAgent-Text] 形态分析失败: {error_msg}")

        # 更新state并返回（不包含图像数据）
        state.update({
            "messages": messages,
            "pattern_report": pattern_report,
            "pattern_image": "",
            "pattern_image_filename": "",
        })
        
        return state

    return pattern_agent_node
