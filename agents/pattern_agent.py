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
        price_action_context = """
        请按以下 Price Action / 市场结构框架分析，不要机械套用形态模板：

        ## 市场结构判定（最优先）
        - 上升结构：更高的高点(HH) + 更高的低点(HL)
        - 下降结构：更低的高点(LH) + 更低的低点(LL)
        - 结构破坏(MSB/Market Structure Break)：价格突破前一个关键摆动点，可能预示趋势反转
        - 震荡结构：无明显HH/HL或LH/LL序列

        ## 关键价位识别
        - 最近50根K线的摆动高点和摆动低点
        - 多次测试但未突破的价位（强支撑/阻力）
        - 当前价格与关键价位的距离

        ## K线组合信号（仅报高可靠性的，需下一根确认）
        - 吞没形态、锤子线/倒锤子、十字星、孕线
        - 强调：单个K线信号需要下一根K线确认

        ## 经典形态（仅报告已完成的）
        - 头肩顶/底、双顶/双底、三角形、旗形/楔形
        - 必须明确标注状态：未完成 / 已完成 / 已失效
        - 如果没有明确形态，诚实说"未发现明确的经典形态"

        ## 禁止行为
        - 不要为套模板而强行匹配形态
        - 不要忽视价格结构只看局部形态
        - 不要在图表中看到十字星就说反转
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
            system_prompt = (
                "你是低频交易的价格行为(Price Action)分析专家，专注于市场结构和长期形态。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"K线图表是基于{time_frame}间隔数据生成的。\n"
                f"{price_range_info}\n"
                "图表生成结果: {chart_result}\n\n"
                "## 分析要求（按优先级）\n"
                "1. **市场结构**: 判定HH/HL上升结构、LH/LL下降结构、或震荡，这是最重要的结论\n"
                "2. **关键价位**: 从图表中识别摆动高/低点，标注强支撑和阻力\n"
                "3. **K线组合**: 仅报告吞没、锤子线等高质量信号，需说明是否已确认\n"
                "4. **经典形态**: 仅报告已完成的形态，标注完成度/可靠性\n"
                "5. **交易含义**: 结合结构、价位、形态给出1-6个月展望\n\n"
                "**注意: 结合实际价格数据分析，无明确形态时诚实说明。**\n"
                "{price_action_context}"
            )
        else:
            system_prompt = (
                "你是高频交易的价格行为(Price Action)分析专家。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"K线图表是基于{time_frame}间隔数据生成的。\n"
                f"{price_range_info}\n"
                "图表生成结果: {chart_result}\n\n"
                "## 分析要求（按优先级）\n"
                "1. **市场结构**: 判定HH/HL、LH/LL或震荡，这是最重要的结论\n"
                "2. **关键价位**: 识别摆动高/低点，标注强支撑和阻力\n"
                "3. **K线组合**: 仅报告高质量信号（吞没、锤子线等），需说明确认状态\n"
                "4. **经典形态**: 仅报告已完成的形态，标注完成度\n"
                "5. **交易含义**: 结合结构和价位给出短线交易建议\n\n"
                "**注意: 结合实际价格数据分析，无明确形态时诚实说明。**\n"
                "{price_action_context}"
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
                "price_action_context": price_action_context,
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
        price_action_context = """
        请按以下 Price Action / 市场结构框架分析，不要机械套用形态模板：

        ## 市场结构判定（最优先）
        - 上升结构：更高的高点(HH) + 更高的低点(HL)
        - 下降结构：更低的高点(LH) + 更低的低点(LL)
        - 结构破坏(MSB)：价格突破前一个关键摆动点，可能预示趋势反转
        - 震荡结构：无明显HH/HL或LH/LL序列

        ## 关键价位
        - 多次测试的价位是强支撑/阻力
        - 当前价格与关键价位的距离决定盈亏比

        ## K线组合（仅高可靠性）
        - 吞没、锤子线/倒锤子、十字星、孕线
        - 需要下一根K线确认

        ## 经典形态（仅完整形态）
        - 头肩、双顶/底、三角形、旗形、楔形
        - 标注：未完成/已完成/已失效
        - 无形态时如实说"未发现"

        ## 禁止
        - 不要强行匹配形态
        - 不要忽视价格结构只看形态
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
            system_prompt = (
                "你是低频交易的价格行为分析专家，专注于市场结构和长期形态。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"时间框架: {time_frame}\n\n"
                "基于以下价格数据进行形态分析:\n"
                "- 开盘价: {open_prices}\n"
                "- 最高价: {high_prices}\n"
                "- 最低价: {low_prices}\n"
                "- 收盘价: {close_prices}\n"
                "- 时间戳: {datetimes}\n\n"
                "近期价格变化: {price_change:.2f}%\n\n"
                "{price_action_context}\n\n"
                "## 分析要求（按优先级逐项判定）\n"
                "1. **市场结构**: 从价格序列中识别HH/HL或LH/LL，判断当前结构状态\n"
                "2. **关键价位**: 标注被多次测试的价位，评估支撑/阻力强度\n"
                "3. **K线组合**: 仅报告高可靠性信号，说明确认状态\n"
                "4. **经典形态**: 仅报告完整的形态，标注「未完成/已完成/已失效」\n"
                "5. **交易含义**: 结合结构和价位给出1-6个月的交易展望\n\n"
                "无明确结论时诚实说明，不要为套模板而强行匹配。"
            )
        else:
            system_prompt = (
                "你是价格行为(Price Action)分析专家。请用中文回答。"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"时间框架: {time_frame}\n\n"
                "基于以下价格数据进行形态分析:\n"
                "- 开盘价: {open_prices}\n"
                "- 最高价: {high_prices}\n"
                "- 最低价: {low_prices}\n"
                "- 收盘价: {close_prices}\n"
                "- 时间戳: {datetimes}\n\n"
                "近期价格变化: {price_change:.2f}%\n\n"
                "{price_action_context}\n\n"
                "## 分析要求（逐项判定）\n"
                "1. **市场结构**: 识别HH/HL或LH/LL，判断当前结构\n"
                "2. **关键价位**: 标注摆动高/低点，评估支撑/阻力\n"
                "3. **K线组合**: 仅高可靠性信号，说明确认状态\n"
                "4. **经典形态**: 仅完整形态，标注完成度\n"
                "5. **交易含义**: 短线交易建议\n\n"
                "无明确结论时诚实说明，不要强行匹配。"
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
                "price_action_context": price_action_context,
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