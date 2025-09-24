from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
import json
import time
import copy
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
        print(f"Tool returned no image, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...")
        time.sleep(wait_sec)
    raise RuntimeError("Tool failed to generate image after multiple retries")


def create_pattern_agent(llm, tools):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses an LLM and a chart generation tool to identify classic trading patterns.
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
            from graph_util import TechnicalTools
            toolkit = TechnicalTools()
            chart_result = invoke_tool_with_retry(
                toolkit.generate_kline_image, 
                {"kline_data": kline_data}
            )
            
            pattern_image = chart_result.get("pattern_image", "")
            pattern_image_filename = chart_result.get("pattern_image_filename", "")
            
        except Exception as e:
            print(f"Error generating pattern chart: {str(e)}")
            pattern_image = ""
            pattern_image_filename = ""
            chart_result = {"error": str(e)}

        # --- Step 2: 生成模式分析报告 ---
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一位专门识别经典高频交易形态的交易形态识别助手。请用中文回答。"
                f"图表是基于{time_frame}间隔数据生成的。\n\n"
                "图表生成结果: {chart_result}\n\n"
                "将生成的图表与经典形态描述进行比较，确定是否存在已知形态:\n\n"
                "{pattern_descriptions}\n\n"
                "请提供详细的中文形态分析报告，包括:\n"
                "1. 识别的形态（如有）\n"
                "2. 形态可靠性和强度\n"
                "3. 交易含义\n"
                "4. 关键支撑/阻力位"
            )
        ])
        
        try:
            final_response = (analysis_prompt | llm).invoke({
                "chart_result": json.dumps(chart_result, indent=2),
                "pattern_descriptions": pattern_text
            })
            
            pattern_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
        except Exception as e:
            pattern_report = f"Error generating pattern analysis: {str(e)}\n\nChart result: {json.dumps(chart_result, indent=2)}"

        # 更新state并返回
        state.update({
            "messages": messages,
            "pattern_report": pattern_report,
            "pattern_image": pattern_image,
            "pattern_image_filename": pattern_image_filename,
        })
        
        return state

    return pattern_agent_node