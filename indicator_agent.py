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
        
        print(f"📈 [IndicatorAgent] 开始分析 {state.get('stock_name', 'Unknown')}")
        print(f"  时间框架: {time_frame}")
        
        # 检查数据格式并打印调试信息
        if isinstance(kline_data, dict):
            print(f"  数据格式: 字典，包含键: {list(kline_data.keys())}")
            if 'Datetime' in kline_data:
                data_length = len(kline_data['Datetime']) if hasattr(kline_data['Datetime'], '__len__') else 'N/A'
                print(f"  K线数据长度: {data_length}")
                
                # 打印前5个数据点用于调试
                if data_length != 'N/A' and data_length > 0:
                    print(f"  前5个时间点:")
                    for i in range(min(5, data_length)):
                        print(f"    {kline_data['Datetime'][i]}: O={kline_data.get('Open', ['N/A'])[i]}, H={kline_data.get('High', ['N/A'])[i]}, L={kline_data.get('Low', ['N/A'])[i]}, C={kline_data.get('Close', ['N/A'])[i]}")
                
                # 检查数据是否来自demo（通过Volume值判断）
                if 'Volume' in kline_data and hasattr(kline_data['Volume'], '__len__') and len(kline_data['Volume']) > 0:
                    first_volume = kline_data['Volume'][0]
                    if isinstance(first_volume, (int, float)) and first_volume >= 1000000 and first_volume <= 10000000:
                        print(f"⚠️  警告: 检测到可能使用demo数据 (Volume: {first_volume})")
                        
        else:
            print(f"  数据格式: {type(kline_data)}")
            print(f"  K线数据长度: {len(kline_data) if hasattr(kline_data, '__len__') else 'N/A'}")
            
            # 检查数据是否来自demo
            if hasattr(kline_data, 'columns') and 'Volume' in kline_data.columns:
                if len(kline_data) > 0:
                    first_volume = kline_data['Volume'].iloc[0] if hasattr(kline_data['Volume'], 'iloc') else kline_data['Volume'][0]
                    if isinstance(first_volume, (int, float)) and first_volume >= 1000000 and first_volume <= 10000000:
                        print(f"⚠️  警告: 检测到可能使用demo数据 (Volume: {first_volume})")
        
        try:
            # 调用所有技术指标工具
            print("🔧 [IndicatorAgent] 调用技术指标工具...")
            macd_result = toolkit.compute_macd.invoke({"kline_data": kline_data})
            rsi_result = toolkit.compute_rsi.invoke({"kline_data": kline_data})
            roc_result = toolkit.compute_roc.invoke({"kline_data": kline_data})
            stoch_result = toolkit.compute_stoch.invoke({"kline_data": kline_data})
            willr_result = toolkit.compute_willr.invoke({"kline_data": kline_data})
            
            tool_results = {
                "macd": macd_result,
                "rsi": rsi_result,
                "roc": roc_result,
                "stochastic": stoch_result,
                "williams_r": willr_result
            }
            
            print(f"✅ [IndicatorAgent] 技术指标计算完成")
            print(f"  MACD数据点: {len(macd_result.get('macd', []))}")
            print(f"  RSI数据点: {len(rsi_result.get('rsi', []))}")
            print(f"  ROC数据点: {len(roc_result.get('roc', []))}")
            print(f"  随机指标数据点: {len(stoch_result.get('stoch_k', []))}")
            print(f"  威廉指标数据点: {len(willr_result.get('willr', []))}")
            
        except Exception as e:
            error_msg = f"Error computing indicators: {str(e)}"
            print(f"❌ [IndicatorAgent] 技术指标计算失败: {str(e)}")
            tool_results = {"error": error_msg}

        # --- Step 2: 根据交易策略生成分析报告 ---
        trading_strategy = state.get('trading_strategy', 'high_frequency')
        
        if trading_strategy == 'low_frequency':
            # 低频交易策略提示词
            system_prompt = (
                "你是一位专业的低频交易分析助手，专注于长期趋势和价格行为分析。"
                "基于以下技术指标结果，提供一份全面的中文分析报告。"
                "总结MACD、RSI、ROC、随机指标和威廉指标的关键发现，重点关注中长期趋势信号。"
                "为低频交易决策提供可操作的中文见解，特别关注长期支撑位、阻力位和趋势变化。\n\n"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"OHLC数据来自{time_frame}间隔，反映了市场行为。\n\n"
                "技术指标数值结果（JSON格式）:\n{indicator_data}\n\n"
                "请用中文详细分析每个指标的含义和长期交易信号。"
            )
        else:
            # 高频交易策略提示词
            system_prompt = (
                "你是一位在时间敏感条件下运作的高频交易(HFT)分析助手。"
                "基于以下技术指标结果，提供一份全面的中文分析报告。"
                "总结MACD、RSI、ROC、随机指标和威廉指标的关键发现。"
                "为高频交易决策提供可操作的中文见解。\n\n"
                f"股票代码: {state.get('stock_name', 'Unknown')}\n"
                f"OHLC数据来自{time_frame}间隔，反映了最近的市场行为。\n\n"
                "技术指标数值结果（JSON格式）:\n{indicator_data}\n\n"
                "请用中文详细分析每个指标的含义和交易信号。"
            )
            
        # 创建提示词模板
        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_prompt
            )
        ])
        
        try:
            print("🤖 [IndicatorAgent] 调用LLM生成分析报告...")
            indicator_data = json.dumps(tool_results, indent=2, ensure_ascii=False)
            print(f"  传递给LLM的数据长度: {len(indicator_data)}")
            
            final_response = (analysis_prompt | llm).invoke({
                "indicator_data": indicator_data
            })
            
            indicator_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            # 确保报告使用UTF-8编码
            if isinstance(indicator_report, str):
                indicator_report = indicator_report.encode('utf-8', errors='replace').decode('utf-8')
            print(f"✅ [IndicatorAgent] LLM分析完成，报告长度: {len(indicator_report)}")
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            indicator_report = f"Error generating indicator analysis: {error_msg}\n\nRaw results:\n" + "\n".join(tool_results)
            print(f"❌ [IndicatorAgent] LLM分析失败: {error_msg}")
        
        # 更新state并返回
        state.update({
            "messages": messages,
            "indicator_report": indicator_report,
        })
        
        return state

    return indicator_agent_node