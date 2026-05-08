"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def _build_indicator_summary(tool_results: dict) -> str:
    """Extract key current values from indicator results for quick reference."""
    lines = []
    try:
        if "adx" in tool_results:
            adx_vals = tool_results["adx"].get("adx", [])
            pdi = tool_results["adx"].get("plus_di", [])
            mdi = tool_results["adx"].get("minus_di", [])
            if adx_vals:
                lines.append(f"ADX={adx_vals[-1]:.1f}, +DI={pdi[-1]:.1f}, -DI={mdi[-1]:.1f}")
        if "rsi" in tool_results:
            rsi_vals = tool_results["rsi"].get("rsi", [])
            if rsi_vals:
                lines.append(f"RSI(14)={rsi_vals[-1]:.1f}")
        if "mfi" in tool_results:
            mfi_vals = tool_results["mfi"].get("mfi", [])
            if mfi_vals:
                lines.append(f"MFI(14)={mfi_vals[-1]:.1f}")
        if "atr" in tool_results:
            atr_vals = tool_results["atr"].get("atr", [])
            if atr_vals:
                lines.append(f"ATR(14)={atr_vals[-1]:.2f}")
        if "bollinger_bands" in tool_results:
            bb_upper = tool_results["bollinger_bands"].get("bb_upper", [])
            bb_mid = tool_results["bollinger_bands"].get("bb_middle", [])
            bb_lower = tool_results["bollinger_bands"].get("bb_lower", [])
            bb_bw = tool_results["bollinger_bands"].get("bb_bandwidth", [])
            if bb_upper:
                lines.append(f"BB上轨={bb_upper[-1]:.2f}, 中轨={bb_mid[-1]:.2f}, 下轨={bb_lower[-1]:.2f}, 带宽={bb_bw[-1]:.2f}%")
        if "ema" in tool_results:
            ema20 = tool_results["ema"].get("ema20", [])
            ema50 = tool_results["ema"].get("ema50", [])
            ema200 = tool_results["ema"].get("ema200", [])
            if ema20:
                lines.append(f"EMA20={ema20[-1]:.2f}, EMA50={ema50[-1]:.2f}, EMA200={ema200[-1]:.2f}")
    except Exception:
        return "统计摘要生成失败"
    return "\n".join(lines) if lines else "无可用摘要"


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
            adx_result = toolkit.compute_adx.invoke({"kline_data": kline_data})
            atr_result = toolkit.compute_atr.invoke({"kline_data": kline_data})
            mfi_result = toolkit.compute_mfi.invoke({"kline_data": kline_data})
            bb_result = toolkit.compute_bb.invoke({"kline_data": kline_data})
            obv_result = toolkit.compute_obv.invoke({"kline_data": kline_data})
            ema_result = toolkit.compute_ema.invoke({"kline_data": kline_data})

            tool_results = {
                "macd": macd_result,
                "rsi": rsi_result,
                "roc": roc_result,
                "stochastic": stoch_result,
                "williams_r": willr_result,
                "adx": adx_result,
                "atr": atr_result,
                "mfi": mfi_result,
                "bollinger_bands": bb_result,
                "obv": obv_result,
                "ema": ema_result,
            }

            print(f"✅ [IndicatorAgent] 技术指标计算完成")
            print(f"  MACD数据点: {len(macd_result.get('macd', []))}")
            print(f"  RSI数据点: {len(rsi_result.get('rsi', []))}")
            print(f"  ADX数据点: {len(adx_result.get('adx', []))}")
            print(f"  ATR数据点: {len(atr_result.get('atr', []))}")
            print(f"  MFI数据点: {len(mfi_result.get('mfi', []))}")
            print(f"  EMA数据点: {len(ema_result.get('ema20', []))}")
            
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
                
            error_msg_full = f"Error computing indicators: {error_msg}"
            print(f"❌ [IndicatorAgent] 技术指标计算失败: {error_msg}")
            tool_results = {"error": error_msg_full}

        # --- Step 2: 根据交易策略生成结构化分析报告 ---
        trading_strategy = state.get('trading_strategy', 'high_frequency')

        extra_stats = _build_indicator_summary(tool_results)

        if trading_strategy == 'low_frequency':
            horizon_note = "持有周期：1-6个月"
            focus_note = "重点关注中长期趋势延续性、EMA200方向、ADX趋势强度和BB带宽变化。"
        else:
            horizon_note = "持有周期：2天-1个月"
            focus_note = "重点关注短中期趋势、波动率环境、量价确认和入场时机。"

        system_prompt = (
            "你是一位量化技术分析专家。请基于指标数据，按以下框架逐项分析，用中文回答。\n\n"
            f"股票代码: {state.get('stock_name', 'Unknown')}\n"
            f"时间框架: {time_frame}\n"
            f"{horizon_note}\n"
            f"{focus_note}\n\n"
            "技术指标数值（JSON）:\n{indicator_data}\n\n"
            "预计算统计摘要（辅助参考）:\n{extra_stats}\n\n"
            "## 分析框架（逐项判定，不可跳过）\n\n"
            "### 1. 趋势状态\n"
            "- ADX 最新值 > 25 为趋势市，< 20 为震荡市\n"
            "- +DI 与 -DI 的交叉和差值判定方向\n"
            "- EMA20/EMA50/EMA200 排列：多头排列（EMA20>EMA50>EMA200）/ 空头排列 / 交织\n"
            "- MACD 柱状图方向 + 与零轴关系\n"
            "→ 结论：趋势上涨 / 趋势下跌 / 震荡盘整\n\n"
            "### 2. 动量状态\n"
            "- RSI 最新值区间：超卖(<30) / 正常(30-70) / 超买(>70)\n"
            "- MFI 是否与 RSI 一致？不一致时指出量价背离\n"
            "- Stoch %K/%D 是否交叉、所处区域\n"
            "- ROC 方向和极值\n"
            "→ 结论：动量偏多 / 动量偏空 / 中性\n\n"
            "### 3. 波动率环境\n"
            "- ATR 最新值占价格百分比，判定高波动(>5%) / 正常(2-5%) / 低波动(<2%)\n"
            "- BB 带宽收窄→潜在突破；带宽扩张→趋势加速\n"
            "- 当前价格在BB上轨/中轨/下轨的位置\n"
            "→ 结论：高波动 / 正常波动 / 低波动\n\n"
            "### 4. 量价关系\n"
            "- OBV 方向是否与价格方向一致？\n"
            "- OBV 出现顶背离/底背离？\n"
            "- MFI 是否确认OBV的背离信号？\n"
            "→ 结论：量价确认 / 量价背离 / 中性\n\n"
            "## 输出格式\n"
            "请仅输出以下 JSON，不要输出任何其他文字：\n"
            "{{\n"
            '  "trend_state": "趋势上涨/趋势下跌/震荡盘整",\n'
            '  "adx_value": 0.0,\n'
            '  "di_cross": "+DI在-DI上方/+DI在-DI下方/交叉中",\n'
            '  "ema_alignment": "多头排列/空头排列/交织",\n'
            '  "momentum_state": "动量偏多/动量偏空/中性",\n'
            '  "rsi_value": 0.0,\n'
            '  "mfi_rsi_consensus": "一致/RSI偏多MFI偏空/RSI偏空MFI偏多",\n'
            '  "volatility_regime": "高波动/正常波动/低波动",\n'
            '  "atr_pct": 0.0,\n'
            '  "bb_position": "上轨附近/中轨附近/下轨附近",\n'
            '  "volume_confirmation": "量价确认/量价背离/中性",\n'
            '  "bullish_signals": [],\n'
            '  "bearish_signals": [],\n'
            '  "composite_score": 0,\n'
            '  "narrative_summary": "1-3句关键发现总结"\n'
            "}}"
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
            if isinstance(indicator_data, bytes):
                indicator_data = indicator_data.decode('utf-8', errors='replace')
            elif isinstance(indicator_data, str):
                indicator_data = indicator_data.encode('utf-8', errors='replace').decode('utf-8')

            extra_stats_str = extra_stats.encode('utf-8', errors='replace').decode('utf-8')
            print(f"  传递给LLM的数据长度: {len(indicator_data)}")

            final_response = (analysis_prompt | llm).invoke({
                "indicator_data": indicator_data,
                "extra_stats": extra_stats_str,
            })
            
            indicator_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
            # 确保报告使用UTF-8编码
            if isinstance(indicator_report, str):
                indicator_report = indicator_report.encode('utf-8', errors='replace').decode('utf-8')
            elif isinstance(indicator_report, bytes):
                indicator_report = indicator_report.decode('utf-8', errors='replace')
                
            print(f"✅ [IndicatorAgent] LLM分析完成，报告长度: {len(indicator_report)}")
            
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
                
            # 安全地处理tool_results
            try:
                raw_results = "\n".join([str(r) for r in tool_results.values()]) if isinstance(tool_results, dict) else str(tool_results)
                raw_results = raw_results.encode('utf-8', errors='replace').decode('utf-8')
            except:
                raw_results = "Unable to display raw results"
                
            indicator_report = f"Error generating indicator analysis: {error_msg}\n\nRaw results:\n{raw_results}"
            print(f"❌ [IndicatorAgent] LLM分析失败: {error_msg}")
        
        # 更新state并返回
        state.update({
            "messages": messages,
            "indicator_report": indicator_report,
        })
        
        return state

    return indicator_agent_node
