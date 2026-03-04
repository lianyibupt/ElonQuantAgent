import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import akshare as ak
import pandas as pd

import talib
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def safe_str(obj: Any) -> str:
    try:
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, str):
            return obj.encode("utf-8", errors="replace").decode("utf-8")
        return str(obj).encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "Error converting to string"


def create_llm(model: str, temperature: float) -> ChatOpenAI:
    llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
    if llm_provider == "deepseek":
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
            deepseek_model = "deepseek-chat" if "gpt" in model.lower() else model
            return ChatOpenAI(
                model=deepseek_model,
                temperature=temperature,
                api_key=deepseek_key,
                base_url="https://api.deepseek.com/v1",
            )
    if llm_provider == "volcengine":
        volcengine_key = os.environ.get("VOLCENGINE_API_KEY")
        if volcengine_key and volcengine_key != "your-volcengine-api-key-here":
            volcengine_model = "ep-20250519162223-96wj4" if "gpt" in model.lower() else model
            return ChatOpenAI(
                model=volcengine_model,
                temperature=temperature,
                api_key=volcengine_key,
                base_url="https://ark-cn-beijing.bytedance.net/api/v3",
            )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and openai_key != "your-openai-api-key-here":
        return ChatOpenAI(model=model, temperature=temperature, api_key=openai_key)
    return ChatOpenAI(model=model, temperature=temperature)


def _normalize_akshare_period(interval: str) -> str:
    v = safe_str(interval).strip().lower()
    if v in {"1d", "d", "day", "daily"}:
        return "daily"
    if v in {"1w", "1wk", "wk", "week", "weekly"}:
        return "weekly"
    if v in {"1mo", "1m", "mo", "month", "monthly"}:
        return "monthly"
    return "daily"


def fetch_akshare_data(symbol: str, interval: str, start: datetime, end: datetime, retries: int = 2) -> pd.DataFrame:
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    period = _normalize_akshare_period(interval)

    functions_to_try = []
    if re.match(r"^[A-Z]{1,5}$", symbol):
        functions_to_try.extend(
            [
                ("stock_us_daily", symbol),
                ("stock_us_spot", symbol),
            ]
        )
    elif re.match(r"^\d{6}$", symbol):
        functions_to_try.extend(
            [
                ("stock_zh_a_hist", symbol),
                ("stock_zh_index_daily_em", symbol),
            ]
        )
    elif symbol.startswith("SH") or symbol.startswith("SZ"):
        functions_to_try.extend(
            [
                ("index_zh_a_hist", symbol),
                ("stock_zh_index_daily_em", symbol),
            ]
        )
    else:
        functions_to_try.extend(
            [
                ("stock_zh_index_daily_em", symbol),
                ("stock_zh_a_hist", symbol),
                ("index_zh_a_hist", symbol),
                ("stock_us_daily", symbol),
                ("stock_us_spot", symbol),
            ]
        )

    last_error = None
    df = pd.DataFrame()
    for func_name, sym in functions_to_try:
        for attempt in range(retries):
            try:
                func = getattr(ak, func_name)
                if func_name == "stock_zh_a_hist":
                    df = func(
                        symbol=sym,
                        period=period,
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="",
                    )
                elif func_name == "index_zh_a_hist":
                    df = func(
                        symbol=sym,
                        period=period,
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                    )
                elif func_name == "stock_us_daily":
                    df = func(symbol=sym)
                elif func_name == "stock_us_spot":
                    df = func()
                    if not df.empty and "symbol" in df.columns:
                        df = df[df["symbol"] == sym]
                else:
                    df = func(symbol=sym)
                if df is None:
                    df = pd.DataFrame()
                if not df.empty:
                    break
            except Exception as e:
                last_error = safe_str(e)
                if attempt < retries - 1:
                    time.sleep(1.2 * (attempt + 1))
                df = pd.DataFrame()
        if not df.empty:
            break

    if df.empty:
        return pd.DataFrame()

    column_mapping = {
        "date": "Datetime",
        "日期": "Datetime",
        "Date": "Datetime",
        "datetime": "Datetime",
        "open": "Open",
        "开盘": "Open",
        "Open": "Open",
        "high": "High",
        "最高": "High",
        "High": "High",
        "low": "Low",
        "最低": "Low",
        "Low": "Low",
        "close": "Close",
        "收盘": "Close",
        "Close": "Close",
        "volume": "Volume",
        "成交量": "Volume",
        "Volume": "Volume",
        "成交额": "Volume",
        "amount": "Volume",
    }
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    df = df.loc[:, ~df.columns.duplicated()]

    if "Datetime" not in df.columns:
        if df.index.name in {"date", "日期", "Date", "datetime"}:
            df = df.reset_index().rename(columns={df.columns[0]: "Datetime"})
        else:
            date_col = next(
                (c for c in df.columns if safe_str(c).strip().lower() in {"date", "datetime", "日期"}),
                None,
            )
            if date_col:
                df = df.rename(columns={date_col: "Datetime"})

    required = ["Datetime", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["Datetime"] >= pd.to_datetime(start_date)) & (df["Datetime"] <= pd.to_datetime(end_date))]

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[required + ["Volume"]].sort_values("Datetime")
    return df.reset_index(drop=True)


def generate_demo_data(start: datetime, end: datetime) -> pd.DataFrame:
    dt_index = pd.date_range(start=start, end=end, freq="D")
    if len(dt_index) < 60:
        dt_index = pd.date_range(end=end, periods=60, freq="D")
    base = pd.Series(range(len(dt_index)), index=dt_index).astype(float)
    close = 100 + base * 0.3 + (base % 7) * 0.15
    open_ = close.shift(1).fillna(close.iloc[0]) + ((base % 5) - 2) * 0.05
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.25
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.25
    vol = (1_000_000 + (base % 13) * 50_000).astype(float)
    return pd.DataFrame(
        {
            "Datetime": dt_index,
            "Open": open_.values,
            "High": high.values,
            "Low": low.values,
            "Close": close.values,
            "Volume": vol.values,
        }
    )


def build_kline_dict(df: pd.DataFrame, max_bars: int = 100) -> dict:
    df_slice = df.tail(max_bars)
    required_price_columns = ["Open", "High", "Low", "Close"]
    has_datetime_column = "Datetime" in df_slice.columns
    has_datetime_index = df_slice.index.name == "Datetime" or isinstance(df_slice.index, pd.DatetimeIndex)
    if not all(col in df_slice.columns for col in required_price_columns) or (not has_datetime_column and not has_datetime_index):
        raise ValueError(f"Missing required columns. Available columns: {list(df_slice.columns)}, Index: {df_slice.index.name}")
    if has_datetime_index:
        df_slice = df_slice.reset_index()
        has_datetime_column = True
    df_slice_dict = {}
    if has_datetime_column:
        try:
            dt_col = df_slice["Datetime"]
            if isinstance(dt_col, pd.DataFrame):
                dt_col = dt_col.iloc[:, 0]
            df_slice_dict["Datetime"] = dt_col.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        except Exception:
            dt_col = df_slice["Datetime"]
            if isinstance(dt_col, pd.DataFrame):
                dt_col = dt_col.iloc[:, 0]
            df_slice_dict["Datetime"] = [safe_str(dt) for dt in dt_col.tolist()]
    for col in required_price_columns:
        try:
            col_data = df_slice[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            df_slice_dict[col] = [float(x) if pd.notna(x) else 0.0 for x in col_data.tolist()]
        except Exception:
            col_data = df_slice[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            df_slice_dict[col] = [safe_str(x) for x in col_data.tolist()]
    if "Volume" in df_slice.columns:
        try:
            vol_col = df_slice["Volume"]
            if isinstance(vol_col, pd.DataFrame):
                vol_col = vol_col.iloc[:, 0]
            df_slice_dict["Volume"] = [float(x) if pd.notna(x) else 0.0 for x in vol_col.tolist()]
        except Exception:
            vol_col = df_slice["Volume"]
            if isinstance(vol_col, pd.DataFrame):
                vol_col = vol_col.iloc[:, 0]
            df_slice_dict["Volume"] = [safe_str(x) for x in vol_col.tolist()]
    return df_slice_dict


def compute_indicators(kline_data: dict) -> dict:
    df = pd.DataFrame(kline_data)
    macd, macd_signal, macd_hist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(df["Close"], timeperiod=14)
    roc = talib.ROC(df["Close"], timeperiod=10)
    stoch_k, stoch_d = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
    willr = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)
    return {
        "macd": {
            "macd": macd.fillna(0).round(2).tolist(),
            "macd_signal": macd_signal.fillna(0).round(2).tolist(),
            "macd_hist": macd_hist.fillna(0).round(2).tolist(),
        },
        "rsi": {"rsi": rsi.fillna(0).round(2).tolist()},
        "roc": {"roc": roc.fillna(0).round(2).tolist()},
        "stochastic": {"stoch_k": stoch_k.fillna(0).round(2).tolist(), "stoch_d": stoch_d.fillna(0).round(2).tolist()},
        "williams_r": {"willr": willr.fillna(0).round(2).tolist()},
    }


PATTERN_TEXT = (
    "请参考以下经典K线形态：\n\n"
    "1. 倒头肩形态：三个低点，中间最低，结构对称，通常预示即将上涨。\n"
    "2. 双底形态：两个相似的低点，中间有反弹，形成'W'形。\n"
    "3. 圆弧底：价格逐渐下跌后逐渐上升，形成'U'形。\n"
    "4. 潜伏底：水平整理后突然向上突破。\n"
    "5. 下降楔形：价格向下收窄，通常向上突破。\n"
    "6. 上升楔形：价格缓慢上升但收敛，经常向下突破。\n"
    "7. 上升三角形：上升支撑线配合水平阻力线，突破通常向上。\n"
    "8. 下降三角形：下降阻力线配合水平支撑线，通常向下突破。\n"
    "9. 看涨旗形：急涨后短暂向下整理，然后继续上涨。\n"
    "10. 看跌旗形：急跌后短暂向上整理，然后继续下跌。\n"
    "11. 矩形：价格在水平支撑和阻力之间波动。\n"
    "12. 岛形反转：两个相反方向的价格缺口形成孤立的价格岛。\n"
    "13. V形反转：急跌后急涨，或相反。\n"
    "14. 圆顶/圆底：逐渐见顶或见底，形成弧形形态。\n"
    "15. 扩散三角形：高点和低点越来越宽，表示波动加剧。\n"
    "16. 对称三角形：高点和低点向顶点收敛，通常伴随突破。\n"
)


def indicator_agent(state: dict, llm: Optional[Any], offline: bool) -> dict:
    time_frame = state["time_frame"]
    kline_data = state["kline_data"]
    tool_results = compute_indicators(kline_data)
    trading_strategy = state.get("trading_strategy", "high_frequency")
    if trading_strategy == "low_frequency":
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
    indicator_data = json.dumps(tool_results, indent=2, ensure_ascii=False)
    if offline:
        state["indicator_report"] = indicator_data
        state["indicator_raw"] = tool_results
        return state
    analysis_prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    final_response = (analysis_prompt | llm).invoke({"indicator_data": indicator_data})
    indicator_report = final_response.content if hasattr(final_response, "content") else str(final_response)
    state["indicator_report"] = safe_str(indicator_report)
    state["indicator_raw"] = tool_results
    return state


def pattern_agent_text_only(state: dict, llm: Optional[Any], offline: bool) -> dict:
    time_frame = state["time_frame"]
    kline_data = state["kline_data"]
    price_data = {
        "open_prices": kline_data.get("Open", []),
        "high_prices": kline_data.get("High", []),
        "low_prices": kline_data.get("Low", []),
        "close_prices": kline_data.get("Close", []),
        "datetimes": kline_data.get("Datetime", []),
    }
    recent_closes = price_data["close_prices"][-10:] if len(price_data["close_prices"]) > 10 else price_data["close_prices"]
    price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100) if recent_closes else 0
    trading_strategy = state.get("trading_strategy", "high_frequency")
    if trading_strategy == "low_frequency":
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
    open_prices_str = safe_str(str(price_data["open_prices"][-20:]))
    high_prices_str = safe_str(str(price_data["high_prices"][-20:]))
    low_prices_str = safe_str(str(price_data["low_prices"][-20:]))
    close_prices_str = safe_str(str(price_data["close_prices"][-20:]))
    datetimes_str = safe_str(str(price_data["datetimes"][-20:]))
    if offline:
        state["pattern_report"] = safe_str(
            json.dumps(
                {
                    "open_prices": price_data["open_prices"][-20:],
                    "high_prices": price_data["high_prices"][-20:],
                    "low_prices": price_data["low_prices"][-20:],
                    "close_prices": price_data["close_prices"][-20:],
                    "datetimes": price_data["datetimes"][-20:],
                    "price_change_pct": round(price_change, 4),
                    "pattern_descriptions": PATTERN_TEXT,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return state
    analysis_prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    final_response = (analysis_prompt | llm).invoke(
        {
            "open_prices": open_prices_str,
            "high_prices": high_prices_str,
            "low_prices": low_prices_str,
            "close_prices": close_prices_str,
            "datetimes": datetimes_str,
            "price_change": price_change,
            "pattern_descriptions": PATTERN_TEXT,
        }
    )
    pattern_report = final_response.content if hasattr(final_response, "content") else str(final_response)
    state["pattern_report"] = safe_str(pattern_report)
    return state


def trend_agent_text_only(state: dict, llm: Optional[Any], offline: bool) -> dict:
    time_frame = state["time_frame"]
    kline_data = state["kline_data"]
    price_data = {
        "open_prices": kline_data.get("Open", []),
        "high_prices": kline_data.get("High", []),
        "low_prices": kline_data.get("Low", []),
        "close_prices": kline_data.get("Close", []),
        "datetimes": kline_data.get("Datetime", []),
    }
    recent_closes = price_data["close_prices"][-20:] if len(price_data["close_prices"]) > 20 else price_data["close_prices"]
    recent_highs = price_data["high_prices"][-20:] if len(price_data["high_prices"]) > 20 else price_data["high_prices"]
    recent_lows = price_data["low_prices"][-20:] if len(price_data["low_prices"]) > 20 else price_data["low_prices"]
    sma_short = sum(recent_closes[-5:]) / 5 if len(recent_closes) >= 5 else None
    sma_long = sum(recent_closes[-20:]) / 20 if len(recent_closes) >= 20 else None
    price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100) if recent_closes and recent_closes[0] != 0 else 0
    support_level = min(recent_lows) if recent_lows else None
    resistance_level = max(recent_highs) if recent_highs else None
    trading_strategy = state.get("trading_strategy", "high_frequency")
    if trading_strategy == "low_frequency":
        system_prompt = (
            "你是低频交易的趋势分析专家，专注于长期趋势和价格行为分析。请用中文回答。"
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
            "1. 长期整体趋势方向（看涨、看跌或横盘）\n"
            "2. 长期关键支撑和阻力位分析\n"
            "3. 长期趋势强度和动量评估\n"
            "4. 长期潜在突破或跌破点\n"
            "5. 基于长期趋势分析的交易建议\n"
            "6. 对未来1-6个月价格走势的预测\n\n"
            "专注于为低频交易决策提供可操作的中文见解，重点关注长期趋势。"
        )
    else:
        system_prompt = (
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
    open_prices_str = safe_str(str(price_data["open_prices"][-20:]))
    high_prices_str = safe_str(str(price_data["high_prices"][-20:]))
    low_prices_str = safe_str(str(price_data["low_prices"][-20:]))
    close_prices_str = safe_str(str(price_data["close_prices"][-20:]))
    datetimes_str = safe_str(str(price_data["datetimes"][-20:]))
    if offline:
        state["trend_report"] = safe_str(
            json.dumps(
                {
                    "open_prices": price_data["open_prices"][-20:],
                    "high_prices": price_data["high_prices"][-20:],
                    "low_prices": price_data["low_prices"][-20:],
                    "close_prices": price_data["close_prices"][-20:],
                    "datetimes": price_data["datetimes"][-20:],
                    "price_change_pct": round(price_change, 4),
                    "sma_short": sma_short,
                    "sma_long": sma_long,
                    "support_level": support_level,
                    "resistance_level": resistance_level,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return state
    analysis_prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    final_response = (analysis_prompt | llm).invoke(
        {
            "open_prices": open_prices_str,
            "high_prices": high_prices_str,
            "low_prices": low_prices_str,
            "close_prices": close_prices_str,
            "datetimes": datetimes_str,
            "price_change": price_change,
            "sma_short": sma_short if sma_short is not None else "N/A",
            "sma_long": sma_long if sma_long is not None else "N/A",
            "support_level": support_level if support_level is not None else "N/A",
            "resistance_level": resistance_level if resistance_level is not None else "N/A",
        }
    )
    trend_report = final_response.content if hasattr(final_response, "content") else str(final_response)
    state["trend_report"] = safe_str(trend_report)
    return state


def decision_agent(state: dict, llm: Optional[Any], offline: bool) -> dict:
    time_frame = state["time_frame"]
    stock_name = state["stock_name"]
    indicator_report = state.get("indicator_report", "No indicator analysis available")
    pattern_report = state.get("pattern_report", "No pattern analysis available")
    trend_report = state.get("trend_report", "No trend analysis available")
    trading_strategy = state.get("trading_strategy", "high_frequency")
    if trading_strategy == "low_frequency":
        system_prompt = (
            "你是一位资深的低频交易决策专家，持有周期以月为单位，最长可达半年。"
            "基于以下综合分析报告，做出最终的交易决策。请用中文回答。\n\n"
            "股票代码: {stock_name}\n"
            "时间周期: {time_frame}\n\n"
            "技术指标分析:\n{indicator_report}\n\n"
            "形态分析:\n{pattern_report}\n\n"
            "趋势分析:\n{trend_report}\n\n"
            "请按照以下JSON格式提供你的最终决策（用中文填写）:\n"
            "{\n"
            '  "decision": "买入/卖出/持有",\n'
            '  "confidence": "高/中/低",\n'
            '  "risk_reward_ratio": "X:Y",\n'
            '  "forecast_horizon": "1-6个月",\n'
            '  "justification": "详细的中文理由说明，重点关注长期趋势和基本面因素"\n'
            "}\n\n"
            "请综合考虑所有三种分析类型，为低频交易提供可操作的中文见解，重点关注长期趋势和基本面因素。"
        )
    else:
        system_prompt = (
            "你是一位资深的高频交易决策专家，最短持有2天，最长持有1个月。"
            "基于以下综合分析报告，做出最终的交易决策。请用中文回答。\n\n"
            "股票代码: {stock_name}\n"
            "时间周期: {time_frame}\n\n"
            "技术指标分析:\n{indicator_report}\n\n"
            "形态分析:\n{pattern_report}\n\n"
            "趋势分析:\n{trend_report}\n\n"
            "请按照以下JSON格式提供你的最终决策（用中文填写）:\n"
            "{\n"
            '  "decision": "买入/卖出/持有",\n'
            '  "confidence": "高/中/低",\n'
            '  "risk_reward_ratio": "X:Y",\n'
            '  "forecast_horizon": "预测时间段",\n'
            '  "justification": "详细的中文理由说明"\n'
            "}\n\n"
            "请综合考虑所有三种分析类型，为高频交易提供可操作的中文见解。"
        )
    if offline:
        state["final_trade_decision"] = json.dumps(
            {
                "decision": "持有",
                "confidence": "低",
                "risk_reward_ratio": "1:1",
                "forecast_horizon": "Unknown",
                "justification": "离线模式未调用LLM",
            },
            ensure_ascii=False,
            indent=2,
        )
        return state
    decision_prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    final_response = (decision_prompt | llm).invoke(
        {
            "stock_name": stock_name,
            "time_frame": time_frame,
            "indicator_report": indicator_report,
            "pattern_report": pattern_report,
            "trend_report": trend_report,
        }
    )
    decision_content = final_response.content if hasattr(final_response, "content") else str(final_response)
    state["final_trade_decision"] = safe_str(decision_content)
    return state


def render_markdown(symbol: str, interval: str, start: datetime, end: datetime, final_state: dict) -> str:
    indicator_report = final_state.get("indicator_report", "")
    pattern_report = final_state.get("pattern_report", "")
    trend_report = final_state.get("trend_report", "")
    decision = final_state.get("final_trade_decision", "")
    return "\n".join(
        [
            f"# 交易分析报告 - {symbol}",
            "",
            "## 元信息",
            f"- 标的: {symbol}",
            f"- 周期: {interval}",
            f"- 起始时间: {start.strftime('%Y-%m-%d')}",
            f"- 结束时间: {end.strftime('%Y-%m-%d')}",
            "",
            "## 指标分析",
            indicator_report or "(空)",
            "",
            "## 形态分析",
            pattern_report or "(空)",
            "",
            "## 趋势分析",
            trend_report or "(空)",
            "",
            "## 最终决策",
            "```json",
            decision or "{}",
            "```",
            "",
        ]
    )


@dataclass
class SkillConfig:
    agent_llm_model: str = "gpt-4o-mini"
    graph_llm_model: str = "gpt-4o"
    agent_llm_temperature: float = 0.1
    graph_llm_temperature: float = 0.1


def run(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    output_path: str = "analysis_output.md",
    trading_strategy: str = "high_frequency",
    offline: bool = False,
    demo: bool = False,
    config: Optional[SkillConfig] = None,
) -> str:
    cfg = config or SkillConfig()
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    df = fetch_akshare_data(symbol, interval, start, end)
    if df.empty and demo:
        df = generate_demo_data(start, end)
    if df.empty:
        raise ValueError("akshare 未返回有效数据")
    kline_data = build_kline_dict(df)
    state: Dict[str, Any] = {
        "kline_data": kline_data,
        "stock_name": symbol,
        "time_frame": interval,
        "trading_strategy": trading_strategy,
        "indicator_report": "",
        "pattern_report": "",
        "trend_report": "",
        "final_trade_decision": "",
    }
    agent_llm = None
    graph_llm = None
    if not offline:
        agent_llm = create_llm(cfg.agent_llm_model, cfg.agent_llm_temperature)
        graph_llm = create_llm(cfg.graph_llm_model, cfg.graph_llm_temperature)
    state = indicator_agent(state, agent_llm, offline)
    state = pattern_agent_text_only(state, agent_llm, offline)
    state = trend_agent_text_only(state, agent_llm, offline)
    state = decision_agent(state, graph_llm, offline)
    markdown = render_markdown(symbol, interval, start, end, state)
    output_file = Path(output_path)
    output_file.write_text(markdown, encoding="utf-8")
    return str(output_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output", default="analysis_output.md")
    parser.add_argument("--strategy", default="high_frequency")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    output_file = run(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        output_path=args.output,
        trading_strategy=args.strategy,
        offline=args.offline,
        demo=args.demo,
    )
    print(json.dumps({"success": True, "output": output_file}, ensure_ascii=False))


if __name__ == "__main__":
    main()
