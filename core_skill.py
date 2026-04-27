import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from core.trading_graph import TradingGraph, safe_str


def fetch_yfinance_data(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        prepost=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    column_mapping = {
        "Date": "Datetime",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    }
    existing = {old: new for old, new in column_mapping.items() if old in df.columns}
    if existing:
        df = df.rename(columns=existing)
    required = ["Datetime", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    df = df[required + (["Volume"] if "Volume" in df.columns else [])]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


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
            df_slice_dict["Datetime"] = df_slice["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        except Exception:
            df_slice_dict["Datetime"] = [safe_str(dt) for dt in df_slice["Datetime"].tolist()]
    for col in required_price_columns:
        try:
            df_slice_dict[col] = [float(x) if pd.notna(x) else 0.0 for x in df_slice[col].tolist()]
        except Exception:
            df_slice_dict[col] = [safe_str(x) for x in df_slice[col].tolist()]
    return df_slice_dict


def render_markdown(symbol: str, interval: str, start: datetime, end: datetime, final_state: dict) -> str:
    indicator_report = final_state.get("indicator_report", "")
    pattern_report = final_state.get("pattern_report", "")
    trend_report = final_state.get("trend_report", "")
    decision = final_state.get("final_trade_decision", "")
    score = final_state.get("single_name_score", {}) or {}
    score_lines = [
        f"- 决策: {score.get('decision', 'N/A')}",
        f"- 推荐账本: {score.get('recommended_book', 'N/A')}",
        f"- 推荐动作: {score.get('recommended_action', 'N/A')}",
        f"- 置信度: {score.get('confidence', 'N/A')}",
        f"- 风险收益比: {score.get('risk_reward_ratio', 'N/A')}",
        f"- 预测周期: {score.get('forecast_horizon', 'N/A')}",
        f"- 趋势分: {score.get('trend_score', 'N/A')}",
        f"- 入场分: {score.get('entry_score', 'N/A')}",
        f"- 拉伸分: {score.get('valuation_stretch_score', 'N/A')}",
        f"- 催化分: {score.get('catalyst_score', 'N/A')}",
        f"- 波动分: {score.get('volatility_score', 'N/A')}",
        f"- 建议仓位: {score.get('suggested_position_range', 'N/A')}",
        f"- 失效价: {score.get('invalidation_price', 'N/A')}",
        f"- 理由: {score.get('justification', 'N/A')}",
    ]
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
            "## 结构化评分卡",
            *score_lines,
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
            ""
        ]
    )


def run_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    output_path: str,
    trading_strategy: str,
    generate_charts: bool
) -> str:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    df = fetch_yfinance_data(symbol, interval, start, end)
    if df.empty:
        raise ValueError("yfinance 未返回有效数据")
    kline_data = build_kline_dict(df)
    graph = TradingGraph()
    if generate_charts:
        result = graph.analyze(kline_data, symbol, interval, trading_strategy)
    else:
        result = graph.analyze_text_only(kline_data, symbol, interval, trading_strategy)
    final_state = result.get("final_state", {})
    markdown = render_markdown(symbol, interval, start, end, final_state)
    output_file = Path(output_path)
    output_file.write_text(markdown, encoding="utf-8")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output", default="analysis_output.md")
    parser.add_argument("--strategy", default="high_frequency")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args()
    output_file = run_pipeline(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        output_path=args.output,
        trading_strategy=args.strategy,
        generate_charts=args.charts
    )
    print(json.dumps({"success": True, "output": output_file}, ensure_ascii=False))


if __name__ == "__main__":
    main()
