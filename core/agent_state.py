from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class PositionState(TypedDict, total=False):
    ticker: Annotated[str, "Ticker symbol for the position or candidate"]
    market: Annotated[str, "Market identifier such as US or HK"]
    shares: Annotated[float, "Position share count"]
    market_value: Annotated[float, "Current market value"]
    cost_basis: Annotated[float, "Average cost basis"]
    unrealized_pnl: Annotated[float, "Unrealized profit and loss"]
    holding_days: Annotated[int, "Holding period in days"]
    book_type: Annotated[str, "Book classification such as core or tactical"]
    factor_tags: Annotated[List[str], "Factor or theme tags for concentration analysis"]
    stop_level: Annotated[float, "Stop or invalidation level"]
    thesis_status: Annotated[str, "Current thesis status"]


class AccountState(TypedDict, total=False):
    nav: Annotated[float, "Current account NAV"]
    cash: Annotated[float, "Available cash balance"]
    gross_exposure: Annotated[float, "Current gross exposure percentage"]
    core_exposure: Annotated[float, "Current core-book exposure percentage"]
    tactical_exposure: Annotated[float, "Current tactical-book exposure percentage"]
    current_drawdown: Annotated[float, "Current account drawdown percentage"]
    recent_nav_high: Annotated[float, "Recent NAV high-water mark"]
    positions: Annotated[List[PositionState], "Current holdings in the account"]


class SingleNameScore(TypedDict, total=False):
    decision: Annotated[str, "Top-level trading decision"]
    confidence: Annotated[str, "Decision confidence"]
    risk_reward_ratio: Annotated[str, "Risk reward ratio estimate"]
    forecast_horizon: Annotated[str, "Expected holding horizon"]
    justification: Annotated[str, "Narrative explanation for the decision"]
    recommended_book: Annotated[str, "Recommended book such as core or tactical"]
    recommended_action: Annotated[str, "Recommended action such as buy, hold, or trim"]
    trend_score: Annotated[int, "Trend quality score on a 0-100 scale"]
    entry_score: Annotated[int, "Entry quality score on a 0-100 scale"]
    valuation_stretch_score: Annotated[int, "Valuation or stretch score on a 0-100 scale"]
    catalyst_score: Annotated[int, "Catalyst score on a 0-100 scale"]
    volatility_score: Annotated[int, "Volatility awareness score on a 0-100 scale"]
    suggested_position_range: Annotated[str, "Suggested position range for the trade"]
    invalidation_price: Annotated[str, "Invalidation price or stop level"]


class PortfolioDirective(TypedDict, total=False):
    market_regime: Annotated[str, "Top-level market regime classification"]
    target_gross_exposure: Annotated[str, "Target total exposure range"]
    target_core_exposure: Annotated[str, "Target core-book exposure range"]
    target_tactical_exposure: Annotated[str, "Target tactical-book exposure range"]
    remaining_risk_budget: Annotated[str, "Remaining risk budget summary"]
    crowded_exposures: Annotated[List[str], "Crowded factor or theme exposures"]
    add_candidates: Annotated[List[str], "Candidates suitable for adding risk"]
    trim_candidates: Annotated[List[str], "Positions suitable for trimming"]
    blocked_candidates: Annotated[List[str], "Candidates blocked by portfolio rules"]
    manager_actions: Annotated[List[str], "Manager-level action guidance"]


class IndicatorAgentState(TypedDict, total=False):
    """State type for the current analysis pipeline."""

    kline_data: Annotated[
        dict, "OHLCV dictionary used for computing technical indicators"
    ]
    data: Annotated[dict, "Raw market data passed through the pipeline"]
    time_frame: Annotated[str, "Time period for k-line data provided"]
    stock_name: Annotated[str, "Stock name or symbol for prompting"]
    asset_symbol: Annotated[str, "Asset symbol used for orchestration"]
    trading_strategy: Annotated[str, "Trading strategy mode for prompt selection"]

    rsi: Annotated[List[float], "Relative Strength Index values"]
    macd: Annotated[List[float], "MACD line values"]
    macd_signal: Annotated[List[float], "MACD signal line values"]
    macd_hist: Annotated[List[float], "MACD histogram values"]
    stoch_k: Annotated[List[float], "Stochastic Oscillator %K values"]
    stoch_d: Annotated[List[float], "Stochastic Oscillator %D values"]
    roc: Annotated[List[float], "Rate of Change values"]
    willr: Annotated[List[float], "Williams %R values"]
    indicator_report: Annotated[
        str, "Final indicator agent summary report to be used by downstream agents"
    ]

    pattern_image: Annotated[
        str, "Base64-encoded K-line chart for pattern recognition agent use"
    ]
    pattern_image_filename: Annotated[
        str, "Local file path to saved K-line chart image"
    ]
    pattern_image_description: Annotated[
        str, "Brief description of the generated K-line image"
    ]
    pattern_report: Annotated[
        str, "Final pattern agent summary report to be used by downstream agents"
    ]

    trend_image: Annotated[
        str,
        "Base64-encoded trend-annotated candlestick chart for trend recognition agent use",
    ]
    trend_image_filename: Annotated[
        str, "Local file path to saved trendline-enhanced K-line chart image"
    ]
    trend_image_description: Annotated[
        str,
        "Brief description of the chart, including support and resistance context",
    ]
    trend_report: Annotated[
        str,
        "Final trend analysis summary for downstream agents",
    ]

    analysis_results: Annotated[str, "Computed result of the analysis or decision"]
    messages: Annotated[
        List[BaseMessage], "List of chat messages used in LLM prompt construction"
    ]
    decision_prompt: Annotated[str, "Decision prompt for reflection"]
    final_trade_decision: Annotated[
        str, "Final trading decision emitted by the decision agent"
    ]
    decision_payload: Annotated[
        Dict[str, Any], "Parsed structured decision payload"
    ]
    single_name_score: Annotated[
        SingleNameScore, "Normalized structured scorecard for the analyzed symbol"
    ]
    account_state: Annotated[
        AccountState, "Optional account context passed into the pipeline"
    ]
    positions: Annotated[
        List[PositionState], "Optional current positions passed into the pipeline"
    ]
    candidates: Annotated[
        List[Dict[str, Any]], "Optional candidate metadata passed into the pipeline"
    ]
    portfolio_directive: Annotated[
        PortfolioDirective, "Optional account-level directive for later stages"
    ]
    dashboard_payload: Annotated[
        Dict[str, Any], "Optional dashboard-ready payload for later stages"
    ]
