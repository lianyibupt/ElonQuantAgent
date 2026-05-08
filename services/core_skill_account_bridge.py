from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def _normalize_action(action: str) -> str:
    text = str(action or "").strip().lower()
    if any(token in text for token in ["buy", "add", "加仓", "买入"]):
        return "add"
    if any(token in text for token in ["trim", "sell", "reduce", "减仓", "卖出"]):
        return "trim"
    return "hold"


def _score_row_from_result(ticker: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    score = (analysis_result or {}).get("single_name_score", {}) or {}
    return {
        "ticker": ticker,
        "decision": str(score.get("decision", "N/A")),
        "recommended_action": str(score.get("recommended_action", "观察")),
        "trend_score": score.get("trend_score", 0),
        "entry_score": score.get("entry_score", 0),
        "volatility_score": score.get("volatility_score", 0),
        "risk_reward_ratio": str(score.get("risk_reward_ratio", "N/A")),
        "suggested_position_range": str(score.get("suggested_position_range", "N/A")),
        "justification": str(score.get("justification", "")),
    }


def build_core_skill_block(
    workspace_name: str,
    workspace: Dict[str, Any],
    fetch_market_data: Callable[..., pd.DataFrame],
    analyze_position: Callable[[str, pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    market_data_source: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    start_dt = current - timedelta(days=90)

    positions = (workspace or {}).get("positions", []) or []
    actions = {"add": [], "trim": [], "hold": []}
    score_table: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for position in positions:
        ticker = str((position or {}).get("ticker", "")).strip().upper()
        if not ticker:
            errors.append({"ticker": "UNKNOWN", "error": "empty ticker in workspace position"})
            continue

        try:
            df = fetch_market_data(ticker, "1d", start_dt, current, market_data_source=market_data_source)
            if df is None or getattr(df, "empty", True):
                errors.append({"ticker": ticker, "error": "no market data"})
                continue

            result = analyze_position(ticker, df, workspace)
            row = _score_row_from_result(ticker, result)
            score_table.append(row)

            bucket = _normalize_action(row.get("recommended_action", ""))
            actions[bucket].append(
                {
                    "ticker": ticker,
                    "action": row.get("recommended_action", "观察"),
                    "reason": row.get("justification", ""),
                }
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            continue

    if not score_table and errors:
        error_summary = "; ".join(f"{e['ticker']}: {e['error']}" for e in errors)
        raise ValueError(f"Core skill analysis failed for all positions: {error_summary}")

    return {
        "enabled": True,
        "generated_at": current.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "workspace_name": str(workspace_name or "default"),
        "summary": {
            "total_positions": len(positions),
            "processed_positions": len(score_table),
            "error_count": len(errors),
            "timeframe": "1d",
            "lookback_days": 90,
        },
        "actions": actions,
        "score_table": score_table,
        "errors": errors,
    }
