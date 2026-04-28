from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone


BOOK_TYPE_CORE = "core"
BOOK_TYPE_TACTICAL = "tactical"
PRICE_SOURCE_ANALYSIS_REFRESH = "analysis_refresh"
TIMESTAMP_FIELDS = ("last_price_update_at", "last_analyzed_at")
CANDIDATE_TIMESTAMP_FIELDS = ("last_screened_at",)

POSITION_MANUAL_DEFAULTS = {
    "ticker": "",
    "book_type": "",
    "cost_basis": 0.0,
    "shares": 0.0,
    "tracking_status": "",
    "factor_tags": [],
    "notes": "",
}

POSITION_DERIVED_DEFAULTS = {
    "latest_price": 0.0,
    "market_value": 0.0,
    "position_weight": 0.0,
    "unrealized_pnl": 0.0,
    "unrealized_pnl_pct": 0.0,
    "price_source": "",
    "last_price_update_at": None,
    "last_analyzed_at": None,
    "last_analysis_summary": "",
}

ACCOUNT_MANUAL_DEFAULTS = {
    "nav": 0.0,
    "cash": 0.0,
    "current_drawdown": 0.0,
}

ACCOUNT_DERIVED_DEFAULTS = {
    "total_market_value": 0.0,
    "cash_pct": 0.0,
    "gross_exposure": 0.0,
    "core_exposure": 0.0,
    "tactical_exposure": 0.0,
    "largest_position": "",
    "largest_position_weight": 0.0,
    "position_count": 0,
    "last_aggregated_at": None,
}

ACCOUNT_STATE_NUMERIC_FIELDS = (
    "total_market_value",
    "cash_pct",
    "gross_exposure",
    "core_exposure",
    "tactical_exposure",
    "largest_position_weight",
)
ACCOUNT_STATE_STRING_FIELDS = ("largest_position",)
ACCOUNT_STATE_INTEGER_FIELDS = ("position_count",)
ACCOUNT_STATE_TIMESTAMP_FIELDS = ("last_aggregated_at",)


def build_default_workspace():
    return {
        "account_state": {**ACCOUNT_MANUAL_DEFAULTS, **ACCOUNT_DERIVED_DEFAULTS},
        "positions": [],
        "candidates": [],
        "notes": "",
    }


def normalize_workspace_payload(payload):
    workspace = build_default_workspace()
    payload = payload or {}

    account_payload = payload.get("account_state") or {}
    workspace["account_state"].update(
        {
            "nav": _to_float(account_payload.get("nav", 0.0)),
            "cash": _to_float(account_payload.get("cash", 0.0)),
            "current_drawdown": _to_float(account_payload.get("current_drawdown", 0.0)),
        }
    )
    for field in ACCOUNT_STATE_NUMERIC_FIELDS:
        workspace["account_state"][field] = _to_float(account_payload.get(field, workspace["account_state"][field]))
    for field in ACCOUNT_STATE_STRING_FIELDS:
        workspace["account_state"][field] = str(account_payload.get(field, workspace["account_state"][field]) or "")
    for field in ACCOUNT_STATE_INTEGER_FIELDS:
        workspace["account_state"][field] = _to_int(account_payload.get(field, workspace["account_state"][field]))
    for field in ACCOUNT_STATE_TIMESTAMP_FIELDS:
        workspace["account_state"][field] = _normalize_timestamp(account_payload.get(field, workspace["account_state"][field]))

    raw_positions = payload.get("positions")
    positions = []
    for raw_position in raw_positions if isinstance(raw_positions, list) else []:
        raw_position = raw_position if isinstance(raw_position, dict) else {}
        normalized = {**POSITION_MANUAL_DEFAULTS, **POSITION_DERIVED_DEFAULTS}
        normalized["ticker"] = str(raw_position.get("ticker", "")).strip().upper()
        normalized["book_type"] = str(raw_position.get("book_type", "")).strip().lower()
        normalized["cost_basis"] = _to_float(raw_position.get("cost_basis", 0.0))
        normalized["shares"] = _to_float(raw_position.get("shares", 0.0))
        normalized["tracking_status"] = str(raw_position.get("tracking_status", "")).strip()
        normalized["factor_tags"] = _normalize_factor_tags(raw_position.get("factor_tags", []))
        normalized["notes"] = str(raw_position.get("notes", "") or "")

        for key, default in POSITION_DERIVED_DEFAULTS.items():
            value = raw_position.get(key, default)
            normalized[key] = default if value is None and default is None else value

        normalized["latest_price"] = _to_float(normalized.get("latest_price", 0.0))
        normalized["market_value"] = _to_float(normalized.get("market_value", 0.0))
        normalized["position_weight"] = _to_float(normalized.get("position_weight", 0.0))
        normalized["unrealized_pnl"] = _to_float(normalized.get("unrealized_pnl", 0.0))
        normalized["unrealized_pnl_pct"] = _to_float(normalized.get("unrealized_pnl_pct", 0.0))
        normalized["price_source"] = str(normalized.get("price_source", "") or "")
        normalized["last_analysis_summary"] = str(normalized.get("last_analysis_summary", "") or "")
        for field in TIMESTAMP_FIELDS:
            normalized[field] = _normalize_timestamp(normalized.get(field))
        positions.append(normalized)

    raw_candidates = payload.get("candidates")
    candidates = []
    for raw_candidate in raw_candidates if isinstance(raw_candidates, list) else []:
        raw_candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
        normalized_candidate = {
            "ticker": str(raw_candidate.get("ticker", "")).strip().upper(),
            "factor_tags": _normalize_factor_tags(raw_candidate.get("factor_tags", [])),
            "notes": str(raw_candidate.get("notes", "") or ""),
            "last_screened_at": None,
        }
        for field in CANDIDATE_TIMESTAMP_FIELDS:
            normalized_candidate[field] = _normalize_timestamp(raw_candidate.get(field, normalized_candidate[field]))
        candidates.append(normalized_candidate)

    workspace["positions"] = positions
    workspace["candidates"] = candidates
    workspace["notes"] = str(payload.get("notes", "") or "")
    return workspace


def recalculate_account_state(workspace):
    recalculated = normalize_workspace_payload(deepcopy(workspace))
    account_state = recalculated["account_state"]
    nav = _to_float(account_state.get("nav", 0.0))

    total_market_value = 0.0
    core_market_value = 0.0
    tactical_market_value = 0.0
    largest_position = ""
    largest_market_value = 0.0

    for position in recalculated["positions"]:
        latest_price = _to_float(position.get("latest_price", 0.0))
        shares = _to_float(position.get("shares", 0.0))
        cost_basis = _to_float(position.get("cost_basis", 0.0))

        market_value = latest_price * shares
        unrealized_pnl = (latest_price - cost_basis) * shares
        cost_total = cost_basis * shares
        unrealized_pnl_pct = (unrealized_pnl / cost_total * 100.0) if cost_total else 0.0
        position_weight = (market_value / nav * 100.0) if nav else 0.0

        position["market_value"] = round(market_value, 4)
        position["unrealized_pnl"] = round(unrealized_pnl, 4)
        position["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 4)
        position["position_weight"] = round(position_weight, 4)

        total_market_value += market_value
        if position["book_type"] == BOOK_TYPE_CORE:
            core_market_value += market_value
        if position["book_type"] == BOOK_TYPE_TACTICAL:
            tactical_market_value += market_value
        if market_value > largest_market_value:
            largest_market_value = market_value
            largest_position = position["ticker"]

    cash = _to_float(account_state.get("cash", 0.0))
    account_state["total_market_value"] = round(total_market_value, 4)
    account_state["cash_pct"] = round((cash / nav * 100.0) if nav else 0.0, 4)
    account_state["gross_exposure"] = round((total_market_value / nav * 100.0) if nav else 0.0, 4)
    account_state["core_exposure"] = round((core_market_value / nav * 100.0) if nav else 0.0, 4)
    account_state["tactical_exposure"] = round((tactical_market_value / nav * 100.0) if nav else 0.0, 4)
    account_state["largest_position"] = largest_position
    account_state["largest_position_weight"] = round((largest_market_value / nav * 100.0) if nav else 0.0, 4)
    account_state["position_count"] = len(recalculated["positions"])
    account_state["last_aggregated_at"] = _utc_now_iso()
    return recalculated


def refresh_position_after_analysis(workspace, asset, latest_price, analysis_summary, updated_at=None):
    refreshed = normalize_workspace_payload(deepcopy(workspace))
    updated_ticker = str(asset or "").strip().upper()
    timestamp = _normalize_timestamp(updated_at) or _utc_now_iso()

    updated_index = None
    for index, position in enumerate(refreshed["positions"]):
        if position["ticker"] != updated_ticker:
            continue
        position["latest_price"] = _to_float(latest_price)
        position["price_source"] = PRICE_SOURCE_ANALYSIS_REFRESH
        position["last_price_update_at"] = timestamp
        position["last_analyzed_at"] = timestamp
        position["last_analysis_summary"] = str(analysis_summary or "")
        updated_index = index
        break

    if updated_index is not None:
        refreshed = recalculate_account_state(refreshed)
    metadata = {
        "updated_ticker": updated_ticker,
        "updated_at": timestamp,
        "position_index": updated_index,
    }
    return refreshed, metadata


def _normalize_factor_tags(value):
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return []


def _normalize_timestamp(value):
    if value is None:
        return None
    return str(value)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
