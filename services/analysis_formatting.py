import json
from typing import Any, Dict, Iterable, List


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_json_object(raw_text: Any) -> Dict[str, Any]:
    text = _safe_text(raw_text).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                payload = json.loads(text[start:end])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _format_list(title: str, items: Iterable[Any]) -> List[str]:
    values = [_safe_text(item).strip() for item in (items or []) if _safe_text(item).strip()]
    if not values:
        return []
    lines = [f"**{title}**"]
    lines.extend(f"- {item}" for item in values)
    return lines


def _format_mapping(mapping: Any) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return ""
    parts = []
    for key, value in mapping.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = _safe_text(value)
        parts.append(f"{key}: {value_text}")
    return "；".join(parts)


def format_agent_report_for_display(raw_report: Any, title: str) -> str:
    """Convert a structured agent JSON protocol into readable Markdown.

    If the report is not JSON, keep the original text so older LLM outputs still render.
    """
    text = _safe_text(raw_report).strip()
    payload = _parse_json_object(text)
    if not payload:
        return text

    lines: List[str] = [f"### {title}"]
    summary = _safe_text(payload.get("summary") or payload.get("analysis") or payload.get("conclusion")).strip()
    if summary:
        lines.append(summary)

    for label, key in [
        ("方向", "direction"),
        ("形态", "pattern"),
        ("完成度", "completion"),
        ("强度评分", "strength_score"),
        ("斜率状态", "slope_state"),
    ]:
        value = payload.get(key)
        if value not in (None, "", []):
            lines.append(f"- {label}：{_safe_text(value)}")

    target_zone = payload.get("target_zone")
    if target_zone not in (None, "", []):
        lines.append(f"- 目标区域：{_safe_text(target_zone)}")

    key_levels = payload.get("key_levels") or {
        key: payload.get(key)
        for key in ["support", "resistance"]
        if payload.get(key) not in (None, "")
    }
    level_text = _format_mapping(key_levels)
    if level_text:
        lines.append(f"- 关键位置：{level_text}")

    invalid_if = payload.get("invalid_if")
    if invalid_if:
        lines.append(f"- 失效条件：{_safe_text(invalid_if)}")

    lines.extend(_format_list("支持证据", payload.get("evidence", [])))
    lines.extend(_format_list("反方证据", payload.get("contradictions", [])))
    return "\n".join(lines)


def format_structured_signal_bundle_for_display(signal_bundle: Dict[str, Any]) -> str:
    if not isinstance(signal_bundle, dict) or not signal_bundle:
        return ""

    data_quality = signal_bundle.get("data_quality", {}) or {}
    profile = signal_bundle.get("profile", {}) or {}
    price = signal_bundle.get("price", {}) or {}
    trend = signal_bundle.get("trend", {}) or {}
    momentum = signal_bundle.get("momentum", {}) or {}
    entry = signal_bundle.get("entry", {}) or {}
    volatility = signal_bundle.get("volatility", {}) or {}
    levels = signal_bundle.get("levels", {}) or {}

    lines = ["### 结构化指标快照"]
    if profile:
        lines.append(
            f"- 策略画像：{profile.get('name', 'N/A')}；趋势窗口：{profile.get('trend_lookback', 'N/A')}；入场窗口：{profile.get('entry_lookback', 'N/A')}；预期周期：{profile.get('horizon', 'N/A')}"
        )
    lines.append(
        f"- 数据：{data_quality.get('bar_count', 'N/A')} 根K线 / {data_quality.get('status', 'unknown')}"
    )
    lines.append(
        f"- 价格：{price.get('last_close', 'N/A')}；近20根收益：{price.get('return_20_pct', 'N/A')}%"
    )
    lines.append(
        f"- 趋势：{trend.get('direction', 'N/A')} / {trend.get('trend_score', 'N/A')}；EMA8/21：{trend.get('ema_8', 'N/A')} / {trend.get('ema_21', 'N/A')}"
    )
    lines.append(
        f"- 动能：{momentum.get('momentum_score', 'N/A')}；RSI：{momentum.get('rsi_14', 'N/A')}；MACD柱：{momentum.get('macd_hist', 'N/A')}"
    )
    lines.append(
        f"- 入场：{entry.get('entry_score', 'N/A')}；距支撑：{entry.get('distance_to_support_pct', 'N/A')}%；距阻力：{entry.get('distance_to_resistance_pct', 'N/A')}%"
    )
    lines.append(
        f"- 波动：{volatility.get('volatility_score', 'N/A')}；ATR%：{volatility.get('atr_pct', 'N/A')}"
    )
    lines.append(
        f"- 支撑 / 阻力：{levels.get('nearest_support', 'N/A')} / {levels.get('nearest_resistance', 'N/A')}"
    )
    lines.extend(_format_list("结构化证据", signal_bundle.get("evidence", [])))
    lines.extend(_format_list("结构化反方", signal_bundle.get("contradictions", [])))
    return "\n".join(lines)


def format_decision_strategy_context(decision_payload: Dict[str, Any]) -> str:
    if not isinstance(decision_payload, dict) or not decision_payload:
        return ""
    profile = decision_payload.get("strategy_profile")
    weights = decision_payload.get("strategy_weights") or {}
    rule_score = decision_payload.get("rule_score")
    if not profile and not weights and rule_score in (None, ""):
        return ""
    parts = []
    if profile:
        parts.append(f"策略画像：{profile}")
    if rule_score not in (None, ""):
        parts.append(f"规则总分：{rule_score}")
    if weights:
        parts.append(
            "权重："
            + " / ".join(
                f"{label}{weights.get(key, 0):.0%}"
                for label, key in [("趋势", "trend"), ("动能", "momentum"), ("入场", "entry"), ("波动", "volatility")]
                if key in weights
            )
        )
    return "；".join(parts)


def format_serenity_lens_for_display(serenity_lens: Dict[str, Any]) -> str:
    if not isinstance(serenity_lens, dict) or not serenity_lens:
        return ""

    lines = ["### Serenity Research Lens"]
    symbol = serenity_lens.get("asset_symbol")
    if symbol:
        lines.append(f"- 标的：{symbol}")
    lines.append(f"- 研究优先级分：{serenity_lens.get('serenity_score', 'N/A')}")
    lines.append(f"- 稀缺性分：{serenity_lens.get('scarcity_score', 'N/A')}")
    lines.append(f"- 证据质量分：{serenity_lens.get('evidence_quality_score', 'N/A')}")

    field_labels = [
        ("市场故事", "market_story"),
        ("系统变化", "system_change"),
        ("卡住的环节", "value_chain_position"),
        ("市场可能没看清", "market_misread"),
        ("研究摘要", "research_summary"),
    ]
    for label, key in field_labels:
        value = _safe_text(serenity_lens.get(key)).strip()
        if value:
            lines.append(f"**{label}**")
            lines.append(value)

    lines.extend(_format_list("接下来可能重新定价的事情", serenity_lens.get("repricing_triggers", [])))
    lines.extend(_format_list("什么情况说明错了", serenity_lens.get("failure_conditions", [])))
    return "\n".join(lines)
