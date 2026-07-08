from typing import Any, Dict, List, Optional


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _normalize_tags(candidate_context: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(candidate_context, dict):
        return []
    tags = candidate_context.get("factor_tags") or candidate_context.get("tags") or []
    if isinstance(tags, str):
        return [tags] if tags.strip() else []
    if isinstance(tags, list):
        return [_safe_text(tag) for tag in tags if _safe_text(tag)]
    return []


def _infer_value_chain_position(asset_symbol: str, tags: List[str]) -> str:
    symbol = asset_symbol.upper()
    joined_tags = " ".join(tags).lower()
    if symbol == "TSLA" or "ev" in joined_tags or "energy" in joined_tags:
        return "电动车、储能、能源基础设施与自动驾驶应用层"
    if symbol == "HIMS" or "health" in joined_tags or "telehealth" in joined_tags:
        return "数字医疗获客、处方履约与消费医疗服务层"
    if symbol == "CRCL" or "stablecoin" in joined_tags or "crypto" in joined_tags:
        return "稳定币发行、支付网络与链上金融基础设施层"
    if symbol == "SBET" or "treasury" in joined_tags or "ethereum" in joined_tags:
        return "加密资产财务储备与资本市场载体层"
    if "ai" in joined_tags:
        return "AI应用、数据或基础设施相关环节"
    return "美股个股商业链条中的待验证环节"


def _infer_market_story(asset_symbol: str, tags: List[str]) -> str:
    readable_tags = "、".join(tags[:4]) if tags else "公司自身增长逻辑"
    return f"市场可能正在按 {readable_tags} 给 {asset_symbol.upper()} 定价。"


def _score_serenity(tags: List[str], signal_bundle: Dict[str, Any]) -> Dict[str, int]:
    trend_score = signal_bundle.get("trend", {}).get("trend_score", 50) or 50
    evidence_count = len(signal_bundle.get("evidence", []) or [])
    contradiction_count = len(signal_bundle.get("contradictions", []) or [])
    thematic_bonus = min(len(tags) * 4, 16)
    scarcity_score = _clamp_score(48 + thematic_bonus + evidence_count * 4 - contradiction_count * 5)
    evidence_quality_score = _clamp_score(45 + min(evidence_count * 7, 28) + (5 if trend_score >= 65 else 0) - contradiction_count * 4)
    serenity_score = _clamp_score(scarcity_score * 0.45 + evidence_quality_score * 0.35 + trend_score * 0.20)
    return {
        "scarcity_score": scarcity_score,
        "evidence_quality_score": evidence_quality_score,
        "serenity_score": serenity_score,
    }


def build_serenity_research_lens(
    asset_symbol: str,
    structured_signal_bundle: Optional[Dict[str, Any]] = None,
    candidate_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    signal_bundle = structured_signal_bundle or {}
    tags = _normalize_tags(candidate_context)
    symbol = _safe_text(asset_symbol).upper() or "UNKNOWN"
    scores = _score_serenity(tags, signal_bundle)
    value_chain_position = _infer_value_chain_position(symbol, tags)
    market_story = _infer_market_story(symbol, tags)
    evidence = signal_bundle.get("evidence", []) or []
    contradictions = signal_bundle.get("contradictions", []) or []

    return {
        "scope": "us_equity",
        "asset_symbol": symbol,
        "market_story": market_story,
        "system_change": "首版基于用户候选标签与技术结构推断真实变化；需要后续用财报、电话会和公告验证。",
        "value_chain_position": value_chain_position,
        "scarcity_score": scores["scarcity_score"],
        "evidence_quality_score": scores["evidence_quality_score"],
        "market_misread": "市场可能只按短期价格强弱定价，而没有区分公司是否处在难以复制、能持续兑现的链条位置。",
        "repricing_triggers": [
            "收入或用户增长证明需求不是一次性波动",
            "毛利率、现金流或订单数据验证商业质量",
            "管理层指引或客户/生态合作强化稀缺位置",
        ],
        "failure_conditions": [
            "核心增长指标放缓或低于市场叙事",
            "竞争者绕开该公司的链条位置",
            "价格跌破技术失效位且基本面证据没有增强",
        ] + [_safe_text(item) for item in contradictions[:2]],
        "serenity_score": scores["serenity_score"],
        "research_summary": "；".join(
            [
                f"{symbol} 的研究重点是确认其是否真的处在：{value_chain_position}",
                f"当前 Serenity 研究优先级分为 {scores['serenity_score']}",
                f"已有技术证据：{'、'.join(evidence[:2]) if evidence else '暂无强证据'}",
            ]
        ),
    }


def blend_serenity_into_decision(
    base_decision: Dict[str, Any],
    serenity_lens: Dict[str, Any],
    trading_strategy: str,
) -> Dict[str, Any]:
    if not isinstance(base_decision, dict):
        base_decision = {}
    if not isinstance(serenity_lens, dict) or not serenity_lens:
        return dict(base_decision)

    result = dict(base_decision)
    base_score = base_decision.get("rule_score", 50) or 50
    serenity_score = serenity_lens.get("serenity_score", 50) or 50
    serenity_weight = 0.55 if trading_strategy == "low_frequency" else 0.25
    technical_weight = 1 - serenity_weight
    blended_score = _clamp_score(base_score * technical_weight + serenity_score * serenity_weight)

    result["rule_score"] = blended_score
    result["serenity_score"] = _clamp_score(serenity_score)
    result["serenity_weight"] = serenity_weight
    result["scarcity_score"] = _clamp_score(serenity_lens.get("scarcity_score", 50) or 50)
    result["evidence_quality_score"] = _clamp_score(serenity_lens.get("evidence_quality_score", 50) or 50)
    result["serenity_summary"] = serenity_lens.get("research_summary", "")

    if trading_strategy == "low_frequency" and blended_score >= 68 and result.get("decision") == "持有":
        result["recommended_action"] = "观察"
        result["recommended_book"] = "观察"
    if trading_strategy == "high_frequency" and serenity_score < 35 and result.get("decision") == "买入":
        result["suggested_position_range"] = "0% - 2%"

    previous_reason = _safe_text(result.get("justification"))
    serenity_reason = f"Serenity研究层：{serenity_lens.get('research_summary', '')}"
    result["justification"] = "\n\n".join(part for part in [previous_reason, serenity_reason] if part)
    return result
