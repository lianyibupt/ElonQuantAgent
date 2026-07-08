import math
from typing import Any, Dict, List, Optional

import pandas as pd


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _series_last(series: pd.Series) -> Optional[float]:
    valid = series.dropna()
    if valid.empty:
        return None
    return _to_float(valid.iloc[-1])


def _pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return ((current / previous) - 1.0) * 100.0


def _slope_pct(series: pd.Series, lookback: int) -> Optional[float]:
    valid = series.dropna()
    if len(valid) <= lookback:
        return None
    current = _to_float(valid.iloc[-1])
    previous = _to_float(valid.iloc[-lookback - 1])
    return _pct_change(current, previous)


def _safe_mean(values: pd.Series) -> Optional[float]:
    valid = values.dropna()
    if valid.empty:
        return None
    return _to_float(valid.mean())


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.mask(avg_loss == 0)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _build_dataframe(kline_data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(kline_data).copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def _strategy_profile(trading_strategy: str, lookback: Optional[int] = None) -> Dict[str, Any]:
    if trading_strategy == "low_frequency":
        return {
            "name": "长期趋势",
            "trend_lookback": lookback or 60,
            "entry_lookback": max(lookback or 60, 60),
            "short_ema": 21,
            "medium_ema": 50,
            "long_ema": 120,
            "horizon": "1-6个月",
        }
    return {
        "name": "短期节奏",
        "trend_lookback": lookback or 20,
        "entry_lookback": lookback or 20,
        "short_ema": 8,
        "medium_ema": 21,
        "long_ema": 50,
        "horizon": "2天-1个月",
    }


def build_structured_signal_bundle(
    kline_data: Dict[str, Any],
    lookback: Optional[int] = None,
    trading_strategy: str = "high_frequency",
) -> Dict[str, Any]:
    profile = _strategy_profile(trading_strategy, lookback)
    df = _build_dataframe(kline_data)
    bar_count = len(df)
    if bar_count == 0:
        return {
            "profile": profile,
            "data_quality": {"bar_count": 0, "status": "empty", "warnings": ["没有可用OHLC数据"]},
            "price": {},
            "trend": {"direction": "unknown", "trend_score": 50},
            "momentum": {"momentum_score": 50},
            "entry": {"entry_score": 50},
            "volatility": {"volatility_score": 50},
            "levels": {},
            "evidence": [],
            "contradictions": ["数据为空，无法形成有效交易信号"],
        }

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype="float64")

    last_close = _to_float(close.iloc[-1], 0.0) or 0.0
    ema_short = _ema(close, min(profile["short_ema"], max(bar_count, 1)))
    ema_medium = _ema(close, min(profile["medium_ema"], max(bar_count, 1)))
    ema_long_span = min(profile["long_ema"], max(5, bar_count))
    ema_long = _ema(close, ema_long_span)
    ema_short_last = _series_last(ema_short)
    ema_medium_last = _series_last(ema_medium)
    ema_long_last = _series_last(ema_long)
    slope_5 = _slope_pct(close, min(5, max(bar_count - 1, 1)))
    trend_lookback = min(profile["trend_lookback"], max(bar_count - 1, 1))
    entry_lookback = min(profile["entry_lookback"], max(bar_count, 1))
    slope_profile = _slope_pct(close, trend_lookback)

    trend_score = 50.0
    evidence: List[str] = []
    contradictions: List[str] = []
    if ema_short_last is not None and ema_medium_last is not None:
        if ema_short_last > ema_medium_last:
            trend_score += 15
            evidence.append("短期均线位于中期均线上方")
        else:
            trend_score -= 15
            contradictions.append("短期均线低于中期均线")
    if ema_medium_last is not None and ema_long_last is not None:
        if ema_medium_last > ema_long_last:
            trend_score += 10
            evidence.append("中期均线位于长期均线上方")
        else:
            trend_score -= 10
            contradictions.append("中期均线低于长期均线")
    if slope_profile is not None:
        slope_multiplier = 1.1 if trading_strategy == "low_frequency" else 2.0
        trend_score += max(-20, min(20, slope_profile * slope_multiplier))
        if slope_profile > 2:
            evidence.append(f"{profile['name']}窗口近{trend_lookback}根收盘价上涨{_round(slope_profile)}%")
        elif slope_profile < -2:
            contradictions.append(f"{profile['name']}窗口近{trend_lookback}根收盘价下跌{_round(abs(slope_profile))}%")

    direction = "sideways"
    if trend_score >= 60:
        direction = "up"
    elif trend_score <= 40:
        direction = "down"

    rsi_series = _rsi(close)
    rsi_last = _series_last(rsi_series) or 50.0
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    macd_hist_last = _series_last(macd_hist) or 0.0
    macd_hist_prev = _to_float(macd_hist.dropna().iloc[-2], 0.0) if len(macd_hist.dropna()) >= 2 else 0.0

    momentum_score = 50.0
    if 45 <= rsi_last <= 70:
        momentum_score += 15
        evidence.append(f"RSI处于可持续区间({_round(rsi_last)})")
    elif rsi_last > 75:
        momentum_score -= 10
        contradictions.append(f"RSI过热({_round(rsi_last)})")
    elif rsi_last < 35:
        momentum_score -= 10
        contradictions.append(f"RSI偏弱({_round(rsi_last)})")
    if macd_hist_last > 0:
        momentum_score += 10
        evidence.append("MACD柱线为正")
    else:
        momentum_score -= 10
        contradictions.append("MACD柱线为负")
    if macd_hist_last > macd_hist_prev:
        momentum_score += 5
        evidence.append("MACD动能环比改善")

    atr_series = _atr(df)
    atr_last = _series_last(atr_series) or 0.0
    atr_pct = (atr_last / last_close) * 100 if last_close else 0.0
    volatility_score = 70.0
    if atr_pct > 8:
        volatility_score -= 30
        contradictions.append(f"ATR波动过高({_round(atr_pct)}%)")
    elif atr_pct > 4:
        volatility_score -= 15
    elif atr_pct < 1:
        volatility_score -= 5
    else:
        evidence.append(f"ATR波动处于可交易区间({_round(atr_pct)}%)")

    recent_window = df.tail(entry_lookback)
    nearest_support = _to_float(recent_window["Low"].min())
    nearest_resistance = _to_float(recent_window["High"].max())
    distance_to_support_pct = _pct_change(last_close, nearest_support)
    distance_to_resistance_pct = _pct_change(nearest_resistance, last_close)

    entry_score = 50.0
    if distance_to_support_pct is not None and 0 <= distance_to_support_pct <= 6:
        entry_score += 15
        evidence.append("价格距离近端支撑不远")
    elif distance_to_support_pct is not None and distance_to_support_pct > 12:
        entry_score -= 15
        contradictions.append("价格距离支撑较远，追高风险上升")
    if distance_to_resistance_pct is not None and distance_to_resistance_pct >= 4:
        entry_score += 10
    elif distance_to_resistance_pct is not None and distance_to_resistance_pct < 2:
        entry_score -= 10
        contradictions.append("价格距离近端阻力较近")

    volume_ratio = None
    if not volume.empty and volume.dropna().shape[0] >= 6:
        recent_volume = _safe_mean(volume.tail(5))
        prior_volume = _safe_mean(volume.tail(20).head(max(len(volume.tail(20)) - 5, 1)))
        if recent_volume is not None and prior_volume not in (None, 0):
            volume_ratio = recent_volume / prior_volume
            if volume_ratio >= 1.15:
                evidence.append(f"近5根成交量较前期放大{_round((volume_ratio - 1) * 100)}%")
                momentum_score += 5

    return {
        "data_quality": {
            "bar_count": bar_count,
            "status": "ok" if bar_count >= 30 else "thin",
            "warnings": [] if bar_count >= 30 else ["K线数量偏少，信号可靠性下降"],
        },
        "profile": profile,
        "price": {
            "last_close": _round(last_close),
            "return_5_pct": _round(slope_5),
            "return_20_pct": _round(_slope_pct(close, min(20, max(bar_count - 1, 1)))),
            "return_profile_pct": _round(slope_profile),
        },
        "trend": {
            "direction": direction,
            "trend_score": _clamp_score(trend_score),
            "ema_8": _round(ema_short_last),
            "ema_21": _round(ema_medium_last),
            "ema_long": _round(ema_long_last),
            "slope_20_pct": _round(_slope_pct(close, min(20, max(bar_count - 1, 1)))),
            "slope_profile_pct": _round(slope_profile),
        },
        "momentum": {
            "momentum_score": _clamp_score(momentum_score),
            "rsi_14": _round(rsi_last),
            "macd_hist": _round(macd_hist_last),
            "macd_hist_delta": _round(macd_hist_last - macd_hist_prev),
            "volume_ratio_5_vs_20": _round(volume_ratio),
        },
        "entry": {
            "entry_score": _clamp_score(entry_score),
            "distance_to_support_pct": _round(distance_to_support_pct),
            "distance_to_resistance_pct": _round(distance_to_resistance_pct),
        },
        "volatility": {
            "volatility_score": _clamp_score(volatility_score),
            "atr": _round(atr_last),
            "atr_pct": _round(atr_pct),
        },
        "levels": {
            "nearest_support": _round(nearest_support),
            "nearest_resistance": _round(nearest_resistance),
        },
        "evidence": evidence[:8],
        "contradictions": contradictions[:8],
    }


def build_rule_based_decision(
    signal_bundle: Dict[str, Any],
    trading_strategy: str = "high_frequency",
) -> Dict[str, Any]:
    profile = _strategy_profile(trading_strategy)
    trend_score = _clamp_score(_to_float(signal_bundle.get("trend", {}).get("trend_score"), 50) or 50)
    momentum_score = _clamp_score(_to_float(signal_bundle.get("momentum", {}).get("momentum_score"), 50) or 50)
    entry_score = _clamp_score(_to_float(signal_bundle.get("entry", {}).get("entry_score"), 50) or 50)
    volatility_score = _clamp_score(_to_float(signal_bundle.get("volatility", {}).get("volatility_score"), 50) or 50)
    contradiction_count = len(signal_bundle.get("contradictions", []) or [])

    if trading_strategy == "low_frequency":
        weights = {"trend": 0.50, "momentum": 0.15, "entry": 0.10, "volatility": 0.25}
        buy_gate = trend_score >= 70 and volatility_score >= 45
        sell_gate = trend_score <= 40 or volatility_score <= 30
    else:
        weights = {"trend": 0.25, "momentum": 0.25, "entry": 0.35, "volatility": 0.15}
        buy_gate = trend_score >= 60 and entry_score >= 55 and momentum_score >= 55
        sell_gate = trend_score <= 35 or entry_score <= 30

    rule_score = _clamp_score(
        trend_score * weights["trend"]
        + momentum_score * weights["momentum"]
        + entry_score * weights["entry"]
        + volatility_score * weights["volatility"]
        - min(15, contradiction_count * 3)
    )

    decision = "持有"
    recommended_action = "观察"
    confidence = "低"
    if rule_score >= 68 and buy_gate:
        decision = "买入"
        recommended_action = "买入"
        confidence = "中" if rule_score < 80 else "高"
    elif rule_score <= 38 or sell_gate:
        decision = "卖出"
        recommended_action = "减仓"
        confidence = "中" if rule_score > 25 else "高"

    recommended_book = "观察"
    if decision == "买入":
        recommended_book = "核心仓" if trading_strategy == "low_frequency" and trend_score >= 75 else "战术仓"

    levels = signal_bundle.get("levels", {}) or {}
    price = signal_bundle.get("price", {}) or {}
    support = _to_float(levels.get("nearest_support"))
    last_close = _to_float(price.get("last_close"))
    atr = _to_float(signal_bundle.get("volatility", {}).get("atr"), 0.0) or 0.0
    invalidation_price = support
    if invalidation_price is None and last_close is not None:
        invalidation_price = last_close - 2 * atr

    resistance = _to_float(levels.get("nearest_resistance"))
    reward = max((resistance - last_close), 0.0) if resistance is not None and last_close is not None else 0.0
    risk = max((last_close - invalidation_price), atr, 0.01) if last_close is not None and invalidation_price is not None else 1.0
    risk_reward_ratio = f"{_round(reward / risk, 1) or 0}:1"

    suggested_position_range = "0% - 0%"
    if decision == "买入":
        if rule_score >= 80:
            suggested_position_range = "4% - 6%"
        elif rule_score >= 68:
            suggested_position_range = "2% - 4%"
    elif decision == "持有":
        suggested_position_range = "0% - 2%"

    evidence = signal_bundle.get("evidence", []) or []
    contradictions = signal_bundle.get("contradictions", []) or []
    decision_path = [
        f"规则总分 {rule_score}",
        f"趋势 {trend_score}",
        f"动能 {momentum_score}",
        f"入场 {entry_score}",
        f"波动 {volatility_score}",
    ]

    return {
        "decision": decision,
        "confidence": confidence,
        "risk_reward_ratio": risk_reward_ratio,
        "forecast_horizon": profile["horizon"],
        "justification": "；".join((evidence[:3] or ["结构化信号未给出强证据"]) + ([f"反方：{contradictions[0]}"] if contradictions else [])),
        "recommended_book": recommended_book,
        "recommended_action": recommended_action,
        "trend_score": trend_score,
        "entry_score": entry_score,
        "valuation_stretch_score": entry_score,
        "catalyst_score": momentum_score,
        "volatility_score": volatility_score,
        "suggested_position_range": suggested_position_range,
        "invalidation_price": f"{invalidation_price:.2f}" if invalidation_price is not None else "待确认",
        "rule_score": rule_score,
        "strategy_profile": profile["name"],
        "strategy_weights": weights,
        "decision_path": decision_path,
        "evidence": evidence,
        "contradictions": contradictions,
    }
