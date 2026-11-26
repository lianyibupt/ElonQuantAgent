from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import re


def _text(payload, field, label, limit, required=False):
    value = payload.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"{label}必须为文本。")
    value = value.strip()
    if required and not value:
        raise ValueError(f"请填写{label}。")
    if len(value) > limit:
        raise ValueError(f"{label}最多可输入 {limit} 个字符。")
    return value


def _amount(payload, field, label, allow_zero=False):
    raw = _text(payload, field, label, 40, required=True)
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"{label}必须为有效数字。") from error
    minimum = 0 if allow_zero else Decimal("0.000000000001")
    if not value.is_finite() or not minimum <= value <= 10**16 or value.as_tuple().exponent < -12:
        raise ValueError(f"请输入有效的{label}，最多 12 位小数。")
    return format(value, "f")


def parse_journal_entry(kind, payload):
    if not isinstance(payload, dict):
        raise ValueError("请输入有效的记录。")
    raw_date = _text(payload, "date", "日期", 10, required=True)
    try:
        entry_date = date.fromisoformat(raw_date).isoformat()
    except ValueError as error:
        raise ValueError("日期格式应为 YYYY-MM-DD。") from error

    if kind == "advice":
        return {
            "date": entry_date,
            "title": _text(payload, "title", "建议标题", 160, required=True),
            "content": _text(payload, "content", "建议内容", 20000, required=True),
            "source": _text(payload, "source", "对话链接或来源", 2000),
            "decision": _text(payload, "decision", "我的判断", 4000),
        }

    side = payload.get("side")
    if side not in ("buy", "sell"):
        raise ValueError("请选择买入或卖出。")
    currency = _text(payload, "currency", "币种", 8, required=True).upper()
    if not re.fullmatch(r"[A-Z0-9]{2,8}", currency):
        raise ValueError("请输入有效的币种代码，如 USD、HKD 或 USDT。")
    advice_id = payload.get("advice_id")
    if advice_id is not None and (type(advice_id) is not int or not 0 < advice_id < 2**63):
        raise ValueError("关联建议编号无效。")

    quantity = _amount(payload, "quantity", "成交数量")
    price = _amount(payload, "price", "成交单价")
    fee = _amount({"fee": payload.get("fee", "0")}, "fee", "手续费", allow_zero=True)
    with localcontext() as context:
        context.prec = 64
        gross_amount = Decimal(quantity) * Decimal(price)
        cash_flow = (gross_amount if side == "sell" else -gross_amount) - Decimal(fee)
    return {
        "date": entry_date,
        "ticker": _text(payload, "ticker", "标的", 32, required=True).upper(),
        "account": _text(payload, "account", "交易账户", 80, required=True),
        "side": side,
        "quantity": quantity,
        "price": price,
        "fee": fee,
        "gross_amount": format(gross_amount, "f"),
        "cash_flow": format(cash_flow, "f"),
        "currency": currency,
        "advice_id": advice_id,
        "notes": _text(payload, "notes", "交易备注", 4000),
    }
