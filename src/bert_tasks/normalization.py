from __future__ import annotations

import re


CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def chinese_number_to_int(value: str) -> int | None:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in CN_DIGITS:
        return CN_DIGITS[value]
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = CN_DIGITS.get(left, 1 if left == "" else None)
        ones = CN_DIGITS.get(right, 0 if right == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if value == "百":
        return 100
    if value.endswith("百") and value[:-1] in CN_DIGITS:
        return CN_DIGITS[value[:-1]] * 100
    return None


def normalize_volume(raw: str) -> str | None:
    raw = raw.strip().rstrip("%")
    if raw in {"最大", "最高", "满", "满格", "最大声"}:
        return "100"
    if raw in {"最小", "最低"}:
        return "0"
    number = chinese_number_to_int(raw)
    if number is None or not 0 <= number <= 100:
        return None
    return str(number)
