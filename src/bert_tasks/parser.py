from __future__ import annotations

from typing import Any

from bert_tasks.normalization import normalize_text
from bert_tasks.rules import PLACES, parse_single_rule
from bert_tasks.schema import UNKNOWN_RESULT, ParsedTask, build_result


CONNECTORS = ["然后", "接着", "随后", "之后", "再帮我", "再给我", "再", "并且", "同时", "顺便", "另外", "，", ",", "。", "；", ";"]
FOLLOW_UP_STARTERS = ("打开", "关闭", "播放", "放", "暂停", "停止", "继续", "音量", "声音", "静音", "回去", "去充电", "取消导航", "把投影")
TASK_BOUNDARY_STARTERS = (
    "打开",
    "关闭",
    "播放",
    "放",
    "暂停",
    "停止",
    "继续",
    "音量",
    "声音",
    "静音",
    "上一首",
    "下一首",
    "切歌",
    "回去充电",
    "去充电",
    "取消导航",
    "休息一下",
    "进入聊天模式",
    "打开闲聊",
    "把投影",
)
def parse(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        return dict(UNKNOWN_RESULT)

    parts = split_multi_task(normalized)
    if len(parts) > 1:
        tasks: list[ParsedTask] = []
        for part in parts:
            task = parse_single_rule(part)
            if task is None:
                return dict(UNKNOWN_RESULT)
            tasks.append(task)
        return build_result(tasks)

    single = parse_single_rule(normalized)
    if single:
        return build_result([single])
    return dict(UNKNOWN_RESULT)


def split_multi_task(text: str) -> list[str]:
    connector_parts = _split_by_connectors(text)
    if len(connector_parts) > 1:
        return connector_parts

    for place in sorted(PLACES, key=len, reverse=True):
        for nav_prefix in ["导航到", "去", "到", "来"]:
            prefix = f"{nav_prefix}{place}"
            if text.startswith(prefix) and len(text) > len(prefix):
                suffix = text[len(prefix) :]
                if suffix.startswith(FOLLOW_UP_STARTERS):
                    suffix_parts = split_multi_task(suffix)
                    if _all_parts_parse([prefix, *suffix_parts]):
                        return [prefix, *suffix_parts]

    boundary_parts = _split_by_task_boundaries(text)
    if len(boundary_parts) > 1:
        return boundary_parts

    return [text]


def _split_by_connectors(text: str) -> list[str]:
    for connector in CONNECTORS:
        if connector in text:
            pieces: list[str] = []
            for piece in [piece for piece in text.split(connector) if piece]:
                pieces.extend(split_multi_task(piece))
            if len(pieces) > 1 and _all_parts_parse(pieces):
                return pieces
    return [text]


def _split_by_task_boundaries(text: str) -> list[str]:
    boundary_indices = sorted(
        {
            index
            for starter in TASK_BOUNDARY_STARTERS
            for index in _find_starter_indices(text, starter)
            if index > 0
        }
    )
    for index in boundary_indices:
        prefix = text[:index]
        suffix = text[index:]
        suffix_parts = split_multi_task(suffix)
        parts = [prefix, *suffix_parts]
        if _all_parts_parse(parts):
            return parts
    return [text]


def _find_starter_indices(text: str, starter: str) -> list[int]:
    indices: list[int] = []
    start = 0
    while True:
        index = text.find(starter, start)
        if index < 0:
            return indices
        indices.append(index)
        start = index + len(starter)


def _all_parts_parse(parts: list[str]) -> bool:
    return len(parts) > 1 and all(parse_single_rule(part) is not None for part in parts)
