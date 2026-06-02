from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from bert_tasks.data_synthesis import (
    SLOT_TASKS,
    bio_tags,
    build_prompt_batches,
    build_task_prompt,
    normalize_row,
    split_and_write,
    split_rows,
    validate_row,
)


def test_build_prompt_batches_are_llm_first() -> None:
    batches = build_prompt_batches(per_task=3, batch_size=2)
    assert batches
    assert all("只能输出 JSON Lines" in batch.prompt for batch in batches)
    assert any(batch.task_type == "slot_filling" for batch in batches)
    assert any(batch.task_type == "multi_intent" for batch in batches)


def test_slot_prompt_requires_span_extraction() -> None:
    prompt = build_task_prompt("music_play_specific", "slot_filling", SLOT_TASKS["music_play_specific"], 10)
    assert "slots 中的每个非空值都必须是 text 中连续出现的原文片段" in prompt
    assert "singer" in prompt
    assert "song" in prompt
    assert "不要输出 slot_enums、tokens、bio_tags、source" in prompt


def test_app_slot_prompt_requires_app_span_and_enum() -> None:
    prompt = build_task_prompt("music_open_app", "slot_filling", SLOT_TASKS["music_open_app"], 10)
    assert "app" in prompt
    assert "QQ音乐" in prompt
    assert "qq_music_app" in prompt
    assert "structured_output.tasks[0].params.app 写枚举值" in prompt
    assert "每行只能包含这 5 个顶层字段" in prompt


def test_normalize_and_validate_slot_row() -> None:
    row = normalize_row(
        {
            "text": "播放蔡琴的渡口",
            "task_type": "slot_filling",
            "label": "music_control:play_specific_music",
            "slots": {"singer": "蔡琴", "song": "渡口"},
            "structured_output": {
                "query_type": "single_task",
                "tasks": [
                    {
                        "user_input": "播放蔡琴的渡口",
                        "intent": "music_control",
                        "value": "music_player",
                        "params": {"singer": "蔡琴", "song": "渡口"},
                    }
                ],
            },
        }
    )
    ok, reason = validate_row(row)
    assert ok, reason
    assert len(row["tokens"]) == len(row["bio_tags"])
    assert "B-singer" in row["bio_tags"]
    assert "B-song" in row["bio_tags"]


def test_multi_intent_rows_reject_bio_slots() -> None:
    row = {
        "text": "打开投影仪",
        "task_type": "multi_intent",
        "label": "projector_control:open",
        "tokens": list("打开投影仪"),
        "bio_tags": bio_tags("打开投影仪", {"place": "投影仪"}),
        "slots": {"place": "投影仪"},
        "structured_output": {},
    }
    ok, reason = validate_row(row)
    assert not ok
    assert "multi_intent" in reason


def test_split_rows_creates_train_validation_test_sets() -> None:
    rows = [
        {
            "text": f"样本{i}",
            "task_type": "multi_intent" if i % 2 == 0 else "slot_filling",
            "label": "unknown",
            "tokens": list(f"样本{i}"),
            "bio_tags": ["O"] * len(f"样本{i}"),
            "slots": {},
            "structured_output": {"query_type": "unknown", "tasks": []},
        }
        for i in range(200)
    ]
    splits = split_rows(rows, train_ratio=0.8, validation_ratio=0.1)
    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(value) for value in splits.values()) == len(rows)
    assert all(splits[name] for name in splits)


def test_split_and_write_adds_slot_rows_to_intent_classifier() -> None:
    rows = [
        normalize_row(
            {
                "text": "打开投影仪",
                "task_type": "multi_intent",
                "label": "projector_control:open",
                "slots": {},
                "structured_output": {"query_type": "unknown", "tasks": []},
            }
        ),
        normalize_row(
            {
                "text": "打开QQ音乐",
                "task_type": "slot_filling",
                "label": "music_control:open_app",
                "slots": {"app": "QQ音乐"},
                "structured_output": {"query_type": "unknown", "tasks": []},
            }
        ),
    ]
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        split_and_write(tmp_path, rows)
        intent_text = (tmp_path / "intent_classification.jsonl").read_text(encoding="utf-8")
        slot_text = (tmp_path / "slot_filling.jsonl").read_text(encoding="utf-8")
        assert "projector_control:open" in intent_text
        assert "music_control:open_app" in intent_text
        assert "music_control:open_app" in slot_text
        assert "B-app" in slot_text
