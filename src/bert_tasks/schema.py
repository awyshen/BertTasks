from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_RULE_TEMPLATE = "rule_template"
SOURCE_BERT_MODEL = "bert_model"
SOURCE_UNKNOWN = "unknown"

UNKNOWN_RESULT = {"query_type": "unknown", "source": SOURCE_UNKNOWN, "tasks": []}


@dataclass(frozen=True)
class ParsedTask:
    user_input: str
    intent: str
    value: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "intent": self.intent,
            "value": self.value,
            "params": self.params,
        }


def build_result(tasks: list[ParsedTask], source: str = SOURCE_RULE_TEMPLATE) -> dict[str, Any]:
    if not tasks:
        return dict(UNKNOWN_RESULT)
    return {
        "query_type": "single_task" if len(tasks) == 1 else "multi_task",
        "source": source,
        "tasks": [task.to_dict() for task in tasks],
    }
