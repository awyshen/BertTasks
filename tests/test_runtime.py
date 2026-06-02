from __future__ import annotations

from bert_tasks.runtime import RuleFirstRuntime


class FakeModelParser:
    intent_threshold = 0.9

    def predict(self, text: str):
        return type(
            "Prediction",
            (),
            {
                "confidence": 0.95,
                "result": {
                    "query_type": "single_task",
                    "tasks": [
                        {
                            "user_input": text,
                            "intent": "assistant_control",
                            "value": "assistant",
                            "params": {"control": "chat"},
                        }
                    ],
                },
            },
        )()


def test_runtime_marks_bert_fallback_source() -> None:
    result = RuleFirstRuntime(model_parser=FakeModelParser()).parse("模型兜底样例")
    assert result["source"] == "bert_model"


def test_runtime_keeps_rule_source_when_rule_matches() -> None:
    result = RuleFirstRuntime(model_parser=FakeModelParser()).parse("打开投影仪")
    assert result["source"] == "rule_template"
