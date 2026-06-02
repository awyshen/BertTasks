from __future__ import annotations

from typing import Any

from bert_tasks.model import BertClassifierAndSlotParser
from bert_tasks.parser import parse as parse_by_rules
from bert_tasks.schema import SOURCE_BERT_MODEL, UNKNOWN_RESULT


class RuleFirstRuntime:
    """Rule template first, BERT service second.

    The edge runtime keeps deterministic rules as the first pass. If rules
    return unknown and model directories are configured, BERT intent
    classification and slot extraction can try a conservative fallback.
    """

    def __init__(self, model_parser: BertClassifierAndSlotParser | None = None) -> None:
        self.model_parser = model_parser

    def parse(self, text: str) -> dict[str, Any]:
        rule_result = parse_by_rules(text)
        if rule_result["query_type"] != "unknown":
            return rule_result
        if self.model_parser is None:
            return dict(UNKNOWN_RESULT)
        prediction = self.model_parser.predict(text)
        if prediction.confidence < self.model_parser.intent_threshold:
            return dict(UNKNOWN_RESULT)
        result = dict(prediction.result)
        result["source"] = SOURCE_BERT_MODEL
        return result
