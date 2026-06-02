from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bert_tasks.schema import UNKNOWN_RESULT


@dataclass
class ModelPrediction:
    confidence: float
    result: dict[str, Any]


class BertClassifierAndSlotParser:
    """Optional BERT parser adapter for two model heads.

    The rule parser is production-ready for high-confidence templates. This
    class keeps the model service boundary stable while allowing deployments to
    plug in:
    1. a sequence-classification BERT for multi-intent fixed-control commands;
    2. a token-classification BERT for slot-filling commands.
    """

    def __init__(
        self,
        classifier_dir: str,
        slot_tagger_dir: str,
        intent_threshold: float = 0.9,
        slot_threshold: float = 0.9,
    ) -> None:
        self.classifier_dir = classifier_dir
        self.slot_tagger_dir = slot_tagger_dir
        self.intent_threshold = intent_threshold
        self.slot_threshold = slot_threshold
        self._loaded = False

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Model dependencies are not installed. Install with `pip install -e .[model]`."
            ) from exc
        self._loaded = True

    def predict(self, text: str) -> ModelPrediction:
        if not self._loaded:
            self.load()
        return ModelPrediction(confidence=0.0, result=dict(UNKNOWN_RESULT))


BertModelParser = BertClassifierAndSlotParser
