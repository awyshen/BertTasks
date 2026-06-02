from __future__ import annotations

import os
from typing import Any

from bert_tasks.model import BertClassifierAndSlotParser
from bert_tasks.runtime import RuleFirstRuntime


def create_app() -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Service dependencies are not installed. Install with `pip install -e .[service]`.") from exc

    class ParseRequest(BaseModel):
        text: str

    app = FastAPI(title="BertTasks Parser", version="0.1.0")
    model_parser = _build_model_parser_from_env()
    runtime = RuleFirstRuntime(model_parser=model_parser)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/parse")
    def parse_endpoint(request: ParseRequest) -> dict[str, Any]:
        return runtime.parse(request.text)

    return app


def _build_model_parser_from_env() -> BertClassifierAndSlotParser | None:
    classifier_dir = os.getenv("BERT_TASKS_CLASSIFIER_DIR")
    slot_tagger_dir = os.getenv("BERT_TASKS_SLOT_TAGGER_DIR")
    if not classifier_dir or not slot_tagger_dir:
        return None
    return BertClassifierAndSlotParser(
        classifier_dir=classifier_dir,
        slot_tagger_dir=slot_tagger_dir,
        intent_threshold=float(os.getenv("BERT_TASKS_INTENT_THRESHOLD", "0.9")),
        slot_threshold=float(os.getenv("BERT_TASKS_SLOT_THRESHOLD", "0.9")),
    )


app = create_app()
