from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from bert_tasks.model import BertClassifierAndSlotParser
from bert_tasks.runtime import RuleFirstRuntime

try:
    from fastapi import FastAPI, Request
    from pydantic import BaseModel
except ImportError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bert_tasks")


class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    query_type: str
    source: str
    tasks: list[dict]


class HealthResponse(BaseModel):
    status: str
    uptime: float
    model_loaded: bool
    rules_enabled: bool


class MetricsResponse(BaseModel):
    total_requests: int
    rule_hits: int
    bert_hits: int
    unknowns: int
    avg_latency_ms: float


def create_app() -> Any:
    app = FastAPI(title="BertTasks Parser", version="0.1.0")
    
    model_parser = _build_model_parser_from_env()
    runtime = RuleFirstRuntime(model_parser=model_parser)
    
    metrics = {
        "total_requests": 0,
        "rule_hits": 0,
        "bert_hits": 0,
        "unknowns": 0,
        "total_latency_ms": 0.0,
        "start_time": time.time()
    }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        uptime = time.time() - metrics["start_time"]
        return {
            "status": "ok",
            "uptime": uptime,
            "model_loaded": model_parser is not None,
            "rules_enabled": True
        }

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics_endpoint() -> dict[str, Any]:
        total = metrics["total_requests"]
        avg_latency = metrics["total_latency_ms"] / total if total > 0 else 0.0
        return {
            "total_requests": metrics["total_requests"],
            "rule_hits": metrics["rule_hits"],
            "bert_hits": metrics["bert_hits"],
            "unknowns": metrics["unknowns"],
            "avg_latency_ms": round(avg_latency, 2)
        }

    @app.post("/parse", response_model=ParseResponse)
    async def parse_endpoint(request: Request, req: ParseRequest) -> dict[str, Any]:
        start_time = time.time()
        
        text = req.text
        logger.info(f"Received request: text='{text}'")
        
        result = runtime.parse(text)
        source = result.get("source", "unknown")
        
        latency_ms = (time.time() - start_time) * 1000
        
        metrics["total_requests"] += 1
        metrics["total_latency_ms"] += latency_ms
        
        if source == "rule_template":
            metrics["rule_hits"] += 1
        elif source == "bert_model":
            metrics["bert_hits"] += 1
        else:
            metrics["unknowns"] += 1
        
        logger.info(
            f"Request completed: text='{text}' -> source={source}, intent={_get_intent(result)}, latency={latency_ms:.2f}ms"
        )
        
        return result

    @app.post("/parse_debug")
    async def parse_debug_endpoint(request: Request, req: ParseRequest) -> dict[str, Any]:
        start_time = time.time()
        
        text = req.text
        logger.debug(f"Debug request: text='{text}'")
        
        result = runtime.parse(text)
        source = result.get("source", "unknown")
        
        latency_ms = (time.time() - start_time) * 1000
        
        debug_info = {
            "input": text,
            "result": result,
            "source": source,
            "latency_ms": round(latency_ms, 2),
            "model_available": model_parser is not None,
            "metrics": {
                "total_requests": metrics["total_requests"],
                "rule_hits": metrics["rule_hits"],
                "bert_hits": metrics["bert_hits"],
                "unknowns": metrics["unknowns"]
            }
        }
        
        return debug_info

    return app


def _get_intent(result: dict) -> str:
    if result.get("tasks"):
        return result["tasks"][0].get("intent", "unknown")
    return "unknown"


def _build_model_parser_from_env() -> Optional[BertClassifierAndSlotParser]:
    classifier_dir = os.getenv("BERT_TASKS_CLASSIFIER_DIR")
    slot_tagger_dir = os.getenv("BERT_TASKS_SLOT_TAGGER_DIR")
    
    if not classifier_dir or not slot_tagger_dir:
        logger.info("No model directories configured, running in rules-only mode")
        return None
    
    intent_threshold = float(os.getenv("BERT_TASKS_INTENT_THRESHOLD", "0.9"))
    slot_threshold = float(os.getenv("BERT_TASKS_SLOT_THRESHOLD", "0.9"))
    
    logger.info(f"Loading models: classifier={classifier_dir}, slot_tagger={slot_tagger_dir}")
    logger.info(f"Thresholds: intent={intent_threshold}, slot={slot_threshold}")
    
    return BertClassifierAndSlotParser(
        classifier_dir=classifier_dir,
        slot_tagger_dir=slot_tagger_dir,
        intent_threshold=intent_threshold,
        slot_threshold=slot_threshold,
    )


app = create_app()