from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from bert_tasks.schema import SOURCE_BERT_MODEL, UNKNOWN_RESULT


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
        
        self.classifier_tokenizer = None
        self.classifier_model = None
        self.classifier_label_map = None
        
        self.slot_tokenizer = None
        self.slot_model = None
        self.slot_label_map = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
        except ImportError as exc:
            raise RuntimeError(
                "Model dependencies are not installed. Install with `pip install -e .[model]`."
            ) from exc
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load intent classifier
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_dir)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_dir).to(device)
        self.classifier_model.eval()
        
        label_map_path = Path(self.classifier_dir) / "label_map.json"
        if label_map_path.exists():
            self.classifier_label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        
        # Load slot tagger
        self.slot_tokenizer = AutoTokenizer.from_pretrained(self.slot_tagger_dir)
        self.slot_model = AutoModelForTokenClassification.from_pretrained(self.slot_tagger_dir).to(device)
        self.slot_model.eval()
        
        slot_label_map_path = Path(self.slot_tagger_dir) / "label_map.json"
        if slot_label_map_path.exists():
            self.slot_label_map = json.loads(slot_label_map_path.read_text(encoding="utf-8"))
        
        self._loaded = True
        print(f"  Models loaded on {device}")

    def predict(self, text: str) -> ModelPrediction:
        if not self._loaded:
            self.load()
        
        # Intent classification
        encoding = self.classifier_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier_model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
        
        if confidence < self.intent_threshold:
            return ModelPrediction(confidence=confidence, result=dict(UNKNOWN_RESULT))
        
        # Get predicted label
        if self.classifier_label_map and "id_to_label" in self.classifier_label_map:
            label = self.classifier_label_map["id_to_label"].get(str(pred_idx), str(pred_idx))
        else:
            label = str(pred_idx)
        
        # Parse label
        parts = label.split(":")
        if len(parts) >= 2:
            intent = parts[0]
            sub_intent = parts[1]
        else:
            intent = label
            sub_intent = label
        
        # Check if slot is required
        slot_required_intents = {
            "volume_control:set_volume",
            "music_control:play_specific_music",
            "music_control:open_app",
            "music_control:close_app",
            "app_control:open_video_app",
            "app_control:close_video_app",
            "app_control:play_video_content",
            "robot_control:navigate_to_place",
        }
        
        slots = {}
        if label in slot_required_intents:
            # Slot extraction
            slot_encoding = self.slot_tokenizer(
                list(text),
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                slot_outputs = self.slot_model(**slot_encoding)
                slot_logits = slot_outputs.logits
                slot_probs = torch.softmax(slot_logits, dim=-1)
                slot_preds = torch.argmax(slot_probs, dim=-1)[0]
            
            # Decode slots
            id_to_label = self.slot_label_map.get("id_to_label", {}) if self.slot_label_map else {}
            word_ids = slot_encoding.word_ids(batch_index=0)
            
            current_slot = None
            current_value = []
            
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id >= len(text):
                    continue
                
                tag_idx = slot_preds[idx].item()
                tag = id_to_label.get(str(tag_idx), "O")
                
                if tag.startswith("B-"):
                    if current_slot:
                        slots[current_slot] = "".join(current_value)
                    current_slot = tag[2:]
                    current_value = [text[word_id]]
                elif tag.startswith("I-") and current_slot == tag[2:]:
                    current_value.append(text[word_id])
                else:
                    if current_slot:
                        slots[current_slot] = "".join(current_value)
                        current_slot = None
                        current_value = []
            
            if current_slot:
                slots[current_slot] = "".join(current_value)
        
        # Build params
        params = dict(slots)
        
        # Add control param for certain intents
        if intent in ["projector_control", "app_control", "music_control"]:
            if sub_intent in ["open", "close", "pause", "stop", "play", "next", "previous", "resume"]:
                params["control"] = sub_intent
            elif sub_intent == "set_volume":
                params["control"] = "set"
            elif sub_intent.startswith("play_"):
                params["control"] = "play"
        
        # Build result directly
        result = {
            "query_type": "single_task",
            "source": SOURCE_BERT_MODEL,
            "tasks": [{
                "user_input": text,
                "intent": intent,
                "value": self._get_value_for_intent(intent, label),
                "params": params
            }]
        }
        
        return ModelPrediction(confidence=confidence, result=result)
    
    def _get_value_for_intent(self, intent: str, label: str) -> str:
        intent_value_map = {
            "volume_control": "speaker",
            "music_control": "music_player",
            "app_control": "video_player",
            "robot_control": "nav",
            "projector_control": "projector",
            "assistant_control": "assistant",
        }
        return intent_value_map.get(intent, label)


BertModelParser = BertClassifierAndSlotParser