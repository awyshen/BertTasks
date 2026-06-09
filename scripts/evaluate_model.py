from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer


def evaluate_intent_classifier(model_dir: str, test_file: str) -> None:
    print(f"Evaluating intent classifier: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    # Load label map
    label_map_path = Path(model_dir) / "label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_label = label_map["id_to_label"]
    
    # Load test data
    test_rows = [json.loads(line) for line in Path(test_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    
    y_true = []
    y_pred = []
    
    for row in test_rows:
        text = row["text"]
        true_label = row["label"]
        
        encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            pred_label = id_to_label[str(pred_idx)]
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    print(classification_report(y_true, y_pred))


def evaluate_slot_tagger(model_dir: str, test_file: str) -> None:
    print(f"\nEvaluating slot tagger: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    
    # Load label map
    label_map_path = Path(model_dir) / "label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_label = label_map["id_to_label"]
    
    # Load test data
    test_rows = [json.loads(line) for line in Path(test_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    
    y_true = []
    y_pred = []
    
    for row in test_rows:
        tokens = row["tokens"]
        true_tags = row["bio_tags"]
        
        encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        word_ids = encoding.word_ids(batch_index=0)
        
        with torch.no_grad():
            outputs = model(**encoding)
            pred_tags = torch.argmax(outputs.logits, dim=-1)[0]
        
        # Align predictions with true tags
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(true_tags):
                y_true.append(true_tags[word_id])
                y_pred.append(id_to_label[str(pred_tags[idx].item())])
    
    print(classification_report(y_true, y_pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BERT models on test data.")
    parser.add_argument("--classifier-dir", default="models/bert_tasks/intent_classifier")
    parser.add_argument("--slot-tagger-dir", default="models/bert_tasks/slot_tagger")
    parser.add_argument("--intent-test-file", default="data/generated/splits/test/intent_classification.jsonl")
    parser.add_argument("--slot-test-file", default="data/generated/splits/test/slot_filling.jsonl")
    args = parser.parse_args()
    
    evaluate_intent_classifier(args.classifier_dir, args.intent_test_file)
    evaluate_slot_tagger(args.slot_tagger_dir, args.slot_test_file)


if __name__ == "__main__":
    main()