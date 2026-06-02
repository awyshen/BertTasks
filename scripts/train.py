from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Chinese BERT multi-task parser.")
    parser.add_argument("--intent-file", default="data/generated/splits/train/intent_classification.jsonl")
    parser.add_argument("--slot-file", default="data/generated/splits/train/slot_filling.jsonl")
    parser.add_argument("--validation-intent-file", default="data/generated/splits/validation/intent_classification.jsonl")
    parser.add_argument("--validation-slot-file", default="data/generated/splits/validation/slot_filling.jsonl")
    parser.add_argument("--test-intent-file", default="data/generated/splits/test/intent_classification.jsonl")
    parser.add_argument("--test-slot-file", default="data/generated/splits/test/slot_filling.jsonl")
    parser.add_argument("--multi-intent-file", default=None, help="Deprecated alias for --intent-file.")
    parser.add_argument("--validation-multi-intent-file", default=None, help="Deprecated alias for --validation-intent-file.")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--output-dir", default="models/bert_tasks")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training")
    parser.add_argument("--smoke", action="store_true", help="Validate data loading without starting full training.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set after training.")
    args = parser.parse_args()

    intent_file = args.multi_intent_file or args.intent_file
    validation_intent_file = args.validation_multi_intent_file or args.validation_intent_file
    intent_rows = _read_jsonl(Path(intent_file))
    slot_rows = _read_jsonl(Path(args.slot_file))
    validation_intent_rows = _read_jsonl(Path(validation_intent_file), required=False)
    validation_slot_rows = _read_jsonl(Path(args.validation_slot_file), required=False)
    intent_labels = Counter(row["label"] for row in intent_rows)
    slot_labels = Counter(tag for row in slot_rows for tag in row["bio_tags"])
    slot_required_labels = sorted({row["label"] for row in slot_rows})
    print(f"Loaded {len(intent_rows)} intent-classification rows from {intent_file}")
    print(f"Loaded {len(slot_rows)} slot-filling rows from {args.slot_file}")
    print(f"Loaded {len(validation_intent_rows)} validation intent-classification rows")
    print(f"Loaded {len(validation_slot_rows)} validation slot-filling rows")
    print(f"Intent labels: {dict(intent_labels)}")
    print(f"Slot-required intent labels: {slot_required_labels}")
    print(f"BIO labels: {dict(slot_labels)}")

    if args.smoke:
        print("Smoke train passed: classifier and slot-tagger data are loadable.")
        return

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("Install model dependencies with `pip install -e .[model]` before full training.") from exc

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    _train_intent_classifier(
        torch=torch,
        trainer_cls=Trainer,
        training_args_cls=TrainingArguments,
        model_cls=AutoModelForSequenceClassification,
        tokenizer=tokenizer,
        model_name=args.model_name,
        train_rows=intent_rows,
        validation_rows=validation_intent_rows,
        test_rows=_read_jsonl(Path(args.test_intent_file), required=False),
        output_dir=output_dir / "intent_classifier",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        evaluate=args.evaluate,
    )
    _train_slot_tagger(
        torch=torch,
        trainer_cls=Trainer,
        training_args_cls=TrainingArguments,
        model_cls=AutoModelForTokenClassification,
        tokenizer=tokenizer,
        model_name=args.model_name,
        train_rows=slot_rows,
        validation_rows=validation_slot_rows,
        test_rows=_read_jsonl(Path(args.test_slot_file), required=False),
        output_dir=output_dir / "slot_tagger",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        evaluate=args.evaluate,
    )
    print(f"Saved BERT models to {output_dir}")


def _read_jsonl(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing training file: {path}. Run scripts/generate_data.py with OPENAI_API_KEY first.")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _train_intent_classifier(
    *,
    torch,
    trainer_cls,
    training_args_cls,
    model_cls,
    tokenizer,
    model_name: str,
    train_rows: list[dict],
    validation_rows: list[dict],
    test_rows: list[dict],
    output_dir: Path,
    epochs: float,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    early_stopping_patience: int,
    evaluate: bool,
) -> None:
    labels = sorted({row["label"] for row in train_rows + validation_rows + (test_rows or [])})
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = model_cls.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    train_dataset = SequenceDataset(torch, tokenizer, train_rows, label_to_id, max_length)
    eval_dataset = SequenceDataset(torch, tokenizer, validation_rows, label_to_id, max_length) if validation_rows else None
    test_dataset = SequenceDataset(torch, tokenizer, test_rows, label_to_id, max_length) if test_rows else None
    _run_trainer(
        trainer_cls=trainer_cls,
        training_args_cls=training_args_cls,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        early_stopping_patience=early_stopping_patience,
        evaluate=evaluate,
    )
    (output_dir / "label_map.json").write_text(json.dumps({"label_to_id": label_to_id, "id_to_label": id_to_label}, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_slot_tagger(
    *,
    torch,
    trainer_cls,
    training_args_cls,
    model_cls,
    tokenizer,
    model_name: str,
    train_rows: list[dict],
    validation_rows: list[dict],
    test_rows: list[dict],
    output_dir: Path,
    epochs: float,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    early_stopping_patience: int,
    evaluate: bool,
) -> None:
    all_rows = train_rows + validation_rows + (test_rows or [])
    labels = sorted({tag for row in all_rows for tag in row["bio_tags"]})
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = model_cls.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    train_dataset = TokenDataset(torch, tokenizer, train_rows, label_to_id, max_length)
    eval_dataset = TokenDataset(torch, tokenizer, validation_rows, label_to_id, max_length) if validation_rows else None
    test_dataset = TokenDataset(torch, tokenizer, test_rows, label_to_id, max_length) if test_rows else None
    _run_trainer(
        trainer_cls=trainer_cls,
        training_args_cls=training_args_cls,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        early_stopping_patience=early_stopping_patience,
        evaluate=evaluate,
    )
    (output_dir / "label_map.json").write_text(json.dumps({"label_to_id": label_to_id, "id_to_label": id_to_label}, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_trainer(
    *,
    trainer_cls,
    training_args_cls,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    test_dataset,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    early_stopping_patience: int,
    evaluate: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = training_args_cls(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=eval_dataset is not None,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if evaluate and test_dataset:
        print(f"Evaluating on test set...")
        results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test results: {results}")


class SequenceDataset:
    def __init__(self, torch, tokenizer, rows: list[dict], label_to_id: dict[str, int], max_length: int) -> None:
        self.torch = torch
        self.rows = rows
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = [label_to_id[row["label"]] for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        item = {key: self.torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = self.torch.tensor(self.labels[index])
        return item


class TokenDataset:
    def __init__(self, torch, tokenizer, rows: list[dict], label_to_id: dict[str, int], max_length: int) -> None:
        self.torch = torch
        self.rows = rows
        all_tokens = [row["tokens"] for row in rows]
        self.encodings = tokenizer(
            all_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = []
        for i, row in enumerate(rows):
            word_ids = self.encodings.word_ids(batch_index=i)
            aligned = []
            for word_id in word_ids:
                if word_id is None or word_id >= len(row["bio_tags"]):
                    aligned.append(-100)
                else:
                    aligned.append(label_to_id[row["bio_tags"][word_id]])
            self.labels.append(aligned)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        item = {key: self.torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = self.torch.tensor(self.labels[index])
        return item


if __name__ == "__main__":
    main()
