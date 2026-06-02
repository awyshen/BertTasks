from __future__ import annotations

import argparse
import json
from pathlib import Path

from bert_tasks.data_synthesis import strip_source, validate_row
from bert_tasks.parser import parse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact structured match on JSONL data.")
    parser.add_argument("--file", default="data/generated/all.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    exact = 0
    valid = 0
    for row in rows:
        ok, _reason = validate_row(row)
        valid += int(ok)
        if strip_source(parse(row["text"])) == strip_source(row["structured_output"]):
            exact += 1
    score = exact / total if total else 0.0
    print(json.dumps({"total": total, "valid_schema": valid / total if total else 0.0, "rule_exact_match": score}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
