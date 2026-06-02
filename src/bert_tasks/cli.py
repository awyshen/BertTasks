from __future__ import annotations

import argparse
import json

from bert_tasks.parser import parse


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a Chinese robot command.")
    parser.add_argument("text", help="User input text")
    args = parser.parse_args()
    print(json.dumps(parse(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
