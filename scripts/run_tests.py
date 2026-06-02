from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    failures: list[str] = []
    for path in sorted((root / "tests").glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                func()
                print(f"PASS {path.stem}.{name}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.stem}.{name}: {exc}")
                print(f"FAIL {path.stem}.{name}: {exc}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
