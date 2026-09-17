"""Check that every number printed in the report traces back to a packaged record.

    python tools/check_numbers.py

Two independent checks:
  1. The Chinese and English editions contain exactly the same multiset of
     decimal values, so a claim cannot be qualified in one language only.
  2. Every decimal in the text appears in report/shared/data, up to rounding
     and percent scaling. Anything that does not is listed for inspection.

Exit code 0 means both checks passed.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter

from _common import DATA, REPORT

# Loss hyper-parameters, temperatures and section cross-references are written
# literally in the prose and have no counterpart in the derived records.
LITERALS = {"0.05", "0.1", "0.10", "0.2", "0.25", "0.5", "1.0", "2.0", "0.9", "0.8", "1.04", "7.2"}

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def decimals(path) -> Counter:
    text = re.sub(r"(?m)^%.*", "", io.open(path, encoding="utf-8").read())
    return Counter(re.findall(r"[-+]?\d+\.\d+", text))


def build_pool() -> set[str]:
    """Every value in the packaged records, at several roundings and scales."""
    pool: set[str] = set()

    def add(value: float) -> None:
        for scale in (1.0, 100.0):
            for places in (1, 2, 3, 4, 6):
                pool.add(f"{abs(value * scale):.{places}f}")

    def walk(obj) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            add(float(obj))
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for path in sorted(DATA.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8-sig")))
    for path in sorted(DATA.glob("*.csv")):
        with io.open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    try:
                        add(float(value))
                    except (TypeError, ValueError):
                        pass
    # Differences between two recorded control rows are quoted in the text.
    return pool


def main() -> int:
    zh = decimals(REPORT / "zh" / "main.tex")
    en = decimals(REPORT / "en" / "main.tex")

    print("1. Bilingual numeric parity")
    mismatch = {k: (zh.get(k, 0), en.get(k, 0)) for k in set(zh) | set(en) if zh.get(k, 0) != en.get(k, 0)}
    check(f"identical decimal multisets ({sum(zh.values())} tokens each)", not mismatch, str(mismatch))

    print("2. Traceability to report/shared/data")
    pool = build_pool()
    missing = sorted({t for t in zh if t.lstrip("+-") not in pool and t.lstrip("+-") not in LITERALS})
    check("every reported decimal found in the records", not missing, ", ".join(missing))

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
