"""Check that the committed figures, tables and PDFs regenerate from the packaged records.

    python tools/verify_reproduction.py

Redraws the six figures and regenerates the seven numeric tables into a scratch
directory and compares them byte for byte against the committed copies. If a
build/ directory from tools/build_report.py is present, also compares the
recompiled PDFs against the shipped ones page by page.

Exit code 0 means every check passed.
"""
from __future__ import annotations

import shutil
import sys

from _common import (BUILD, DATA, FIGURE_NAMES, FIGURES, LANGS, REPORT, ROOT, SHARED,
                     TABLE_NAMES, TABLES, generate_tables, python, run, sha)

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    scratch = BUILD / "verify"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    print("1. Figures redrawn from report/shared/data")
    redraw = scratch / "figures"
    run([python(), str(SHARED / "build_figures.py"), "--data", str(DATA), "--out", str(redraw)])
    for name in FIGURE_NAMES:
        committed, fresh = FIGURES / f"{name}.pdf", redraw / f"{name}.pdf"
        check(f"{name}.pdf SHA-256", fresh.is_file() and sha(fresh) == sha(committed))

    print("2. Numeric tables regenerated from report/shared/data")
    tscratch = generate_tables(scratch / "tables")
    for lang in LANGS:
        for name in TABLE_NAMES:
            committed = TABLES / lang / f"{name}.tex"
            fresh = tscratch / lang / "tables" / f"{name}.tex"
            ok = committed.is_file() and fresh.is_file() and sha(fresh) == sha(committed)
            check(f"{lang}/{name}.tex", ok)

    print("3. Recompiled PDFs versus the shipped PDFs")
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  SKIP  pypdf not installed (pip install -r requirements-audit.txt)")
    else:
        for lang in LANGS:
            rebuilt = BUILD / lang / "main.pdf"
            shipped = REPORT / "pdf" / f"dispatch_value_report_{lang}.pdf"
            if not rebuilt.is_file():
                print(f"  SKIP  {lang}: no build/{lang}/main.pdf (run tools/build_report.py first)")
                continue
            a, b = PdfReader(rebuilt), PdfReader(shipped)
            check(f"{lang} page count", len(a.pages) == len(b.pages),
                  f"{len(a.pages)} vs {len(b.pages)}")
            same = len(a.pages) == len(b.pages) and all(
                x.extract_text() == y.extract_text() for x, y in zip(a.pages, b.pages))
            check(f"{lang} page text identical", same)

    print("4. No source data or model weights in the repository")
    banned = sorted(
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and BUILD not in p.parents
        and p.suffix in (".parquet", ".npz", ".pt", ".nc", ".cbm", ".joblib")
    )
    check("no .parquet/.npz/.pt/.nc/.cbm/.joblib tracked", not banned, ", ".join(banned[:5]))

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
