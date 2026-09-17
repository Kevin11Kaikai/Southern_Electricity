"""Rebuild both report editions from the packaged evidence records.

    python tools/build_report.py

Reads only report/shared/data/. Draws the six figures, generates the seven
numeric tables, stages self-contained language trees, and compiles both PDFs
into build/. No model is fitted and no source dataset is read.
"""
from __future__ import annotations

import argparse
import shutil

from _common import (BUILD, FIGURES, LANGS, SHARED, TABLE_NAMES, TABLES, compile_tex,
                     generate_tables, python, run, stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-compile", action="store_true",
                        help="regenerate figures and tables but do not run LaTeX")
    args = parser.parse_args()

    print("[1/4] drawing figures from report/shared/data")
    run([python(), str(SHARED / "build_figures.py"),
         "--data", str(SHARED / "data"), "--out", str(FIGURES)])

    print("[2/4] generating numeric tables into report/shared/tables")
    scratch = generate_tables(BUILD / "tables_scratch")
    for lang in LANGS:
        (TABLES / lang).mkdir(parents=True, exist_ok=True)
        for name in TABLE_NAMES:
            shutil.copy2(scratch / lang / "tables" / f"{name}.tex", TABLES / lang / f"{name}.tex")

    print("[3/4] staging self-contained language trees")
    staging = BUILD / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    stage(staging)

    if args.skip_compile:
        print("[4/4] skipped (--skip-compile)")
        print(f"\nStaged sources: {staging}")
        return 0

    print("[4/4] compiling both editions")
    for lang in LANGS:
        pdf = compile_tex(staging / lang, BUILD / lang)
        print(f"  {lang}: {pdf}")

    print("\nDone. Compare against the shipped PDFs with:")
    print("  python tools/verify_reproduction.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
