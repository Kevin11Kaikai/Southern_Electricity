"""Package each report edition as a self-contained Overleaf project.

    python tools/build_report.py && python tools/make_overleaf_zip.py

Writes build/dispatch_value_report_{zh,en}.zip. Upload either as a new Overleaf
project, set main.tex as the main file, and select XeLaTeX. Edit author.tex to
add a name and affiliation; it is deliberately left blank.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from _common import BUILD, LANGS, ROOT, sha

README = """# {title}

Main file: `main.tex`. Compiler: **XeLaTeX** (not pdfLaTeX), on a current TeX Live.
The Chinese edition uses ctex with the bundled Fandol fonts; both editions share
the same English vector figures.

Add your name and affiliation in `author.tex`.

`data/` holds the aggregated evidence records the figures and tables are drawn
from. `scripts/build_figures.py` and `scripts/build_tables.py` regenerate every
quantitative figure and table from those records alone; no model is fitted.

The source dataset is not included. See the project repository for DATA.md,
which explains how to obtain it and rebuild the upstream experiment outputs.
"""

TITLES = {
    "zh": "从点预测误差到受约束调度价值（中文版）",
    "en": "From Pointwise Forecast Error to Constrained Dispatch Value (English edition)",
}


def main() -> int:
    staging = BUILD / "staging"
    if not staging.is_dir():
        raise SystemExit("build/staging is missing. Run tools/build_report.py first.")

    for lang in LANGS:
        source = staging / lang
        bbl = BUILD / lang / "main.bbl"
        if bbl.is_file():
            (source / "main.bbl").write_bytes(bbl.read_bytes())
        (source / "README.md").write_text(README.format(title=TITLES[lang]), encoding="utf-8")

        target = BUILD / f"dispatch_value_report_{lang}.zip"
        files = sorted(p for p in source.rglob("*") if p.is_file())
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, path.relative_to(source).as_posix())

        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            assert archive.testzip() is None
            assert {"main.tex", "equations.tex", "preamble.tex"} <= set(names)
            assert all(not n.startswith(("/", "\\")) and ".." not in Path(n).parts for n in names)
            assert not any(n.endswith((".parquet", ".npz", ".pt", ".nc")) for n in names)
        print(f"  {target.relative_to(ROOT)}  {target.stat().st_size:,} bytes  "
              f"{len(files)} files  sha256={sha(target)[:16]}")

    print("\nUpload either ZIP to Overleaf as a new project; compiler: XeLaTeX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
