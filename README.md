# From Pointwise Forecast Error to Constrained Dispatch Value

Reproduction package for a bilingual technical report on **what happens when a
price forecast is judged by the scheduling decision it produces rather than by
its own error.**

*中文说明见 [README_ZH.md](README_ZH.md)。*

📄 **Read the report:** [`report/pdf/dispatch_value_report_en.pdf`](report/pdf/dispatch_value_report_en.pdf) (18 pages) ·
[`report/pdf/dispatch_value_report_zh.pdf`](report/pdf/dispatch_value_report_zh.pdf) (16 pages)

## Method at a glance

[![SCALE method overview: learn relative charging and discharging window values, average five forecasts, and select a feasible dispatch schedule.](docs/assets/scale_method_overview_20260917.png)](docs/assets/scale_method_overview_20260917.png)

SCALE connects decision-oriented learning to constrained storage dispatch:
learn relative window values → average forecasts → select a feasible schedule.
Figure 1 from the revised report dated 17 September 2026; the bundled PDFs above
are the earlier report edition. Click the figure to view it at full resolution.

---

## The problem

A day is 96 fifteen-minute slots. A battery may charge at fixed power for eight
consecutive slots, then discharge for eight, or stay idle — 3321 legal active
schedules plus idle. A forecaster predicts the 96 prices; a solver picks one
schedule from the forecast; the schedule is then settled against the prices that
actually occurred.

The project began by ranking forecasters on RMSE. Replaying the saved forecasts
through the scheduler showed that **lower pointwise error did not reliably
identify higher realized dispatch value.** The report works out why from the
geometry of the feasible action set, reformulates the learning objective around
the decision, implements a trainable surrogate, and evaluates the result under a
protocol frozen before scoring.

## What the evidence supports — and what it does not

| Claim | Status |
|---|---|
| On the specified 59-day window the frozen protocol realized **6693.88** vs **6214.36** for the MSE-trained reference — an observed gain of **479.51 (7.72%)** | Settled fact, independently re-derived |
| Point error does not reliably rank dispatch value across the saved candidates | Supported (v1 *r* = +0.351, v2 *r* = +0.005; day-centered variants likewise) |
| Scale-free window-ranking consistency tracks realized value far better than point error | Supported as a **structural diagnostic** (*r* = +0.849 / +0.777), not as validated model-selection ability |
| The actual training surrogate is a better cross-model selector | **Not supported** — its association is unstable (*r* = +0.320 / −0.165) and the report says so |
| The gain transfers to a new period | **Not established.** 86.0% of the net gain falls on one day; the paired 7-day block interval is [−227.87, 1626.28]; the headline's worst single day (−5922.66) is deeper than the reference's (−2340.48) |
| The loss function alone caused the gain | **Not established.** +160.99 / +318.52 is an arithmetic split, not a causal decomposition; a matched-initialization control of the *older* loss scores higher than the new one's single-seed mean |
| A full battery model was validated | **No.** No SOC, efficiency, degradation or network constraint is implemented; results hold only on the 8+8 action contract |

The 59-day evaluation window had already been inspected earlier in the project.
It is described throughout as a *historically exposed evaluation period*, not a
prospective confirmation set.

## Reproduce it

Three entry points, in increasing depth.

### 1. Read it

Open the PDFs in `report/pdf/`. No toolchain needed.

### 2. Rebuild every figure, table and PDF (≈2 minutes)

```bash
pip install -r requirements.txt -r requirements-audit.txt
python tools/build_report.py          # 6 figures, 7 tables, both PDFs -> build/
python tools/verify_reproduction.py   # byte-compare against the committed artifacts
python tools/check_numbers.py         # every number traces to report/shared/data
```

`tools/build_report.py` reads only the aggregated evidence records in
`report/shared/data/`. It fits no model and reads no source data. A LaTeX engine
is required — [Tectonic](https://tectonic-typesetting.github.io) is detected
first, otherwise `latexmk` + `xelatex`.

For byte-identical figure output install `requirements.lock.txt` instead;
different matplotlib versions reproduce the same numbers but may emit
cosmetically different PDF bytes.

`tools/make_overleaf_zip.py` packages each edition for upload to Overleaf.

### 3. Re-derive everything from the source data

The dataset is **not distributed here** — see [DATA.md](DATA.md) for where to
obtain it, where to place it, and the exact order in which to run the eight
experiment directories. Once you have regenerated `reports/`:

```bash
python report/shared/derive_records.py --repo-root . --out report/shared/data
python tools/independent_audit.py     # re-solves all 62 curves without the report code
```

`tools/independent_audit.py` rebuilds the block sums with `np.convolve`,
re-solves every saved day through `src/phase_b/dispatch.py`, and compares against
the packaged records. Run against the original artifacts it reports zero action
mismatches across 62 curves × 59 days and agreement to 1.5 × 10⁻¹¹.

## Layout

```
report/shared/          single source of truth
  data/                 16 aggregated evidence records (no price curves)
  figures/              6 vector figures + PNG previews
  tables/               7 generated LaTeX tables
  equations.tex         all 19 numbered equations, shared by both editions
  derive_records.py     rebuilds data/ from reports/ (needs your own data)
  build_figures.py      draws figures/ from data/ alone
  build_tables.py       generates tables/ from data/ alone
report/{zh,en}/         main.tex + author.tex per edition
report/pdf/             the two compiled PDFs
src/phase_b/            the dispatch solver, action contract and feature pipeline
experiments/            the eight experiment directories (code only)
notebooks/              notebooks 11–14, read-only research record with outputs
tools/                  build, verify and audit entry points
audit_records/          two independent review rounds and their machine-readable output
configs/phase_b.toml    paths and split definition
```

`src/phase_b/dispatch.py` is the authority on the task: `optimize_day` enumerates
the 3321 legal windows, picks the argmax under a deterministic tie rule, and
`contracts.py` validates that a day's power vector is legal.

## Verification status

Everything below was checked locally, with a LaTeX engine and the pinned
package versions:

- both editions recompile from a clean checkout; page counts (16 / 18) and
  per-page extracted text match the shipped PDFs
- all six figure PDFs redraw with identical SHA-256
- all fourteen table files regenerate identically
- the Chinese and English editions carry the same 96 decimal values, so no claim
  is qualified in one language only
- every decimal in the text resolves to a value in `report/shared/data/`

**The packages were not tested on Overleaf online.** `overleaf_online_tested` is
`false` in the build metadata, and the ZIPs target ordinary XeLaTeX on TeX Live.

## Provenance and limits

The target is an anonymized price series. The referenced literature provides
methodological background; it is not evidence that the data come from any
particular electricity market, and the report does not claim a market identity,
authoritative physical units, or any production deployment. Scores are reported
in the original scoring units — they are not currency and not net trading profit.

The report and the audit scripts were prepared with AI assistance from the
project's recorded experiments; every quantitative claim is traceable to the
records in `report/shared/data/` and the two review rounds in `audit_records/`.
See [AUDIT.md](AUDIT.md) for what those reviews found, including a sign error in
an arithmetic sentence that the second round caught and fixed.

## Licence

Code and documentation: MIT, see [LICENSE](LICENSE).
Data records and dataset attribution: see [NOTICE.md](NOTICE.md).
