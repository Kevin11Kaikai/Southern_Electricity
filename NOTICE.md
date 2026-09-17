# Notice: licensing and data attribution

## Code and documentation

MIT, see [LICENSE](LICENSE). This covers everything under `src/`,
`experiments/`, `tools/`, `notebooks/`, `report/shared/*.py`, and the Markdown
documents in this repository.

## The report text and figures

The report source under `report/` and the compiled PDFs in `report/pdf/` are
released under the same MIT terms. If you quote results, please carry the stated
limits with them — in particular that the reported gain is an observed value on
one 59-day window whose paired interval crosses zero, not a transferable effect
size.

## Evidence records

`report/shared/data/` contains sixteen aggregated records derived from the
competition dataset: per-day settlement scores, per-candidate summary metrics,
correlations and their resampling intervals. They are statistics **about** model
outputs, not the dataset. The finest granularity is one score per model per day;
no price curve and no 15-minute observation is included.

They are published so the report's arithmetic can be checked independently. They
are not a substitute for the dataset and cannot be used to reconstruct it.

## The dataset itself

**Not distributed here.** The source data comes from the AI4S competition:

> <https://competition.ai4s.com.cn/race/9/description>

Rights remain with the competition organizers and the data provider. Obtain it
under their terms; this repository asserts no right to redistribute it and makes
no representation about what those terms permit.

The series is anonymized and rescaled: the node is reduced to the label `A`, and
values carry no physical unit and no currency. Nothing in this repository
identifies a market participant, a settlement point, or a real revenue figure.

## Third-party references

The five works cited in the report are cited for methodological background only.
No dataset, code or result from them is redistributed here.

## AI assistance

The report prose, the build and audit scripts, and this documentation were
prepared with AI assistance working from the project's recorded experiments.
Every quantitative claim is traceable to `report/shared/data/` and to the two
independent review rounds recorded in `audit_records/`.
