# P1 post-hoc baseline correction

This note supersedes the P1 interpretation in the existing `D:\te_paper\README.md` and paper draft. It does not replace or rewrite the original raw experiment logs.

## Corrected baseline definition

P1 generates the same continuation as unguarded P0, then scores that continuation using `unitary/toxic-bert`. At risk threshold 0.5, it **delivers** the continuation if below threshold and **rejects** it otherwise. There is no resampling. The original P1 runner did not call its final filter, so its saved P1 outputs are simply unguarded trajectories and its original latency excludes the post-hoc decision.

The repaired runner at `D:\te_paper\scripts\run_experiments.py` now calls `evaluate_final`, logs `posthoc_accepted`, `delivered_continuation_text`, and `filter_latency_sec`, and adds output coverage to its summary. Original result files remain unchanged. Its P1 latency measures generation plus a separate filter-score call; the original score used to evaluate all policies is outside that timing. Treat the first-run P0/P1 timing comparison as warm-up sensitive.

## Saved-output reject-only P1 analysis

The analysis applies the filter to immutable saved P0 outputs, paired with the independent `s-nlp/roberta_toxicity_classifier` scores. See `posthoc_baseline_summary.csv` and `posthoc_baseline_decisions.csv`. The independent classifier is a proxy, not human ground truth.

| Split | Generated | Delivered | Rejected | Coverage | Independently flagged among delivered | Independently safe among rejected |
|---|---:|---:|---:|---:|---:|---:|
| Test | 200 | 187 | 13 | 93.5% | 1 | 1 |
| Robustness | 100 | 90 | 10 | 90.0% | 1 | 2 |

The filter rejects all continuations above its **own** threshold by definition; any claim of zero served violations under that same classifier is tautological. The independent-score column shows one accepted continuation flagged on each split. Because rejected outputs are absent rather than rewritten, P1 trades coverage for risk and is not directly comparable to P5/P6 on raw generation violation rate alone. Report coverage, residual risk, and cost together.

## Smoke verification

The updated runner completed five debug prompts × P0/P1 on the local RTX 3050. P1 logged two rejections and three accepted outputs, for 60% coverage. `delivered_continuation_text` is empty exactly for the rejected outputs. It logged an average 0.0398 seconds for the P1 filter on these five cases. Do not generalize its relative total latency from this small, order-sensitive run.

## Paper changes needed

1. Relabel the existing P1 row as *passive post-hoc scoring, filter not applied in original logs*, or replace it with the reject-only analysis above plus coverage. Do not call the old P1 row a functioning filter.
2. Describe all toxicity outcomes as automated proxy labels. The RQ1 AUROC uses the same scorer for intermediate prediction and final label; it is within-scorer predictability.
3. The original test comparison P5 6.0% and P6 5.0% versus P0 6.5% has paired intervals containing zero. The independent rescoring also has intervals including zero. Avoid conclusive safety-improvement wording.
4. For a latency comparison, run a fresh warm-up-controlled benchmark on the local GPU with a fixed prompt set and record sampling, filtering, and any resampling costs separately.
