# Guard-Trajectory: Inference-Time Monitoring and Interception for Masked Diffusion Language Models

**Research draft v2 — 13 October 2026.** This draft replaces the interpretation, not the raw data, of the earlier manuscript. It is not ready for conference submission. Numerical claims below are tied to saved logs; bracketed items require a fresh experiment or editorial decision.

## Abstract

Masked diffusion language models expose intermediate predictions while constructing a text continuation, creating an opportunity for inference-time monitoring without changing model weights. We implement Guard-Trajectory on a pretrained 169.6-million-parameter MDLM and study six monitoring or intervention policies against unguarded decoding. On 200 stratified RealToxicityPrompts test prompts, a toxicity classifier used both for control and evaluation flags 13/200 unguarded continuations, 12/200 under an adaptive single-scorer controller, and 10/200 under an adaptive ensemble. Paired bootstrap intervals for both reductions include zero. A distinct toxicity classifier, applied after generation, flags 13/200, 8/200, and 8/200 respectively; its paired intervals also include zero. On a separate 100-prompt robustness set, the primary classifier flags 10/100, 12/100, and 8/100, respectively, showing that the single-scorer policy does not improve every slice. Intermediate classifier scores on 50 development trajectories predict that same classifier's final full-text judgment at an early reverse step (AUROC 0.932), a measure of proxy-score predictability rather than independently validated safety. A corrected reject-only post-hoc baseline delivers 187/200 test outputs at 93.5% coverage. These experiments establish a runnable, low-memory monitoring prototype and identify the need for independent labels, stronger statistical power, and output-quality assessment before claiming a safety benefit.

## 1. Introduction

[Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) generate text through iterative refinement rather than a strictly left-to-right token stream. The intermediate predicted clean sequence is visible before decoding ends. We ask whether that visibility can support a practical safety monitor and whether intervention improves outcomes relative to simply filtering completed text. The present study focuses on toxic-language proxy scores for short continuations. It does not establish broader policy compliance or protection against all unsafe content.

Our contributions are a step-hook implementation on a pretrained MDLM, a paired seven-policy comparison on fixed prompts and seeds, and an audit that distinguishes generated-output toxicity from delivered-output coverage. We also report where the first implementation fails to support the intended claim: early detection and the primary endpoint share one scorer, a two-sentence scorer smoke test is not a model-selection benchmark, and observed policy differences are uncertain.

## 2. Related work

Sahoo et al.'s MDLM supplies the masked diffusion backbone and sampling framework. [RealToxicityPrompts](https://aclanthology.org/2020.findings-emnlp.301/) supplies naturally occurring prompts with toxicity metadata used here for stratification. [Constrained Discrete Diffusion](https://papers.nips.cc/paper_files/paper/2025/file/124c1cc9f7750f512ed81cdbf2fd5bf5-Paper-Conference.pdf) studies constraint satisfaction through a different optimization-based approach. We do not infer a speedup over that paper: its hardware, model, decoding length, and workload differ from ours.

## 3. Method

The backbone is the pretrained `kuleshov-group/mdlm-owt` checkpoint, loaded locally without fine-tuning. A 32-token sequence, including the fixed prompt prefix, is decoded over 16 reverse steps. At each step, the wrapper exposes the masked state, predicted clean tokens, proposed next state, and timestep. A step hook may return a revised next state or a set of taboo token IDs. A zero-hook path preserves the baseline generation under matched seeds.

For a candidate clean sequence, a pretrained `unitary/toxic-bert` model returns a scalar risk: the maximum sigmoid probability over its six toxicity labels. This is a proxy signal, and the checkpoint's multi-label configuration requires sigmoid rather than softmax. The policy score is computed on the decoded candidate **including the prompt**. This matters because the selected prompts deliberately include high-toxicity prefixes; the monitor may react to the prefix even when the continuation itself is benign. Final outcomes are therefore reported primarily on the continuation alone.

We compare unguarded decoding (P0); a reject-only post-hoc filter (P1); every-step rollback (P2); fixed-interval rollback (P3); late-start rollback (P4); adaptive single-scorer monitoring with rollback and taboo tokens (P5); and an adaptive two-scorer variant (P6). P5 skips timesteps above 0.85, checks even-numbered reverse steps, uses a threshold of `0.5 × (0.8 + 0.4t)`, requires two accumulated high-risk checks, and allows up to five interventions. P6 checks every second step and combines `unitary/toxic-bert` with `cardiffnlp/twitter-roberta-base-offensive`, using a 0.35 disagreement margin and the same persistence and budget limits.

Upon an intervention, P5/P6 remask positions newly unmasked in that step. Their taboo set consists of all token IDs in the candidate continuation, except the mask token, and is applied to subsequent categorical transitions. This is a broad candidate-token exclusion rule; the code does not identify which individual tokens are toxic. We therefore avoid claiming that a taboo token is a proven toxic token or that this mechanism guarantees safe final text.

P1 generates an untouched continuation, scores it, and delivers it only if its continuation score is below 0.5. It does not resample. The original saved P1 runs did not execute this final decision; their generated texts are identical to P0 on every test prompt. We reconstruct the correct reject-only P1 from saved P0 text and retain coverage as an explicit outcome. The current runner now logs accept/reject decisions and separate filter time for future experiments.

## 4. Experimental protocol

The manifest draws 355 nonduplicate prompts from RealToxicityPrompts: 5 debug, 50 development, 200 test, and 100 robustness. Each split is stratified by the source prompt toxicity into high, medium, and low bins. The saved policy comparison uses one matched seed per prompt and seven policy outputs: 2,485 trajectories across all four splits. The independent check does **not** regenerate text; it rescored the 1,400 test and 700 robustness outputs with a cached `s-nlp/roberta_toxicity_classifier`, using its toxic-class softmax probability and threshold 0.5. That scorer is independent of P0–P5 control, but it is still an automated proxy, not human ground truth.

The principal generated-output measure is the fraction of continuations with classifier risk at least 0.5. Paired bootstrap intervals compare a policy with P0 on the same prompts. For P1 we additionally report delivery coverage and flagged accepted outputs. The original end-to-end latency table times sampling before final evaluation and is unsuitable as a P1 filtering-cost comparison. A separate, warm-up-controlled development benchmark is reported in Section 5 once verified.

No independent human quality or safety ratings are available. Distinct-n and token entropy are diversity proxies, not judgments of fluency, meaning preservation, or usefulness. The scorer registry's two-sentence smoke check verifies loadability and an elementary score response; it is not a discrimination benchmark and should not rank candidate monitors.

## 5. Results

### 5.1 Within-scorer early prediction

On the 50 development P1 trajectories, the first inspected stage near `t=0.813` (about 18% of the reverse loop complete) has AUROC 0.9324 and area under the precision–recall curve 0.8932 for predicting the *same classifier's* final full-text threshold label. The reported mean lead is 13.36 of 16 reverse steps for positive cases. Since prompt text is present at every stage and the final label uses the same scorer, this is evidence of early **proxy-score consistency**, not an independently validated early-warning rate for harmful content. This analysis uses the development split and was not repeated with blinded external labels.

### 5.2 Generated-output policy comparison

| Policy | Test, primary proxy (N=200) | Robustness, primary proxy (N=100) | Test, independent proxy (N=200) | Robustness, independent proxy (N=100) |
|---|---:|---:|---:|---:|
| P0 unguarded | 13 (6.5%) | 10 (10%) | 13 (6.5%) | 9 (9%) |
| P1 original passive run | 13 (6.5%) | 10 (10%) | 13 (6.5%) | 9 (9%) |
| P2 every step | 21 (10.5%) | 11 (11%) | 18 (9%) | 9 (9%) |
| P3 fixed interval | 13 (6.5%) | 8 (8%) | 14 (7%) | 8 (8%) |
| P4 late start | 13 (6.5%) | 11 (11%) | 13 (6.5%) | 10 (10%) |
| P5 adaptive | 12 (6.0%) | 12 (12%) | 8 (4%) | 6 (6%) |
| P6 ensemble | 10 (5.0%) | 8 (8%) | 8 (4%) | 5 (5%) |

The original primary-proxy paired test reductions versus P0 are P5 +0.5 percentage points, 95% CI [-2.0, +2.5], and P6 +1.5 points, CI [-1.0, +4.5]. Both intervals include zero. For independent rescoring, P5 and P6 each show +2.5 points on the test set, with six prompts changing from flagged to unflagged and one in the opposite direction for each comparison; paired-bootstrap intervals start at zero. On robustness outputs, independent rescoring gives P5 +3.0 points, CI [-2.0, +8.0], and P6 +4.0 points, CI [0.0, +9.0]. These are exploratory estimates, especially because independent rescoring was added after inspection of the original results.

P2 has the highest primary-proxy test violation rate (10.5%), which is consistent with the possibility that repeated rollback can be counterproductive. The data do not identify a unique causal mechanism for that difference. P5 is worse than P0 on the original robustness proxy, while P6 is lower on both splits, but all observed differences are too uncertain for a definitive superiority claim.

### 5.3 Actual reject-only P1

Applying the post-hoc rule to saved P0 continuations delivers 187/200 test outputs (93.5% coverage) and 90/100 robustness outputs (90% coverage). The separate evaluator flags one delivered continuation in each split. The primary control scorer flags none among delivered outputs by construction, which must not be used as independent evidence of safety. Of the rejected outputs, the separate evaluator considers one test and two robustness continuations below its own threshold. A five-prompt GPU smoke run of the corrected runner logged the decisions and filter time, but it is not used for a latency claim.

### 5.4 Local timing

A warm-up-controlled local benchmark on the RTX 3050 uses 30 development prompts after three warm-up pairs, alternates policy order, and verifies exact P0/P1 token equality under matched seeds. Mean P0 generation time is 0.794 seconds. Mean P1 generation plus filtering is 0.872 seconds, including 0.036 seconds for the filter call. The paired mean overhead is 0.078 seconds (paired-bootstrap 95% CI [0.020, 0.137]), about 9.8% of the mean P0 time. Seven of the 30 P1 outputs were rejected. This is a short local benchmark for the reject-only baseline, not a cross-paper speed comparison. It does not include costs for resampling or serving a replacement after rejection.

## 6. Discussion and limitations

The experiment establishes technical feasibility on a local 4 GB GPU and exposes intermediate MDLM states for low-memory monitoring. The saved peak allocated memory is about 1,841 MB; this is PyTorch allocated GPU memory, not a guarantee about all GPU or host-memory use. Safety efficacy remains unresolved. We lack human adjudication, the controller's primary metric shares its scorer, and the independent evaluator remains fallible. The short continuation length and one-seed protocol leave small numbers of flagged outputs, limiting precision. The P5/P6 candidate-token taboo rule may suppress innocuous vocabulary. No ablation isolates its effect from check frequency, persistence, or rollback. Source prompt toxicity can trigger full-text monitoring, and the registry's tiny scorer smoke set offers no reliability calibration.

Before submission, we should freeze an evaluation protocol with fresh prompts, an independent primary evaluator and threshold, explicit delivered-output coverage, paired uncertainty, output-quality measures, and an ablation of rollback versus taboo steering. Without human reviewers, the paper must state that all safety and quality outcomes are automated proxies. Cross-paper runtime ratios require comparable hardware, model, decoding length, and implementation; we do not present one here.

## 7. Conclusion

Guard-Trajectory is a working inference-time monitoring and interception prototype for a pretrained MDLM. Intermediate proxy scores anticipate that scorer's final judgment, and local tests show that adaptive policies can change generated continuations within a 4 GB GPU budget. Current paired results do not establish a reliable reduction in independently assessed toxic output. The corrected post-hoc baseline makes the central trade-off explicit: lower delivered proxy risk requires rejection and reduced coverage. A new preregistered evaluation is necessary before claiming a safety advance.

## Reproducibility record

- Original experiment code, manifests, and logs: `D:\te_paper\src`, `D:\te_paper\scripts`, `D:\te_paper\data\manifests`, `D:\te_paper\results`.
- Corrected P1 record: `D:\te_paper\paper\P1_CORRECTION_2026_10_13.md` and `D:\te_paper\results\posthoc_baseline_*.csv`.
- Independent per-output scores: `D:\te_paper\results\mdlm_independent_{test,robustness}_scores.csv`. Audit and timing scripts: the `guard_trajectory` workspace directory containing this draft.
- Backbone: [MDLM checkpoint](https://huggingface.co/kuleshov-group/mdlm-owt). Prior work: [MDLM](https://arxiv.org/abs/2406.07524), [RealToxicityPrompts](https://aclanthology.org/2020.findings-emnlp.301/), [CDD](https://papers.nips.cc/paper_files/paper/2025/file/124c1cc9f7750f512ed81cdbf2fd5bf5-Paper-Conference.pdf).
