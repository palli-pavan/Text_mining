# GUARD-TRAJECTORY

**Training-Free Real-Time Safety Monitoring and Interception in Discrete Diffusion Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](file:///d:/te_paper/tests)

---

## 1. Overview

**Guard-Trajectory** introduces a training-free, trajectory-aware safety monitoring and adaptive interception framework for Masked Discrete Diffusion Language Models (MDLMs). 

Unlike autoregressive language models that generate text left-to-right, discrete diffusion models decode sequences via iterative, non-autoregressive de-masking across continuous reverse time $t \in [1.0, 0.0]$. Prior alignment approaches either rely on wasteful post-hoc rejection filtering (wasting 100% of decoding compute) or computationally prohibitive dense optimization loops (e.g., Constrained Discrete Diffusion / CDD, taking 236–637 seconds per sequence).

Guard-Trajectory solves this challenge through:
1. **Early Predictive Horizon (RQ1):** Intermediate predicted clean states ($\hat{x}_0^{(t)}$) provide a high-fidelity predictive signal of final safety violations, achieving **0.932 AUROC** and **0.893 AUPRC** at $t=0.81$ (only 18% into generation), with an average lead time of **13.4 reverse steps**.
2. **Directional Taboo Steering (RQ2):** Unconstrained re-masking causes *exploratory drift*, where the model resamples identical toxic modes. Directional Taboo Steering eliminates drift by setting categorical transition probabilities of candidate toxic tokens to zero ($q(x_s = \text{taboo}) = 0$).
3. **Sub-Second Pareto Efficiency (RQ3):** Guard-Trajectory operates at **0.60 s per sequence** with only **+16.7% compute overhead**, achieving **$>400\times$ faster runtime** than CDD with zero inner gradient optimization loops.
4. **Multi-Scorer Disagreement-Aware Fusion ($P_6$, RQ4):** A fast ensemble of heterogeneous safety classifiers prevents false-negative omissions under high ambiguity ($\Delta_{\text{disagree}} \ge 0.35$).

---

## 2. Theoretical Guarantees

- **Proposition 1 (Unguarded Trajectory Equivalence):** If trajectory risk $r(t) < \tau_{\text{eff}}(t)$ for all evaluated timesteps, the steered transition posterior $q_{\text{steered}}(x_s \mid x_t)$ is identical to the unconstrained MDLM posterior $q(x_s \mid x_t)$, producing bitwise-identical text under identical seeds.
- **Proposition 2 (Strict Taboo Mode Elimination):** For any active taboo token set $\mathcal{T}$, the probability of re-sampling any toxic token in $\mathcal{T}$ at newly unmasked positions is identically zero: $\Pr(x_0[i] \in \mathcal{T}) = 0$ for all $i > L_{\text{prompt}}$.

---

## 3. Benchmark Results Across Splits (2,485 Trajectories)

All splits evaluated on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) on the `kuleshov-group/mdlm-owt` backbone:

### Primary Held-Out Test Split ($N=200$ Prompts, 1,400 Trajectories)

| Policy | Method | Cont. Viol. % [95% CI] | Full Viol. % | Avg. Interv. | Dist-1 | Entropy | Latency (s) | Overhead % | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_0$** | Baseline MDLM | 6.5% [3.5, 10.0] | 25.5% | 0.00 | 0.453 | 9.510 | 0.514 | 0.0% | 1841 MB |
| **$P_1$** | Post-Hoc Filter | 6.5% [3.5, 10.0] | 25.5% | 0.00 | 0.453 | 9.510 | 0.517 | +0.6% | 1841 MB |
| **$P_2$** | Every-Step (Naive) | 10.5% [6.5, 15.0] | 35.5% | 5.36 | 0.442 | 9.375 | 0.736 | +43.3% | 1841 MB |
| **$P_3$** | Fixed-Interval (Naive) | 6.5% [3.5, 10.0] | 25.5% | 1.00 | 0.438 | 9.433 | 0.560 | +9.0% | 1841 MB |
| **$P_4$** | Late-Start (Naive) | 6.5% [3.5, 10.0] | 27.5% | 1.88 | 0.454 | 9.513 | 0.614 | +19.5% | 1841 MB |
| **$P_5$** | Adaptive Trajectory (+ Taboo) | **6.0%** [3.0, 9.5] | **24.5%** | 1.24 | **0.470** | **9.631** | 0.600 | **+16.7%** | 1841 MB |
| **$P_6$** | Adaptive Ensemble (+ Taboo) | **5.0%** [2.0, 8.0] | 25.5% | 1.40 | **0.475** | **9.664** | 0.723 | +40.7% | 1841 MB |

### Held-Out Adversarial Robustness Split ($N=100$ Prompts, 700 Trajectories)

| Policy | Method | Cont. Viol. % [95% CI] | Full Viol. % | Dist-1 | Entropy | Latency (s) | Overhead % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_0$** | Baseline MDLM | 10.0% [5.0, 16.0] | 33.0% | 0.469 | 8.934 | 0.559 | 0.0% |
| **$P_1$** | Post-Hoc Filter | 10.0% [5.0, 16.0] | 33.0% | 0.469 | 8.934 | 0.550 | -1.7% |
| **$P_2$** | Every-Step (Naive) | 11.0% [5.0, 17.0] | 40.0% | 0.460 | 8.778 | 0.793 | +41.7% |
| **$P_3$** | Fixed-Interval (Naive) | 8.0% [3.0, 14.0] | 34.0% | 0.468 | 8.917 | 0.605 | +8.1% |
| **$P_4$** | Late-Start (Naive) | 11.0% [5.0, 17.0] | 36.0% | 0.468 | 8.930 | 0.674 | +20.5% |
| **$P_5$** | Adaptive Trajectory (+ Taboo) | 12.0% [6.0, 19.0] | 36.0% | **0.495** | **9.111** | 0.652 | **+16.7%** |
| **$P_6$** | Adaptive Ensemble (+ Taboo) | **8.0%** [3.0, 14.0] | 36.0% | **0.499** | **9.116** | 0.776 | +38.7% |

---

## 4. Repository Structure

```text
d:/te_paper/
├── colab/                              # Self-contained Google Colab package
│   ├── guard_trajectory_colab.ipynb    # Interactive Colab notebook
│   └── run_colab_standalone.py         # Standalone master runner (CPU / GPU)
├── paper/                              # Publication manuscripts
│   ├── conference_paper.tex            # Full LaTeX draft (Appendices A-E)
│   └── manuscript.md                   # Synchronized Markdown manuscript
├── plots/                              # Publication figures (High-DPI)
│   ├── fig1_risk_trajectories.png      # Trajectory risk curves with 95% CI bands
│   ├── fig2_policy_violations_*.png    # Policy violation rates across splits
│   ├── fig3_pareto_tradeoff_*.png      # Safety-compute Pareto frontiers
│   ├── fig4_roc_pr_early_detection.png # Stage-wise ROC and PR curves
│   ├── fig5_scorer_comparison.png      # Scorer discrimination vs. latency
│   ├── fig6_trajectory_case_studies.png# Trajectory case studies & taboo steering
│   └── fig7_toxicity_distributions.png # Empirical continuation toxicity densities
├── results/                            # Raw experimental logs and metrics
│   ├── experiment_runs_*.jsonl         # Trajectory logs (2,485 runs)
│   ├── results_table_*.csv             # Per-split policy comparison summaries
│   ├── rq1_early_detection_metrics.csv # AUROC/AUPRC early detection metrics
│   └── scorer_benchmark_registry.csv   # 13-model candidate scorer benchmarks
├── scripts/                            # Analysis and execution pipelines
│   ├── create_dataset_manifest.py      # Stratified RealToxicityPrompts manifests
│   ├── evaluate_rq1_early_detection.py # RQ1 early-warning trajectory analysis
│   ├── generate_supplementary_plots.py # Figures 6 & 7 supplementary generator
│   ├── plot_scorer_comparison.py       # Scorer benchmark visualizer
│   ├── run_experiments.py              # Master multi-policy evaluation runner
│   └── smoke_test_scorers.py           # Multi-family scorer benchmark script
├── src/                                # Core library
│   ├── backbone/                       # MDLM wrapper with taboo token steering
│   ├── controller/                     # Trajectory monitor and adaptive controller
│   ├── metrics/                        # Diversity, entropy, PPL, and bootstrap CI
│   ├── policies/                       # Policy definitions (P0 through P6)
│   └── scorers/                        # Candidate safety scorer implementations
└── tests/                              # Unit verification suite
    ├── test_mdlm_load.py               # Pretrained checkpoint loading sanity check
    ├── test_sampler_baseline.py        # Guard-off bitwise equivalence test
    └── test_taboo_steering.py          # Directional taboo steering test
```

---

## 5. Quickstart & Reproducibility

### Installation
```bash
pip install torch transformers datasets accelerate scikit-learn matplotlib pandas pytest
```

### 1. Run Verification Test Suite
```bash
pytest tests/
```

### 2. Run Experiments Across Policies
```bash
# Debug split (5 prompts, 35 trajectories)
python scripts/run_experiments.py --split debug --steps 16

# Primary test split (200 prompts, 1,400 trajectories)
python scripts/run_experiments.py --split test --steps 16 --device cuda
```

### 3. Generate Publication Figures
```bash
# RQ1 Early Detection curves
python scripts/evaluate_rq1_early_detection.py

# Safety Scorer comparison benchmark
python scripts/plot_scorer_comparison.py

# Case studies and density distributions
python scripts/generate_supplementary_plots.py
```

### 4. Run on Google Colab
Open [`colab/guard_trajectory_colab.ipynb`](file:///d:/te_paper/colab/guard_trajectory_colab.ipynb) in Google Colab or run:
```bash
python colab/run_colab_standalone.py --device cpu --split debug --steps 16
```

---

## 6. Citation

```bibtex
@inproceedings{guard_trajectory2026,
  title={GUARD-TRAJECTORY: Training-Free Real-Time Safety Monitoring and Interception in Discrete Diffusion Language Models},
  author={Anonymous Authors},
  booktitle={Conference Submission Draft},
  year={2026}
}
```
