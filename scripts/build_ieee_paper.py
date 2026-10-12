"""
Converts conference_paper.tex into official IEEE conference format (IEEEtran.cls).
Properly formats:
- \\documentclass[conference]{IEEEtran}
- IEEE author blocks
- IEEEkeywords
- table* and figure* for two-column page spans
- IEEEtran bibliography integration
"""

import re
import os

with open("paper/conference_paper.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Replace Document Class and Preamble
ieee_preamble = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}
\usepackage{microtype}

\graphicspath{{../plots/}{plots/}}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\newtheorem{proposition}{Proposition}

\begin{document}

\title{GUARD-TRAJECTORY: Training-Free Real-Time Safety Monitoring and Interception in Discrete Diffusion Language Models}

\author{\IEEEauthorblockN{Anonymous Authors}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{IEEE Conference Submission Draft} \\
anonymous@conference.org}
}

\maketitle

\begin{abstract}
Masked Discrete Diffusion Language Models (MDLMs) offer a compelling alternative to autoregressive architectures, generating text through iterative, non-autoregressive de-masking. However, aligning discrete diffusion trajectories to safety constraints remains an open challenge. Conventional post-hoc rejection filters discard complete generations upon constraint violation, wasting 100\% of the decoding compute. Existing constrained discrete diffusion methods rely on uniform step-wise projections that incur prohibitive computational overhead. In this paper, we propose \textbf{Guard-Trajectory}, a training-free, trajectory-aware safety monitoring and adaptive interception framework for discrete diffusion. We first investigate whether intermediate discrete diffusion states predict final safety violations prior to completion (RQ1). Empirically evaluating reverse diffusion trajectories on stratified RealToxicityPrompts, we show that intermediate predicted clean states ($\hat{x}_0^{(t)}$) provide a remarkably early, high-fidelity predictive signal: achieving \textbf{0.932 AUROC} and \textbf{0.893 AUPRC} as early as $t=0.81$ (18\% into reverse diffusion), providing an average early detection lead time of \textbf{13.4 diffusion steps}. Leveraging this early predictive window, we formulate an adaptive controller governed by stage-weighted thresholding, temporal persistence filtering, and \textbf{Directional Taboo Steering}. We demonstrate that naive unconstrained re-masking induces exploratory drift, whereas directional taboo steering eliminates drift, matches or reduces baseline violations, and enhances unigram diversity ($\text{Dist-1}=0.470$) and entropy ($9.631$) while incurring only 16.7\% latency overhead compared to 43.3\% for dense step-wise checking. Finally, we establish a multi-family candidate safety scorer registry spanning 13 models to identify optimal speed-to-discrimination Pareto points for real-time trajectory tracking.
\end{abstract}

\begin{IEEEkeywords}
Discrete diffusion language models, trajectory safety monitoring, directional taboo steering, training-free alignment, non-autoregressive generation.
\end{IEEEkeywords}
"""

# Extract content starting from Section 1 Introduction up to the bibliography
intro_match = re.search(r"(\\section\{Introduction\}.*?)\\begin\{thebibliography\}", text, re.DOTALL)
if not intro_match:
    raise ValueError("Could not find section Introduction up to bibliography!")

body_text = intro_match.group(1)

# Format wide tables and figures as table* and figure* for IEEE two-column layout
# Table on line 219 (Stage metrics)
body_text = body_text.replace(
    r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{ccccccc}",
    r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{ccccccc}"
)
# Table on line 262 (Primary test results)
body_text = body_text.replace(
    r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{lcccccc}",
    r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{lcccccc}"
)
# Table on line 343 (Robustness results)
body_text = body_text.replace(
    r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{lcccccc}",
    r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\begin{tabular}{lcccccc}"
)

# Convert minipage figures to figure* for two-column span
body_text = body_text.replace(r"\begin{figure}[t]" + "\n" + r"\centering" + "\n" + r"\begin{minipage}{0.48\textwidth}",
                              r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\begin{minipage}{0.48\textwidth}")

# Fix matching \end{table} and \end{figure}
body_text = body_text.replace(r"\label{tab:rq1_stages}" + "\n" + r"\end{table}",
                              r"\label{tab:rq1_stages}" + "\n" + r"\end{table*}")
body_text = body_text.replace(r"\label{tab:main_results}" + "\n" + r"\end{table}",
                              r"\label{tab:main_results}" + "\n" + r"\end{table*}")
body_text = body_text.replace(r"\label{tab:robustness}" + "\n" + r"\end{table}",
                              r"\label{tab:robustness}" + "\n" + r"\end{table*}")
body_text = body_text.replace(r"\label{fig:trajectories}" + "\n" + r"\end{figure}",
                              r"\label{fig:trajectories}" + "\n" + r"\end{figure*}")
body_text = body_text.replace(r"\label{fig:violations_pareto}" + "\n" + r"\end{figure}",
                              r"\label{fig:violations_pareto}" + "\n" + r"\end{figure*}")

# Extract appendix content
app_match = re.search(r"(\\appendix.*)\\end\{document\}", text, re.DOTALL)
app_text = app_match.group(1) if app_match else ""

# Format appendix wide tables and figures as table* and figure*
app_text = app_text.replace(r"\begin{figure}[h]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\textwidth]",
                            r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\textwidth]")
app_text = app_text.replace(r"\label{fig:app_trajectories}" + "\n" + r"\end{figure}",
                            r"\label{fig:app_trajectories}" + "\n" + r"\end{figure*}")

# IEEE Bibliography Section
ieee_bib = r"""
\bibliographystyle{IEEEtran}
\bibliography{references}
"""

ieee_full_paper = ieee_preamble + "\n" + body_text.strip() + "\n" + ieee_bib.strip() + "\n\n" + app_text.strip() + "\n\\end{document}\n"

with open("paper/ieee_conference_paper.tex", "w", encoding="utf-8") as f:
    f.write(ieee_full_paper)

print("paper/ieee_conference_paper.tex written successfully!")
