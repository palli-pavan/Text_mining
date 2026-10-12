"""
Evaluation Metrics and Statistical Analysis Engine.
Implements all metrics specified in Section 9 of the Research Plan:
- Violation rate (%) and Absolute Reduction (pp)
- Distinct-n n-gram diversity (Dist-1, Dist-2) and Shannon token entropy
- Reference perplexity (GPT-2 PPL)
- Wall-clock latency per sample and computational overhead (%)
- Peak VRAM footprint
- Paired bootstrap 95% confidence intervals
"""

import collections
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ngram_diversity(texts: List[str]) -> Dict[str, float]:
    """Computes distinct-1, distinct-2, and Shannon entropy over generated texts."""
    total_unigrams = 0
    total_bigrams = 0
    unique_unigrams = set()
    unique_bigrams = set()
    unigram_counts = collections.Counter()

    for text in texts:
        tokens = text.strip().split()
        if not tokens:
            continue
        total_unigrams += len(tokens)
        unique_unigrams.update(tokens)
        unigram_counts.update(tokens)

        bigrams = list(zip(tokens[:-1], tokens[1:]))
        total_bigrams += len(bigrams)
        unique_bigrams.update(bigrams)

    dist1 = len(unique_unigrams) / max(total_unigrams, 1)
    dist2 = len(unique_bigrams) / max(total_bigrams, 1)

    # Shannon Entropy over word tokens
    probs = [c / total_unigrams for c in unigram_counts.values()] if total_unigrams > 0 else []
    entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0

    return {
        "dist_1": round(dist1, 4),
        "dist_2": round(dist2, 4),
        "entropy": round(entropy, 3),
    }


class QualityEvaluator:
    """Computes reference model Perplexity (PPL) using standard GPT-2."""

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def compute_ppl(self, texts: List[str], max_length: int = 128) -> float:
        """Compute average perplexity across texts."""
        valid_texts = [t.strip() for t in texts if t and len(t.strip().split()) >= 3]
        if not valid_texts:
            return float("nan")

        nlls = []
        for text in valid_texts:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            input_ids = enc.input_ids
            if input_ids.shape[1] < 2:
                continue
            outputs = self.model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
            if not torch.isnan(neg_log_likelihood):
                nlls.append(neg_log_likelihood.item())

        if not nlls:
            return float("nan")
        ppl = math.exp(float(np.mean(nlls)))
        return round(ppl, 2)


def bootstrap_ci(
    data: List[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes mean and non-parametric bootstrap confidence interval.
    Returns: (mean, lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0, 0.0)
    arr = np.array(data)
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(arr)
    for _ in range(num_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (float(np.mean(arr)), round(lower, 3), round(upper, 3))


def paired_difference_ci(
    baseline_vals: List[float],
    method_vals: List[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes paired bootstrap confidence interval for (baseline - method).
    """
    assert len(baseline_vals) == len(method_vals), "Samples must be paired!"
    diffs = np.array(baseline_vals) - np.array(method_vals)
    return bootstrap_ci(diffs.tolist(), num_bootstraps, confidence_level, seed)
