"""
Safety Monitoring and Intervention Policies for Discrete Diffusion.
Implements Policies P0 through P6 per Section 7 of the Research Plan:
- P0: Baseline MDLM (unguarded)
- P1: Post-hoc Filter (rejection sampling / flag only)
- P2: Every-step Intervention
- P3: Fixed Interval Intervention
- P4: Late-start Intervention (ablation)
- P5: Adaptive Trajectory Guard (magnitude + persistence + stage + budget)
- P6: Adaptive Ensemble Guard (multi-scorer disagreement-aware)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class BaseGuardPolicy(ABC):
    """Abstract base class for trajectory guard policies."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset state between generation sequences."""
        self.step_history: List[Dict] = []
        self.interventions_count = 0
        self.checks_count = 0

    @abstractmethod
    def evaluate_step(
        self,
        step_info: Dict,
        tokenizer,
        mask_token_id: int = 50257,
    ) -> Optional[torch.Tensor]:
        """
        Evaluate intermediate state and return modified x_next if intervening,
        or None to accept the baseline reverse step.
        """
        pass

    def get_stats(self) -> Dict:
        """Return execution statistics."""
        return {
            "policy_name": self.name,
            "checks_count": self.checks_count,
            "interventions_count": self.interventions_count,
            "intervention_rate": (self.interventions_count / max(self.checks_count, 1)),
        }


class BaselinePolicy(BaseGuardPolicy):
    """P0: Baseline MDLM - No monitoring, no intervention."""

    def __init__(self):
        super().__init__("P0_baseline")

    def reset(self):
        super().reset()

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        # Unguarded: return None to preserve exact diffusion trajectory
        return None


class PostHocFilterPolicy(BaseGuardPolicy):
    """P1: Post-hoc Filter - Trajectory runs untouched; final output evaluated and flagged."""

    def __init__(self, scorer, threshold: float = 0.5):
        super().__init__("P1_post_hoc")
        self.scorer = scorer
        self.threshold = threshold
        self.is_rejected = False

    def reset(self):
        super().reset()
        self.is_rejected = False

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        # Post-hoc does not alter intermediate steps
        return None

    def evaluate_final(self, text: str) -> bool:
        """Returns True if output passes, False if rejected."""
        risk = self.scorer.score_text(text)
        self.is_rejected = (risk >= self.threshold)
        return not self.is_rejected


class EveryStepPolicy(BaseGuardPolicy):
    """P2: Evaluates risk at every reverse step and intervenes whenever risk exceeds threshold."""

    def __init__(self, scorer, threshold: float = 0.5, temperature_boost: float = 1.5):
        super().__init__("P2_every_step")
        self.scorer = scorer
        self.threshold = threshold
        self.temperature_boost = temperature_boost

    def reset(self):
        super().reset()

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        self.checks_count += 1
        x_0_hat = step_info["x_0_hat"][0]
        text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)
        risk = self.scorer.score_text(text)

        if risk >= self.threshold:
            self.interventions_count += 1
            # Intervention: roll back freshly unmasked risky positions and resample with temperature boost
            x_t = step_info["x_t"]
            x_next = step_info["x_next"]
            prompt_len = step_info.get("prompt_len", 0)

            # Positions that just changed from MASK to a token in x_next
            freshly_unmasked = (x_t == mask_token_id) & (x_next != mask_token_id)
            if prompt_len > 0:
                freshly_unmasked[:, :prompt_len] = False

            # Keep 50% most confident or re-mask to force alternative clean decoding path
            override_next = x_next.clone()
            override_next[freshly_unmasked] = mask_token_id
            return override_next

        return None


class FixedIntervalPolicy(BaseGuardPolicy):
    """P3: Fixed interval intervention - Checks every K steps."""

    def __init__(self, scorer, interval: int = 4, threshold: float = 0.5):
        super().__init__("P3_fixed_interval")
        self.scorer = scorer
        self.interval = interval
        self.threshold = threshold

    def reset(self):
        super().reset()

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        step_idx = step_info["step"]
        if step_idx % self.interval != 0:
            return None

        self.checks_count += 1
        x_0_hat = step_info["x_0_hat"][0]
        text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)
        risk = self.scorer.score_text(text)

        if risk >= self.threshold:
            self.interventions_count += 1
            x_t = step_info["x_t"]
            x_next = step_info["x_next"]
            prompt_len = step_info.get("prompt_len", 0)
            freshly_unmasked = (x_t == mask_token_id) & (x_next != mask_token_id)
            if prompt_len > 0:
                freshly_unmasked[:, :prompt_len] = False
            override_next = x_next.clone()
            override_next[freshly_unmasked] = mask_token_id
            return override_next

        return None


class LateStartPolicy(BaseGuardPolicy):
    """P4: Late-start intervention (checks only when t < t_start, e.g. t < 0.5)."""

    def __init__(self, scorer, t_start: float = 0.5, threshold: float = 0.5):
        super().__init__("P4_late_start")
        self.scorer = scorer
        self.t_start = t_start
        self.threshold = threshold

    def reset(self):
        super().reset()

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        t_val = step_info["t"]
        if t_val >= self.t_start:
            return None

        self.checks_count += 1
        x_0_hat = step_info["x_0_hat"][0]
        text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)
        risk = self.scorer.score_text(text)

        if risk >= self.threshold:
            self.interventions_count += 1
            x_t = step_info["x_t"]
            x_next = step_info["x_next"]
            prompt_len = step_info.get("prompt_len", 0)
            freshly_unmasked = (x_t == mask_token_id) & (x_next != mask_token_id)
            if prompt_len > 0:
                freshly_unmasked[:, :prompt_len] = False
            override_next = x_next.clone()
            override_next[freshly_unmasked] = mask_token_id
            return override_next

        return None


class AdaptiveTrajectoryPolicy(BaseGuardPolicy):
    """
    P5: Adaptive Trajectory Guard.
    Decides when to intervene based on:
    1. Risk magnitude r(t)
    2. Risk persistence (consecutive violations k >= k_min)
    3. Diffusion stage weighting w(t)
    4. Explicit intervention budget B
    """

    def __init__(
        self,
        scorer,
        base_threshold: float = 0.5,
        persistence_min: int = 2,
        check_interval: int = 2,
        max_budget: int = 5,
        early_stage_cutoff: float = 0.85,
    ):
        self.scorer = scorer
        self.base_threshold = base_threshold
        self.persistence_min = persistence_min
        self.check_interval = check_interval
        self.max_budget = max_budget
        self.early_stage_cutoff = early_stage_cutoff
        super().__init__("P5_adaptive_trajectory")

    def reset(self):
        super().reset()
        self.consecutive_high_risk = 0
        self.budget_remaining = self.max_budget

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        step_idx = step_info["step"]
        t_val = step_info["t"]

        # Do not waste checks in ultra-noisy initial step (t > early_stage_cutoff)
        if t_val > self.early_stage_cutoff:
            return None

        if step_idx % self.check_interval != 0:
            return None

        if self.budget_remaining <= 0:
            return None

        self.checks_count += 1
        x_0_hat = step_info["x_0_hat"][0]
        text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)
        risk = self.scorer.score_text(text)

        # Dynamic stage-weighted threshold: allow higher tolerance early, strict late
        # When t is high (early), effective threshold is slightly higher
        # When t is low (crystallizing), effective threshold is tighter
        effective_threshold = self.base_threshold * (0.8 + 0.4 * t_val)

        if risk >= effective_threshold:
            self.consecutive_high_risk += 1
        else:
            self.consecutive_high_risk = max(0, self.consecutive_high_risk - 1)

        # Intervene only if risk is persistent
        if self.consecutive_high_risk >= self.persistence_min and self.budget_remaining > 0:
            self.interventions_count += 1
            self.budget_remaining -= 1

            x_t = step_info["x_t"]
            x_next = step_info["x_next"]
            prompt_len = step_info.get("prompt_len", 0)

            # Roll back freshly unmasked positions to MASK
            freshly_unmasked = (x_t == mask_token_id) & (x_next != mask_token_id)
            if prompt_len > 0:
                freshly_unmasked[:, :prompt_len] = False

            override_next = x_next.clone()
            override_next[freshly_unmasked] = mask_token_id

            # Extract taboo tokens from risky continuation candidate
            risky_tokens = x_0_hat[prompt_len:].tolist()
            taboo_ids = [tok for tok in set(risky_tokens) if tok != mask_token_id]

            return {
                "x_next": override_next,
                "add_taboo_ids": taboo_ids,
            }

        return None


class AdaptiveEnsemblePolicy(BaseGuardPolicy):
    """
    P6: Adaptive Ensemble Guard.
    Combines primary toxicity scorer with secondary detector.
    Uses calibrated risk fusion and disagreement awareness.
    """

    def __init__(
        self,
        scorers: List,
        weights: Optional[List[float]] = None,
        base_threshold: float = 0.5,
        disagreement_margin: float = 0.35,
        persistence_min: int = 2,
        check_interval: int = 2,
        max_budget: int = 5,
    ):
        self.scorers = scorers
        self.weights = weights or [1.0 / len(scorers)] * len(scorers)
        self.base_threshold = base_threshold
        self.disagreement_margin = disagreement_margin
        self.persistence_min = persistence_min
        self.check_interval = check_interval
        self.max_budget = max_budget
        super().__init__("P6_adaptive_ensemble")

    def reset(self):
        super().reset()
        self.consecutive_high_risk = 0
        self.budget_remaining = self.max_budget

    def evaluate_step(self, step_info: Dict, tokenizer, mask_token_id: int = 50257) -> Optional[torch.Tensor]:
        step_idx = step_info["step"]
        if step_idx % self.check_interval != 0 or self.budget_remaining <= 0:
            return None

        self.checks_count += 1
        x_0_hat = step_info["x_0_hat"][0]
        text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)

        # Multi-scorer scoring
        scores = [scorer.score_text(text) for scorer in self.scorers]
        ensemble_risk = sum(w * s for w, s in zip(self.weights, scores))
        disagreement = max(scores) - min(scores)

        # If high disagreement, apply conservative safety bias
        if disagreement >= self.disagreement_margin:
            effective_risk = max(ensemble_risk, max(scores) * 0.85)
        else:
            effective_risk = ensemble_risk

        if effective_risk >= self.base_threshold:
            self.consecutive_high_risk += 1
        else:
            self.consecutive_high_risk = max(0, self.consecutive_high_risk - 1)

        if self.consecutive_high_risk >= self.persistence_min and self.budget_remaining > 0:
            self.interventions_count += 1
            self.budget_remaining -= 1

            x_t = step_info["x_t"]
            x_next = step_info["x_next"]
            prompt_len = step_info.get("prompt_len", 0)
            freshly_unmasked = (x_t == mask_token_id) & (x_next != mask_token_id)
            if prompt_len > 0:
                freshly_unmasked[:, :prompt_len] = False

            override_next = x_next.clone()
            override_next[freshly_unmasked] = mask_token_id

            risky_tokens = x_0_hat[prompt_len:].tolist()
            taboo_ids = [tok for tok in set(risky_tokens) if tok != mask_token_id]

            return {
                "x_next": override_next,
                "add_taboo_ids": taboo_ids,
            }

        return None
