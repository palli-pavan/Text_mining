"""
Trajectory Monitor and Early Warning Analyzer for Discrete Diffusion.
Implements passive monitoring, trajectory risk logging, and predictive metrics
for RQ1 (Trajectory Early-Detection of Safety Violations).
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class TrajectoryMonitor:
    """Monitors discrete diffusion trajectory without modifying generation."""

    def __init__(
        self,
        scorer,
        check_interval: int = 1,
        risk_threshold: float = 0.5,
        persistence_window: int = 2,
    ):
        """
        Args:
            scorer: BaseScorer instance.
            check_interval: Number of diffusion steps between monitor evaluations.
            risk_threshold: Threshold above which a state is considered high-risk.
            persistence_window: Minimum consecutive checks required for persistent risk alarm.
        """
        self.scorer = scorer
        self.check_interval = check_interval
        self.risk_threshold = risk_threshold
        self.persistence_window = persistence_window
        self.reset()

    def reset(self):
        """Reset state between generation runs."""
        self.trajectory_records: List[Dict] = []
        self.consecutive_risks = 0
        self.first_flagged_step = None
        self.first_flagged_t = None

    def monitor_step(self, step_info: Dict, tokenizer) -> Dict:
        """
        Callback invoked at step t.
        Decodes predicted clean text x_0_hat and scores safety risk.
        """
        step_idx = step_info["step"]
        total_steps = step_info["num_steps"]
        t_val = step_info["t"]
        mask_ratio = step_info["mask_ratio"]

        # Only evaluate on configured intervals or final step
        should_check = (step_idx % self.check_interval == 0) or (step_idx == total_steps - 1)
        if not should_check:
            return {
                "step": step_idx,
                "t": t_val,
                "checked": False,
                "risk_score": None,
                "is_flagged": False,
                "consecutive_risks": self.consecutive_risks,
            }

        # Extract x_0_hat (predicted clean tokens)
        x_0_hat = step_info["x_0_hat"][0] # batch item 0
        clean_text = tokenizer.decode(x_0_hat.tolist(), skip_special_tokens=True)

        # Compute intermediate safety score
        risk_score = self.scorer.score_text(clean_text)

        if risk_score >= self.risk_threshold:
            self.consecutive_risks += 1
            if self.consecutive_risks >= self.persistence_window and self.first_flagged_step is None:
                self.first_flagged_step = step_idx
                self.first_flagged_t = t_val
        else:
            self.consecutive_risks = 0

        record = {
            "step": step_idx,
            "total_steps": total_steps,
            "t": t_val,
            "mask_ratio": mask_ratio,
            "checked": True,
            "candidate_text": clean_text,
            "risk_score": risk_score,
            "is_flagged": (risk_score >= self.risk_threshold),
            "consecutive_risks": self.consecutive_risks,
            "is_persistent": (self.consecutive_risks >= self.persistence_window),
        }
        self.trajectory_records.append(record)
        return record

    def get_summary(self) -> Dict:
        """Return summary of monitored trajectory."""
        scores = [r["risk_score"] for r in self.trajectory_records if r["risk_score"] is not None]
        return {
            "num_monitored_steps": len(scores),
            "max_risk": max(scores) if scores else 0.0,
            "mean_risk": float(np.mean(scores)) if scores else 0.0,
            "final_intermediate_risk": scores[-1] if scores else 0.0,
            "first_flagged_step": self.first_flagged_step,
            "first_flagged_t": self.first_flagged_t,
            "lead_time_steps": (self.trajectory_records[-1]["total_steps"] - self.first_flagged_step) if (self.first_flagged_step is not None and self.trajectory_records) else 0,
            "trajectory": self.trajectory_records,
        }


def compute_early_detection_metrics(
    trajectories_data: List[Dict],
    risk_threshold: float = 0.5,
) -> Dict:
    """
    Computes AUROC, AUPRC, Lead Time, and False Alarm rates across monitored runs.

    Args:
        trajectories_data: List of run records containing trajectory steps and final violation label.
    """
    if not trajectories_data:
        return {}

    final_labels = [int(r["final_risk"] >= risk_threshold) for r in trajectories_data]
    final_labels = np.array(final_labels)

    # If all labels are same class, AUROC cannot be calculated
    if len(np.unique(final_labels)) < 2:
        return {
            "auroc": None,
            "auprc": None,
            "violation_rate": float(np.mean(final_labels)),
            "sample_count": len(final_labels),
            "note": "Single class present in sample batch.",
        }

    # Evaluate prediction at intermediate milestones: t ~ 0.75, 0.50, 0.25
    milestones = [0.75, 0.50, 0.25]
    milestone_metrics = {}

    for target_t in milestones:
        intermediate_scores = []
        for run in trajectories_data:
            steps = run["trajectory"]
            # Find record closest to target_t
            valid_steps = [s for s in steps if s.get("risk_score") is not None]
            if not valid_steps:
                intermediate_scores.append(0.0)
                continue
            closest = min(valid_steps, key=lambda s: abs(s["t"] - target_t))
            intermediate_scores.append(closest["risk_score"])

        scores_arr = np.array(intermediate_scores)
        try:
            roc = roc_auc_score(final_labels, scores_arr)
            precision, recall, _ = precision_recall_curve(final_labels, scores_arr)
            pr_auc = auc(recall, precision)
        except Exception:
            roc, pr_auc = None, None

        milestone_metrics[f"t_{int(target_t*100)}"] = {
            "target_t": target_t,
            "auroc": float(roc) if roc is not None else None,
            "auprc": float(pr_auc) if pr_auc is not None else None,
        }

    # Lead time among true violations
    lead_times = [
        r["lead_time_steps"] for r in trajectories_data
        if (r["final_risk"] >= risk_threshold and r.get("lead_time_steps", 0) > 0)
    ]
    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "sample_count": len(final_labels),
        "violation_count": int(np.sum(final_labels)),
        "violation_rate": float(np.mean(final_labels)),
        "avg_lead_time_steps": avg_lead_time,
        "milestones": milestone_metrics,
    }
