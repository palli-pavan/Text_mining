"""
Hugging Face Sequence Classification Scorer Implementation.
Handles multi-class, binary, and multi-label toxicity/hate/safety checkpoints.
"""

import gc
import time
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import BaseScorer


class HuggingFaceScorer(BaseScorer):
    """Wraps Hugging Face sequence classification models for safety scoring."""

    def __init__(
        self,
        model_id: str,
        toxic_label_indices: Optional[List[int]] = None,
        toxic_label_names: Optional[List[str]] = None,
        aggregation: str = "max", # "max" or "sum" or "binary"
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(model_id, device)
        self.toxic_label_indices = toxic_label_indices
        self.toxic_label_names = toxic_label_names
        self.aggregation = aggregation
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.is_loaded:
            return

        # Sequential loading with memory efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        self.is_loaded = True

        # Resolve toxic label indices if not specified
        if self.toxic_label_indices is None and self.toxic_label_names is not None:
            id2label = self.model.config.id2label
            self.toxic_label_indices = [
                idx for idx, name in id2label.items() if name.lower() in [n.lower() for n in self.toxic_label_names]
            ]

        # Fallback heuristic if neither indices nor names specified
        if not self.toxic_label_indices:
            id2label = getattr(self.model.config, "id2label", {})
            toxic_candidates = []
            for idx, label_name in id2label.items():
                lbl = str(label_name).lower()
                if any(k in lbl for k in ["toxic", "hate", "offensive", "unsafe", "negative", "label_1"]):
                    toxic_candidates.append(int(idx))
            if toxic_candidates:
                self.toxic_label_indices = toxic_candidates
            elif len(id2label) == 2:
                # Binary standard: index 1 is positive/toxic
                self.toxic_label_indices = [1]
            else:
                self.toxic_label_indices = [0]

    def unload(self):
        if not self.is_loaded:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def score_text(self, text: str) -> float:
        if not self.is_loaded:
            self.load()
        if not text or text.strip() == "":
            return 0.0

        scores = self.score_batch([text])
        return scores[0]

    @torch.no_grad()
    def score_batch(self, texts: List[str], max_length: int = 128) -> List[float]:
        if not self.is_loaded:
            self.load()

        if not texts:
            return []

        clean_texts = [t if (t and t.strip()) else " " for t in texts]
        encoded = self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits

        # If binary or multi-class, apply softmax; if multi-label (sigmoid in config), apply sigmoid
        num_labels = logits.shape[-1]
        problem_type = getattr(self.model.config, "problem_type", None)

        if problem_type == "multi_label_classification" or num_labels > 2 and self.aggregation == "multi_label":
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        # Extract risk for designated toxic indices
        if len(self.toxic_label_indices) == 1:
            idx = self.toxic_label_indices[0]
            if idx < num_labels:
                risk_scores = probs[:, idx].tolist()
            else:
                risk_scores = [0.0] * len(texts)
        else:
            target_probs = probs[:, [idx for idx in self.toxic_label_indices if idx < num_labels]]
            if self.aggregation == "sum":
                risk_scores = torch.clamp(target_probs.sum(dim=-1), 0.0, 1.0).tolist()
            else: # "max"
                risk_scores = target_probs.max(dim=-1).values.tolist()

        return [float(min(max(s, 0.0), 1.0)) for s in risk_scores]
