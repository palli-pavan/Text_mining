"""
Base interface and abstractions for safety / toxicity scorers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import torch


class BaseScorer(ABC):
    """Abstract base class for trajectory safety scoring models."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

    @abstractmethod
    def load(self):
        """Load model weights and tokenizer into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload model from VRAM/RAM to conserve hardware resources."""
        pass

    @abstractmethod
    def score_text(self, text: str) -> float:
        """
        Score a single string.
        Returns risk score in [0.0, 1.0], where higher indicates more toxic/unsafe.
        """
        pass

    @abstractmethod
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a list of strings."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model_id='{self.model_id}' loaded={self.is_loaded} device='{self.device}'>"
