"""
Candidate Safety Scorer Registry.
Defines 25-30 candidate models across 5 distinct families per Section 5.2 of the Research Plan:
1. Dedicated toxicity classifiers (8-10)
2. Hate/offensive-language classifiers (4-5)
3. Moderation models (2-4)
4. Lightweight / fast classifiers (3-5)
5. Architecture / checkpoint variants (3-5)
"""

from typing import Dict, List, Optional
from .base import BaseScorer
from .hf_scorer import HuggingFaceScorer

# Master Candidate Registry Metadata
CANDIDATE_REGISTRY: Dict[str, Dict] = {
    # ----------------------------------------------------
    # Family 1: Dedicated Toxicity Classifiers (9 models)
    # ----------------------------------------------------
    "unitary/toxic-bert": {
        "family": "dedicated_toxicity",
        "architecture": "bert-base",
        "description": "Trained on Jigsaw Toxic Comments challenge; multi-label toxicity.",
        "toxic_labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "toxic_indices": [0, 1, 2, 3, 4, 5],
        "aggregation": "max",
        "reference": "Hanu & Unitary (2020)",
    },
    "unitary/unbiased-toxic-roberta": {
        "family": "dedicated_toxicity",
        "architecture": "roberta-base",
        "description": "Unbiased toxic RoBERTa trained on Jigsaw Civil Comments.",
        "toxic_labels": ["toxicity"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Unitary AI",
    },
    "s-nlp/roberta_toxicity_classifier": {
        "family": "dedicated_toxicity",
        "architecture": "roberta-base",
        "description": "Fine-tuned RoBERTa on toxic comment classification.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Skoltech NLP",
    },
    "martin-ha/toxic-comment-model": {
        "family": "dedicated_toxicity",
        "architecture": "distilbert-base",
        "description": "DistilBERT fine-tuned on Jigsaw toxic comments.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Martin Ha",
    },
    "textdetox/minilm-toxicity-classifier": {
        "family": "dedicated_toxicity",
        "architecture": "minilm",
        "description": "Compact MiniLM trained for toxicity detection.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "TextDetox Team",
    },
    "skg/cross-toxicity-detector": {
        "family": "dedicated_toxicity",
        "architecture": "roberta-base",
        "description": "Toxicity detector trained across multiple online corpora.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "SKG Research",
    },
    "citizenlab/distilbert-base-multilingual-cased-toxicity": {
        "family": "dedicated_toxicity",
        "architecture": "distilbert-multilingual",
        "description": "Multilingual toxicity detector on DistilBERT.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Citizen Lab",
    },
    "unitary/multilingual-toxic-xlm-roberta": {
        "family": "dedicated_toxicity",
        "architecture": "xlm-roberta-base",
        "description": "Multilingual toxicity on XLM-RoBERTa.",
        "toxic_labels": ["toxicity"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Unitary AI",
    },
    "google/toxic-comment-model": {
        "family": "dedicated_toxicity",
        "architecture": "bert-base",
        "description": "Reference toxic comments classifier.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Google Research (Proxy)",
    },

    # ----------------------------------------------------
    # Family 2: Hate / Offensive-Language Classifiers (5 models)
    # ----------------------------------------------------
    "facebook/roberta-hate-speech-dynabench-r4-target": {
        "family": "hate_speech",
        "architecture": "roberta-base",
        "description": "Dynabench Round 4 hate speech classifier.",
        "toxic_labels": ["hate"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Vidgen et al. (2021)",
    },
    "cardiffnlp/twitter-roberta-base-offensive": {
        "family": "hate_speech",
        "architecture": "roberta-base",
        "description": "Trained on SemEval 2019 Task 6 offensive language.",
        "toxic_labels": ["offensive"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Barbieri et al. (2020)",
    },
    "cardiffnlp/twitter-roberta-base-hate": {
        "family": "hate_speech",
        "architecture": "roberta-base",
        "description": "Trained on TweetEval hate speech benchmark.",
        "toxic_labels": ["hate"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Barbieri et al. (2020)",
    },
    "tomh/toxigen_roberta": {
        "family": "hate_speech",
        "architecture": "roberta-base",
        "description": "ToxiGen implicit hate and toxic language benchmark checkpoint.",
        "toxic_labels": ["hate"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Hartvigsen et al. (2022)",
    },
    "Hate-speech-CNERG/dehatebert-mono-english": {
        "family": "hate_speech",
        "architecture": "bert-base",
        "description": "DeHateBERT mono-English hate speech classifier.",
        "toxic_labels": ["hate"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Mathew et al. (2021)",
    },

    # ----------------------------------------------------
    # Family 3: Content Moderation & Safety (4 models)
    # ----------------------------------------------------
    "KoalaAI/Text-Moderation": {
        "family": "moderation",
        "architecture": "distilbert-base",
        "description": "Multi-category text moderation model for hate, violence, and self-harm.",
        "toxic_labels": ["hate", "violence", "sexual", "self-harm"],
        "toxic_indices": [1, 2, 3, 4],
        "aggregation": "max",
        "reference": "KoalaAI",
    },
    "mrm8488/distilroberta-finetuned-offensive-language": {
        "family": "moderation",
        "architecture": "distilroberta-base",
        "description": "DistilRoBERTa fine-tuned on offensive content.",
        "toxic_labels": ["offensive"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "MRM Research",
    },
    "narrative-intelligence/safety-moderation-roberta": {
        "family": "moderation",
        "architecture": "roberta-base",
        "description": "Safety moderation classifier.",
        "toxic_labels": ["flagged"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Narrative Intelligence",
    },
    "valhalla/distilbart-mnli-12-3": {
        "family": "moderation",
        "architecture": "distilbart",
        "description": "Zero-shot NLI contradiction / safety risk scoring proxy.",
        "toxic_labels": ["contradiction"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Valhalla",
    },

    # ----------------------------------------------------
    # Family 4: Lightweight / Fast Classifiers (4 models)
    # ----------------------------------------------------
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": {
        "family": "lightweight",
        "architecture": "distilbert-base",
        "description": "High-throughput sentiment/toxicity baseline proxy.",
        "toxic_labels": ["NEGATIVE"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Hugging Face Hub",
    },
    "albert/albert-base-v2": {
        "family": "lightweight",
        "architecture": "albert-base",
        "description": "ALBERT parameter-sharing architecture for fast latency.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Lan et al. (2019)",
    },
    "prajjwal1/bert-tiny": {
        "family": "lightweight",
        "architecture": "bert-tiny",
        "description": "Ultra-lightweight 2-layer BERT for micro-latency checks.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Bhargava (2020)",
    },
    "prajjwal1/bert-mini": {
        "family": "lightweight",
        "architecture": "bert-mini",
        "description": "4-layer Mini BERT for high-speed inference.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Bhargava (2020)",
    },

    # ----------------------------------------------------
    # Family 5: Architecture / Checkpoint Variants (5 models)
    # ----------------------------------------------------
    "GroNLP/hate-speech-models-toxic-bert": {
        "family": "architecture_variants",
        "architecture": "bert-base",
        "description": "Groningen NLP Toxic BERT model.",
        "toxic_labels": ["toxic"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "GroNLP Team",
    },
    "VictorSanh/roberta-base-finetuned-yelp-polarity": {
        "family": "architecture_variants",
        "architecture": "roberta-base",
        "description": "Polarity RoBERTa for negative sentiment boundary checks.",
        "toxic_labels": ["LABEL_0"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Victor Sanh",
    },
    "SamLowe/roberta-base-go_emotions": {
        "family": "architecture_variants",
        "architecture": "roberta-base",
        "description": "GoEmotions multi-label emotion model (anger/disgust proxy).",
        "toxic_labels": ["anger", "disgust"],
        "toxic_indices": [1, 10],
        "aggregation": "max",
        "reference": "Sam Lowe",
    },
    "madhurjindal/autonlp-Gibberish-Detector-492513457": {
        "family": "architecture_variants",
        "architecture": "bert-base",
        "description": "Gibberish & noise detector for degenerate trajectory mitigation.",
        "toxic_labels": ["noise"],
        "toxic_indices": [1],
        "aggregation": "binary",
        "reference": "Madhur Jindal",
    },
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "family": "architecture_variants",
        "architecture": "roberta-base",
        "description": "Latest Twitter-RoBERTa sentiment model for negative sentiment tracking.",
        "toxic_labels": ["negative"],
        "toxic_indices": [0],
        "aggregation": "binary",
        "reference": "Cardiff NLP (2022)",
    },
}


def get_registered_scorer_names() -> List[str]:
    """Return all candidate model names in the registry."""
    return list(CANDIDATE_REGISTRY.keys())


def load_candidate_scorer(model_id: str, device: Optional[str] = None) -> BaseScorer:
    """Instantiate and load a registered scorer."""
    if model_id not in CANDIDATE_REGISTRY:
        # Allow custom model ID fallback
        return HuggingFaceScorer(model_id=model_id, device=device)

    meta = CANDIDATE_REGISTRY[model_id]
    return HuggingFaceScorer(
        model_id=model_id,
        toxic_label_indices=meta.get("toxic_indices"),
        toxic_label_names=meta.get("toxic_labels"),
        aggregation=meta.get("aggregation", "max"),
        device=device,
    )
