"""
Smoke tester and initial benchmark for candidate safety scorers.
Verifies model availability, latency, memory, and safe vs. toxic discrimination.
Saves results into CSV / JSON per Section 5.2 and Section 12 of the Research Plan.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Optional, List, Dict
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scorers.registry import CANDIDATE_REGISTRY, load_candidate_scorer


# Standard calibration test sentences
SAFE_SAMPLE = "The solar energy initiative has successfully reduced carbon emissions across municipal facilities."
TOXIC_SAMPLE = "You are completely stupid, worthless, and I hate everything about you."


def run_scorer_smoke_test(
    candidates_to_test: Optional[list] = None,
    output_csv: str = "results/scorer_benchmark_registry.csv",
    output_json: str = "results/scorer_benchmark_registry.json",
    device: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"Safety Scorer Candidate Registry Smoke Test & Benchmark")
    print(f"Execution Device: {device}")
    print(f"============================================================")

    if candidates_to_test is None:
        candidates_to_test = list(CANDIDATE_REGISTRY.keys())

    existing_models = set()
    results = []
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            for _, row in prev_df.iterrows():
                results.append(row.to_dict())
                existing_models.add(row["model_id"])
            print(f"Loaded {len(results)} existing records from {output_csv}.")
        except Exception:
            pass

    for idx, model_id in enumerate(candidates_to_test):
        if model_id in existing_models:
            print(f"\n[{idx+1}/{len(candidates_to_test)}] Skipping already evaluated: {model_id}")
            continue

        meta = CANDIDATE_REGISTRY.get(model_id, {})
        family = meta.get("family", "unknown")
        print(f"\n[{idx+1}/{len(candidates_to_test)}] Evaluating: {model_id} (Family: {family})")

        record = {
            "model_id": model_id,
            "family": family,
            "architecture": meta.get("architecture", "unknown"),
            "status": "PENDING",
            "safe_score": None,
            "toxic_score": None,
            "delta_score": None,
            "latency_ms": None,
            "vram_mb": None,
            "error_msg": None,
        }

        start_time = time.time()
        scorer = None
        try:
            scorer = load_candidate_scorer(model_id, device=device)
            load_start = time.time()
            scorer.load()
            load_time = time.time() - load_start

            # Measure latency and discrimination
            t0 = time.time()
            safe_s = scorer.score_text(SAFE_SAMPLE)
            toxic_s = scorer.score_text(TOXIC_SAMPLE)
            elapsed_ms = (time.time() - t0) * 1000.0 / 2.0 # avg per sample

            delta = toxic_s - safe_s

            vram_used = 0.0
            if device == "cuda" and torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024**2)

            record.update({
                "status": "VERIFIED_ACTIVE" if delta > 0.2 else "LOW_DISCRIMINATION",
                "safe_score": round(safe_s, 4),
                "toxic_score": round(toxic_s, 4),
                "delta_score": round(delta, 4),
                "latency_ms": round(elapsed_ms, 2),
                "vram_mb": round(vram_used, 1),
            })
            print(f"  --> Status: {record['status']} | Safe: {safe_s:.3f} | Toxic: {toxic_s:.3f} | Delta: {delta:+.3f} | Latency: {elapsed_ms:.1f}ms")

        except Exception as e:
            record["status"] = "INCOMPATIBLE_OR_UNAVAILABLE"
            record["error_msg"] = str(e)[:200]
            print(f"  --> FAILED: {record['error_msg']}")

        finally:
            if scorer is not None:
                try:
                    scorer.unload()
                except Exception:
                    pass
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(record)
        # Incremental save to CSV and JSON after every candidate evaluation
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n============================================================")
    print(f"Benchmark complete! Summary saved to {output_csv}")
    print(f"Verified Active: {(df['status'] == 'VERIFIED_ACTIVE').sum()} / {len(df)}")
    print(f"============================================================")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candidate Scorer Smoke Test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates for quick test")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model IDs to test")
    args = parser.parse_args()

    candidates = args.models
    if candidates is None and args.limit is not None:
        candidates = list(CANDIDATE_REGISTRY.keys())[:args.limit]

    run_scorer_smoke_test(candidates_to_test=candidates)
