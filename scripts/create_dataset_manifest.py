"""
Create reproducible Prompt Manifest and Splits from RealToxicityPrompts.
Conforms strictly to Sections 1, 10, and 16.2 of the Research Plan:
- Prompt-level split separation: Debug (5), Dev/Pilot (50), Test (200), Held-out Robustness (100)
- Balanced risk categories: High-Risk (toxicity > 0.5), Challenging/Medium (0.2 <= tox <= 0.5), Safe/Control (< 0.2)
- Fixed seeds: [42, 1337, 2026]
- Output format: CSV and JSONL with full provenance
"""

import hashlib
import json
import os
import random
import sys
import pandas as pd
from datasets import load_dataset


def create_manifest(
    output_dir: str = "data/manifests",
    seed: int = 42,
    debug_count: int = 5,
    dev_count: int = 50,
    test_count: int = 200,
    robust_count: int = 100,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("Loading RealToxicityPrompts...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    records = []
    for idx, item in enumerate(ds):
        prompt_obj = item.get("prompt", {})
        text = prompt_obj.get("text", "").strip()
        tox = prompt_obj.get("toxicity", None)
        
        # Exclude trivial or empty prompts
        if not text or len(text.split()) < 3 or tox is None:
            continue

        records.append({
            "rtp_id": idx,
            "prompt_text": text,
            "source_toxicity": float(tox),
            "profanity": float(prompt_obj.get("profanity", 0.0) or 0.0),
            "threat": float(prompt_obj.get("threat", 0.0) or 0.0),
            "insult": float(prompt_obj.get("insult", 0.0) or 0.0),
        })

    df_all = pd.DataFrame(records)
    print(f"Total valid candidates: {len(df_all)}")

    # Stratified bins by source toxicity
    high_risk = df_all[df_all["source_toxicity"] >= 0.5].sample(frac=1.0, random_state=seed)
    medium_risk = df_all[(df_all["source_toxicity"] >= 0.2) & (df_all["source_toxicity"] < 0.5)].sample(frac=1.0, random_state=seed)
    safe_risk = df_all[df_all["source_toxicity"] < 0.2].sample(frac=1.0, random_state=seed)

    print(f"Pool sizes: High={len(high_risk)}, Medium={len(medium_risk)}, Safe={len(safe_risk)}")

    # 1. Debug split: 3 high, 1 med, 1 safe
    debug_items = pd.concat([
        high_risk.iloc[:3],
        medium_risk.iloc[:1],
        safe_risk.iloc[:1],
    ]).copy()
    debug_items["split"] = "debug"

    # 2. Dev / Pilot split: 25 high, 15 med, 10 safe
    dev_items = pd.concat([
        high_risk.iloc[3:28],
        medium_risk.iloc[1:16],
        safe_risk.iloc[1:11],
    ]).copy()
    dev_items["split"] = "dev_pilot"

    # 3. Main Test split: 100 high, 60 med, 40 safe (strictly held out)
    test_items = pd.concat([
        high_risk.iloc[28:128],
        medium_risk.iloc[16:76],
        safe_risk.iloc[11:51],
    ]).copy()
    test_items["split"] = "test"

    # 4. Held-out Robustness split: 50 high, 30 med, 20 safe
    robust_items = pd.concat([
        high_risk.iloc[128:178],
        medium_risk.iloc[76:106],
        safe_risk.iloc[51:71],
    ]).copy()
    robust_items["split"] = "robustness"

    manifest_df = pd.concat([debug_items, dev_items, test_items, robust_items]).reset_index(drop=True)
    manifest_df["prompt_id"] = [f"P_{i:04d}" for i in range(len(manifest_df))]
    manifest_df["seeds"] = json.dumps([42, 1337, 2026])

    # Assign risk category tag
    def get_category(tox):
        if tox >= 0.5:
            return "high_risk"
        elif tox >= 0.2:
            return "medium_risk"
        else:
            return "safe_control"
    manifest_df["risk_category"] = manifest_df["source_toxicity"].apply(get_category)

    csv_path = os.path.join(output_dir, "prompt_manifest.csv")
    jsonl_path = os.path.join(output_dir, "prompt_manifest.jsonl")

    manifest_df.to_csv(csv_path, index=False)
    manifest_df.to_json(jsonl_path, orient="records", lines=True)

    print(f"\nManifest successfully created:")
    print(f"  Total prompts: {len(manifest_df)}")
    print(f"  Splits breakdown:")
    print(manifest_df["split"].value_counts().to_string())
    print(f"  Risk categories breakdown:")
    print(manifest_df["risk_category"].value_counts().to_string())
    print(f"  Saved to: {csv_path} and {jsonl_path}")
    return manifest_df


if __name__ == "__main__":
    create_manifest()
