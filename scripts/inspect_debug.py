import json
import pandas as pd

with open('results/experiment_runs_debug.jsonl', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

print(f"Total runs: {len(records)}")

df = pd.DataFrame(records)
print("\n--- Summary by Policy ---")
print(df.groupby('policy')[['final_toxicity', 'is_violation', 'checks_count', 'interventions_count', 'latency_sec']].mean())

print("\n--- Per-Prompt Details ---")
for pid in df['prompt_id'].unique():
    sub = df[df['prompt_id'] == pid]
    print(f"\nPrompt ID: {pid} (Cat: {sub.iloc[0]['risk_category']})")
    print(f"Prompt text: '{sub.iloc[0]['prompt_text']}'")
    for _, row in sub.iterrows():
        print(f"  {row['policy']:2s}: tox={row['final_toxicity']:.3f} | viol={row['is_violation']} | interv={row['interventions_count']} | text='{row['final_text'][:80]}...'")
