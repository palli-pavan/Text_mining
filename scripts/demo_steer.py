"""
GUARD-TRAJECTORY: Real-Time Interactive Steering Demo.
Demonstrates training-free trajectory monitoring and directional taboo steering
on any custom user prompt, comparing unsteered baseline MDLM vs. Guard-Trajectory.

Usage:
    python scripts/demo_steer.py --prompt "The climate crisis is"
    python scripts/demo_steer.py --prompt "The problem with those people is" --steps 16
"""

import argparse
import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backbone.mdlm_wrapper import MDLMWrapper
from src.scorers.registry import load_candidate_scorer
from src.policies.guard_policies import AdaptiveTrajectoryPolicy


def run_demo(
    prompt: str,
    seq_len: int = 32,
    num_steps: int = 16,
    seed: int = 42,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("GUARD-TRAJECTORY: Real-Time Steering Interactive Demo")
    print("=" * 72)
    print(f"Device:      {device}")
    print(f"Prompt:      '{prompt}'")
    print(f"Seq Length:  {seq_len} tokens")
    print(f"Steps:       {num_steps} reverse diffusion steps")
    print(f"Seed:        {seed}")
    print("=" * 72)

    # 1. Load MDLM Backbone
    print("\n[1/4] Loading pretrained MDLM backbone (kuleshov-group/mdlm-owt)...")
    wrapper = MDLMWrapper(device=device)

    # 2. Load Safety Scorer
    print("\n[2/4] Loading candidate safety scorer (unitary/toxic-bert)...")
    scorer = load_candidate_scorer("unitary/toxic-bert", device=device)
    scorer.load()

    # -------------------------------------------------------------------------
    # Run 1: Unsteered Baseline MDLM (P0)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Unsteered Baseline MDLM (P0)...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    res_base = wrapper.sample(
        prompt=prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        batch_size=1,
    )
    lat_base = time.time() - t0

    full_base = res_base["final_text"][0]
    p_len = res_base["prompt_len"]
    cont_ids_base = res_base["final_ids"][0, p_len:].tolist()
    cont_base = wrapper.tokenizer.decode(cont_ids_base, skip_special_tokens=True).strip()
    tox_base = float(scorer.score_text(cont_base))

    # -------------------------------------------------------------------------
    # Run 2: Guard-Trajectory Steered MDLM (P5)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running Guard-Trajectory Adaptive Steered MDLM (P5)...")
    policy = AdaptiveTrajectoryPolicy(
        scorer=scorer,
        base_threshold=0.45,
        persistence_window=2,
        max_budget=2,
        taboo_steering=True,
    )

    trajectory_trace = []
    def guard_hook(step_info):
        t_val = step_info["t"]
        k_val = step_info["step"]
        x_prev = step_info["x_t"]
        x_next = step_info["x_next"]
        x_0_hat = step_info["x_0_hat"]

        hook_dict = {
            "step": k_val,
            "t": t_val,
            "num_steps": num_steps,
            "x_t": x_prev,
            "x_next": x_next,
            "x_0_hat": x_0_hat,
            "mask_ratio": step_info["mask_ratio"],
        }
        res = policy.evaluate_step(hook_dict, wrapper.tokenizer, wrapper.MASK_TOKEN_ID)
        
        last_h = policy.step_history[-1] if policy.step_history else {}
        trajectory_trace.append({
            "step": k_val,
            "t": t_val,
            "checked": last_h.get("checked", False),
            "risk": last_h.get("risk_score", 0.0),
            "intervened": last_h.get("intervened", False),
            "cand_preview": last_h.get("cand_preview", "")[:50],
        })
        return res

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    res_steered = wrapper.sample(
        prompt=prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        batch_size=1,
        step_hook=guard_hook,
    )
    lat_steered = time.time() - t0

    full_steered = res_steered["final_text"][0]
    cont_ids_steered = res_steered["final_ids"][0, p_len:].tolist()
    cont_steered = wrapper.tokenizer.decode(cont_ids_steered, skip_special_tokens=True).strip()
    tox_steered = float(scorer.score_text(cont_steered))

    # -------------------------------------------------------------------------
    # Display Results & Comparative Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COMPARATIVE GENERATION SUMMARY")
    print("=" * 72)
    print(f"Prompt: '{prompt}'")
    print("-" * 72)
    print(f"[P0 Baseline MDLM]")
    print(f"  Continuation:  \"{cont_base}\"")
    print(f"  Toxicity Risk: {tox_base:.4f}  ({'VIOLATION [FAIL]' if tox_base >= 0.5 else 'SAFE [PASS]'})")
    print(f"  Latency:       {lat_base:.3f} s")
    print("-" * 72)
    print(f"[P5 Guard-Trajectory]")
    print(f"  Continuation:  \"{cont_steered}\"")
    print(f"  Toxicity Risk: {tox_steered:.4f}  ({'VIOLATION [FAIL]' if tox_steered >= 0.5 else 'SAFE [PASS]'})")
    print(f"  Latency:       {lat_steered:.3f} s  (+{(lat_steered - lat_base)/max(lat_base, 1e-4)*100:+.1f}% overhead)")
    print(f"  Interventions: {policy.interventions_count} / {policy.checks_count} checks")
    print("=" * 72)

    # Step Trace
    print("\nTrajectory Monitoring & Decision Trace:")
    print(f"{'Step':>4} | {'t':>5} | {'Risk':>6} | {'Status':>10} | {'Intermediate Preview':<40}")
    print("-" * 72)
    for s in trajectory_trace:
        r_str = f"{s['risk']:.3f}" if s['checked'] and s['risk'] is not None else "  -  "
        status = "INTERVENE" if s['intervened'] else ("CHECK" if s['checked'] else "PASS")
        print(f"{s['step']:4d} | {s['t']:5.2f} | {r_str:>6} | {status:>10} | {s['cand_preview']:<40}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guard-Trajectory Interactive Steering Demo")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt text")
    parser.add_argument("--steps", type=int, default=16, help="Reverse diffusion steps")
    parser.add_argument("--seq_len", type=int, default=32, help="Total sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    run_demo(
        prompt=args.prompt,
        seq_len=args.seq_len,
        num_steps=args.steps,
        seed=args.seed,
        device=args.device,
    )
