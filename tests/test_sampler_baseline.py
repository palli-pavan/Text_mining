"""
Test MDLM baseline sampler, reproducibility, and guard-off equivalence.
Roadmap Item 1-3 verification.
"""

import os
import sys
import torch

# Add workspace to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backbone.mdlm_wrapper import MDLMWrapper


def test_baseline_generation():
    print("\n--- Test 1: MDLM Baseline Generation ---")
    wrapper = MDLMWrapper()
    
    test_prompt = "The climate crisis is"
    seq_len = 32
    num_steps = 16
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    result = wrapper.sample(
        prompt=test_prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        batch_size=1,
    )
    
    generated_text = result["final_text"][0]
    print(f"Prompt: '{test_prompt}'")
    print(f"Generated text: '{generated_text}'")
    print(f"Final IDs shape: {result['final_ids'].shape}")
    assert len(generated_text) > 0, "Generated text should not be empty"
    print("[PASS] Test 1: Baseline generation succeeded.")


def test_guard_off_equivalence():
    print("\n--- Test 2: Guard-Off Equivalence & Reproducibility ---")
    wrapper = MDLMWrapper()
    test_prompt = "Technology has enabled"
    seq_len = 32
    num_steps = 16
    seed = 12345
    
    # Run A: Unguarded baseline
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    res_unguarded = wrapper.sample(
        prompt=test_prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        step_hook=None,
    )
    ids_unguarded = res_unguarded["final_ids"]

    # Run B: Passive monitoring hook (observes x_t and x_0_hat, returns None)
    observed_steps = []
    def passive_monitor_hook(step_info):
        observed_steps.append({
            "step": step_info["step"],
            "t": step_info["t"],
            "mask_ratio": step_info["mask_ratio"],
        })
        return None # No intervention

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    res_monitored = wrapper.sample(
        prompt=test_prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        step_hook=passive_monitor_hook,
    )
    ids_monitored = res_monitored["final_ids"]

    # Exact token equality verification
    exact_match = torch.equal(ids_unguarded, ids_monitored)
    print(f"Observed monitor steps: {len(observed_steps)} / {num_steps}")
    print(f"Unguarded text: '{res_unguarded['final_text'][0]}'")
    print(f"Monitored text: '{res_monitored['final_text'][0]}'")
    print(f"Tokens match bit-for-bit: {exact_match}")
    assert exact_match, "Guard-off mode MUST produce identical tokens to baseline!"
    assert len(observed_steps) == num_steps, "Monitor hook must receive all steps"
    print("[PASS] Test 2: Guard-off equivalence verified.")


if __name__ == "__main__":
    test_baseline_generation()
    test_guard_off_equivalence()
    print("\nAll baseline sampler tests PASSED successfully!")
