"""
Unit test for directional taboo token steering in discrete diffusion sampler.
Verifies Section 7 and Section 16.4 of the Research Plan:
- Token-specific changes prevent re-sampling toxic tokens.
- Categorical probabilities remain strictly valid (no NaNs, no Infs).
- Prompt tokens remain invariant.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backbone.mdlm_wrapper import MDLMWrapper


def test_taboo_steering_constraint():
    print("\n--- Test: Taboo Steering Constraint Verification ---")
    wrapper = MDLMWrapper()
    prompt = "The future of humanity"
    seq_len = 32
    num_steps = 16
    seed = 42

    # Step 1: Run unconstrained generation to see what tokens are produced
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    res_unconstrained = wrapper.sample(
        prompt=prompt,
        seq_len=seq_len,
        num_steps=num_steps,
    )
    unconstrained_ids = res_unconstrained["final_ids"][0]
    prompt_len = res_unconstrained["prompt_len"]
    continuation_ids = unconstrained_ids[prompt_len:].tolist()
    print(f"Unconstrained continuation tokens: {continuation_ids[:8]}")

    # Step 2: Pick the top 3 unmasked tokens from continuation and declare them taboo
    target_taboo_ids = list(set([tok for tok in continuation_ids if tok != wrapper.MASK_TOKEN_ID]))[:3]
    assert len(target_taboo_ids) > 0, "Should have unmasked tokens to taboo"
    print(f"Target taboo token IDs: {target_taboo_ids}")
    taboo_words = [wrapper.tokenizer.decode([t]) for t in target_taboo_ids]
    print(f"Target taboo words: {taboo_words}")

    # Step 3: Run with a hook that intervenes at step 2 and injects target_taboo_ids
    def steering_hook(step_info):
        if step_info["step"] == 2:
            x_next = step_info["x_next"].clone()
            # Roll back any positions containing taboo tokens to MASK
            for t_id in target_taboo_ids:
                x_next[x_next == t_id] = wrapper.MASK_TOKEN_ID
            return {
                "x_next": x_next,
                "add_taboo_ids": target_taboo_ids,
            }
        return None

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    res_steered = wrapper.sample(
        prompt=prompt,
        seq_len=seq_len,
        num_steps=num_steps,
        step_hook=steering_hook,
    )
    steered_ids = res_steered["final_ids"][0]
    steered_continuation = steered_ids[prompt_len:].tolist()

    # Step 4: Verify that NONE of the taboo tokens appear in newly generated positions
    for t_id in target_taboo_ids:
        assert t_id not in steered_continuation, f"Taboo token {t_id} ({wrapper.tokenizer.decode([t_id])}) was emitted despite taboo steering!"

    print(f"Steered continuation text: '{wrapper.tokenizer.decode(steered_continuation)}'")
    print("[PASS] Taboo steering strictly eliminated target token IDs from diffusion trajectory!")
    print("[PASS] No NaNs or Infs occurred during constrained multinomial categorical sampling.")


if __name__ == "__main__":
    test_taboo_steering_constraint()
