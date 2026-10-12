"""
MDLM Backbone Wrapper and Trajectory-Aware Sampler.
Implements Masked Discrete Diffusion Language Model sampling with support for:
- Prefix conditioning (prompt continuation)
- Exposing intermediate trajectory states (x_t, predicted clean tokens x_0_hat, logits/probabilities)
- Trajectory monitoring and intervention hooks
- Guard-off equivalence guarantees
"""

import math
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel


def apply_transformers_compat_patches():
    """Apply compatibility patches for transformers >= 4.40 / 5.x."""
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


class MDLMWrapper:
    """Wrapper around kuleshov-group/mdlm-owt discrete diffusion model."""

    MASK_TOKEN_ID = 50257
    BOS_TOKEN_ID = 50256
    VOCAB_SIZE = 50258

    def __init__(
        self,
        model_name: str = "kuleshov-group/mdlm-owt",
        tokenizer_name: str = "gpt2",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        apply_transformers_compat_patches()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure padding / mask token conventions
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[MDLMWrapper] Loading {model_name} onto {self.device}...")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self._patch_rotary_emb()
        self.model.eval().to(self.device)
        print(f"[MDLMWrapper] Ready. Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _patch_rotary_emb(self):
        """Fix rotary positional embedding tensor dimensions for PyTorch / transformers."""
        mdlm_mod_name = self.model.__class__.__module__
        mod = sys.modules.get(mdlm_mod_name)
        if mod is None:
            return

        def fixed_apply_rotary_pos_emb(qkv, cos, sin):
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]
            cos_sliced = cos[:, :, 0, :, :]
            sin_sliced = sin[:, :, 0, :, :]
            q = q * cos_sliced + mod.rotate_half(q) * sin_sliced
            k = k * cos_sliced + mod.rotate_half(k) * sin_sliced
            return torch.stack((q, k, v), dim=2)

        mod.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb

    def get_logits(self, input_ids: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through MDLM backbone."""
        if timesteps is None:
            timesteps = torch.zeros(input_ids.shape[0], device=self.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).expand(input_ids.shape[0])

        out = self.model(input_ids=input_ids, timesteps=timesteps)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        return logits

    @torch.no_grad()
    def sample(
        self,
        prompt: Optional[str] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        seq_len: int = 64,
        num_steps: int = 64,
        batch_size: int = 1,
        eps: float = 1e-3,
        step_hook: Optional[Callable[[Dict], Optional[torch.Tensor]]] = None,
        temperature: float = 1.0,
    ) -> Dict:
        """
        Discrete reverse diffusion sampling (DDPM update for absorbing state).

        Args:
            prompt: Optional text prompt to continue.
            prompt_ids: Optional pre-tokenized prompt tensor [B, L_p].
            seq_len: Total length of sequence (prompt + continuation).
            num_steps: Number of reverse diffusion steps (T).
            batch_size: Number of sequences to sample.
            eps: Noise schedule lower bound.
            step_hook: Callback invoked at each step t:
                       hook(dict(step=i, t=t_val, x_t=x, x_0_hat=x0_hat, logits=logits, probs=probs))
                       If hook returns a modified tensor, it overrides x_next or logits.
            temperature: Sampling temperature for categorical sampling.

        Returns:
            Dictionary containing:
                final_ids: generated token IDs [B, seq_len]
                final_text: decoded text list
                prompt_len: length of conditioned prompt
                trajectories: list of step diagnostics if logged
        """
        # 1. Setup prompt conditioning
        prompt_mask = None
        if prompt is not None:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            prompt_ids = encoded.input_ids.to(self.device)

        if prompt_ids is not None:
            prompt_ids = prompt_ids.to(self.device)
            # Ensure prompt leaves sufficient room for continuation (at least 8 tokens)
            if prompt_ids.shape[1] >= seq_len:
                max_prompt_len = max(seq_len - 8, seq_len // 2)
                prompt_ids = prompt_ids[:, :max_prompt_len]
            prompt_len = prompt_ids.shape[1]
            
            x = torch.full((batch_size, seq_len), self.MASK_TOKEN_ID, dtype=torch.long, device=self.device)
            x[:, :prompt_len] = prompt_ids
            # prompt mask: True where tokens are fixed prompt tokens
            prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
            prompt_mask[:, :prompt_len] = True
        else:
            prompt_len = 0
            x = torch.full((batch_size, seq_len), self.MASK_TOKEN_ID, dtype=torch.long, device=self.device)
            prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)

        # 2. Setup schedule
        # Time t runs from 1.0 down to eps in num_steps steps
        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=self.device)
        dt = (1.0 - eps) / num_steps

        trajectory_logs = []
        active_taboo_ids = set()

        # 3. Reverse diffusion loop
        for step_idx in range(num_steps):
            t_curr = timesteps[step_idx]
            s_next = timesteps[step_idx + 1]

            # Linear log-linear move chance for absorbing state:
            # move_chance_t = (1 - eps) * t
            # move_chance_s = (1 - eps) * s
            move_chance_t = (1.0 - eps) * t_curr
            move_chance_s = (1.0 - eps) * s_next

            # Forward pass: predict clean distribution x_0
            t_tensor = t_curr.unsqueeze(0).expand(batch_size)
            logits = self.get_logits(x, timesteps=t_tensor) # [B, L, V]

            # Temperature scaling if desired
            if temperature != 1.0 and temperature > 0:
                scaled_logits = logits / temperature
            else:
                scaled_logits = logits

            probs = F.softmax(scaled_logits, dim=-1) # [B, L, V]

            # Predicted clean tokens (x_0_hat)
            x_0_hat = probs.argmax(dim=-1)
            # Prompt tokens remain clamped in x_0_hat
            if prompt_len > 0:
                x_0_hat[:, :prompt_len] = prompt_ids

            # Compute transition probability for unmasking
            delta_move = max(float(move_chance_t - move_chance_s), 1e-8)
            q_xs = probs * delta_move # [B, L, V]

            # Apply directional taboo constraints if active
            if active_taboo_ids:
                taboo_tensor = torch.tensor(list(active_taboo_ids), dtype=torch.long, device=self.device)
                q_xs[:, :, taboo_tensor] = 0.0

            q_xs[:, :, self.MASK_TOKEN_ID] = float(move_chance_s)

            # Sample categorical update for masked positions
            # Flatten to sample
            flat_probs = q_xs.reshape(-1, self.VOCAB_SIZE)
            sampled_tokens = torch.multinomial(flat_probs, num_samples=1).reshape(batch_size, seq_len)

            # Absorbing state property: once unmasked, tokens never revert to MASK
            already_unmasked = (x != self.MASK_TOKEN_ID)
            x_next = torch.where(already_unmasked, x, sampled_tokens)

            # Always ensure prompt remains fixed
            if prompt_len > 0:
                x_next[:, :prompt_len] = prompt_ids

            # Trajectory hook (for monitoring and intervention)
            if step_hook is not None:
                step_info = {
                    "step": step_idx,
                    "num_steps": num_steps,
                    "t": float(t_curr),
                    "x_t": x.clone(),
                    "x_0_hat": x_0_hat.clone(),
                    "x_next": x_next.clone(),
                    "probs": probs,
                    "prompt_len": prompt_len,
                    "mask_ratio": float((x == self.MASK_TOKEN_ID).float().mean()),
                }
                override = step_hook(step_info)
                if override is not None:
                    # Hook can return a Tensor or dict containing x_next and add_taboo_ids
                    if isinstance(override, dict):
                        if "x_next" in override:
                            x_next = override["x_next"]
                        if "add_taboo_ids" in override and override["add_taboo_ids"]:
                            active_taboo_ids.update(override["add_taboo_ids"])
                    elif isinstance(override, torch.Tensor):
                        x_next = override

                    if prompt_len > 0:
                        x_next[:, :prompt_len] = prompt_ids

            x = x_next

        # 4. Final step noise removal (denoising any remaining mask tokens)
        final_masks = (x == self.MASK_TOKEN_ID)
        if final_masks.any():
            final_logits = self.get_logits(x, timesteps=torch.zeros(batch_size, device=self.device))
            if active_taboo_ids:
                taboo_tensor = torch.tensor(list(active_taboo_ids), dtype=torch.long, device=self.device)
                final_logits[:, :, taboo_tensor] = -1e9
            clean_preds = final_logits.argmax(dim=-1)
            x = torch.where(final_masks, clean_preds, x)
            if prompt_len > 0:
                x[:, :prompt_len] = prompt_ids

        # 5. Decode generated text
        decoded_texts = []
        decoded_prompts = []
        decoded_conts = []
        for i in range(batch_size):
            text = self.tokenizer.decode(x[i].tolist(), skip_special_tokens=True)
            decoded_texts.append(text)
            p_text = self.tokenizer.decode(x[i, :prompt_len].tolist(), skip_special_tokens=True) if prompt_len > 0 else ""
            c_text = self.tokenizer.decode(x[i, prompt_len:].tolist(), skip_special_tokens=True) if prompt_len > 0 else text
            decoded_prompts.append(p_text)
            decoded_conts.append(c_text)

        return {
            "final_ids": x,
            "final_text": decoded_texts,
            "prompt_text": decoded_prompts,
            "continuation_text": decoded_conts,
            "prompt_len": prompt_len,
            "seq_len": seq_len,
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token ids."""
        texts = []
        for row in token_ids:
            # Replace mask token if decoding intermediate
            clean_ids = [tok for tok in row.tolist() if tok != self.MASK_TOKEN_ID]
            texts.append(self.tokenizer.decode(clean_ids, skip_special_tokens=skip_special_tokens))
        return texts
