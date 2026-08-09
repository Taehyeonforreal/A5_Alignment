from __future__ import annotations

import torch

from cs336_alignment.utils import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    nll_per_example = -masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant, dim=-1)
    loss = nll_per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"nll_per_example": nll_per_example.detach()}
