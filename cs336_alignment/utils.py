from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    prompt_ids_list = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    output_ids_list = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]

    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    seq_lens = [len(ids) for ids in full_ids_list]
    max_len = max(seq_lens)

    batch_size = len(prompt_strs)
    padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    is_response = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (prompt_ids, full_ids) in enumerate(zip(prompt_ids_list, full_ids_list)):
        padded_ids[i, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.long)
        is_response[i, len(prompt_ids) : len(full_ids)] = 1

    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:]
    response_mask = is_response[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1) # log-softmax for numerical stability
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits  # (batch, seq_len, vocab)
    log_probs_full = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_full.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask.to(tensor.dtype)
    return masked.sum(dim=dim) / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)
