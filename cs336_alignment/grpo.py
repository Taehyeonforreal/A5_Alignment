from __future__ import annotations

from typing import Callable, Literal

import torch

from cs336_alignment.utils import masked_mean


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    rewards_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        result_dict = reward_fn(resp, gt)
        reward_value = result_dict["reward"]
        rewards_list.append(reward_value)

    raw_rewards = torch.tensor(rewards_list, dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_std = grouped.std(dim=1, keepdim=True)
        advantages = (grouped - group_mean) / (group_std + advantage_eps)
    else:
        advantages = grouped - group_mean

    advantages = advantages.reshape(-1)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_obj = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    clipped_obj = clipped_ratio * advantages

    per_token_loss = -torch.minimum(unclipped_obj, clipped_obj)
    metadata = {"is_clipped": (clipped_obj < unclipped_obj)}
    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline requires raw_rewards"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline requires advantages"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    if loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip requires advantages"
        assert old_log_probs is not None, "grpo_clip requires old_log_probs"
        assert cliprange is not None, "grpo_clip requires cliprange"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=-1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
