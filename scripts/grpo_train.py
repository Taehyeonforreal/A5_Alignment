"""One off-policy GRPO training round: score a rollout batch (group-normalized
advantages), compute old_log_probs once under the current (pre-update) policy,
then take multiple gradient steps over the batch (`epochs_per_rollout_batch`)
using the GRPO-Clip loss.

Off-policy (epochs_per_rollout_batch > 1, train_batch_size < rollout_batch_size)
means the policy drifts away from old_log_probs across steps within the same
round, so the importance ratio pi_theta/pi_old actually moves and clipping has
something real to do (unlike the on-policy case, where old == current always).
"""

import json
from pathlib import Path

import torch
import typer
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.utils import get_response_log_probs, tokenize_prompt_and_output


def load_rollouts(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(
    rollout_path: str = "data/grpo/round_rollouts.jsonl",
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    output_dir: str = "/workspace/grpo_model_out/step1",
    group_size: int = 8,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    learning_rate: float = 1e-5,
    micro_batch_size: int = 1,
    train_batch_size: int = 32,
    epochs_per_rollout_batch: int = 2,
    cliprange: float = 0.2,
):
    rollouts = load_rollouts(rollout_path)
    responses = [r["response"] for r in rollouts]
    ground_truths = [r["ground_truth"] for r in rollouts]

    advantages, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )
    print(f"mean reward: {metadata['mean_reward']:.3f}, std: {metadata['std_reward']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Tokenize the whole rollout batch once so old_log_probs stay exactly
    # aligned (same input_ids/padding) with every later training slice,
    # across however many epochs we take over this batch.
    tokenized = tokenize_prompt_and_output(
        [r["prompt"] for r in rollouts], [r["response"] for r in rollouts], tokenizer
    )
    all_input_ids = tokenized["input_ids"].to("cuda")
    all_labels = tokenized["labels"].to("cuda")
    all_response_mask = tokenized["response_mask"].to("cuda")
    all_advantages = advantages.unsqueeze(-1).to("cuda")

    n = len(rollouts)

    model.eval()
    old_log_probs_chunks = []
    with torch.no_grad():
        for start in range(0, n, micro_batch_size):
            end = start + micro_batch_size
            lp = get_response_log_probs(model, all_input_ids[start:end], all_labels[start:end])["log_probs"]
            old_log_probs_chunks.append(lp)
    all_old_log_probs = torch.cat(old_log_probs_chunks, dim=0)

    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

    gradient_accumulation_steps = train_batch_size // micro_batch_size
    n_clipped, n_total = 0, 0

    for epoch in range(1, epochs_per_rollout_batch + 1):
        for step_start in range(0, n, train_batch_size):
            optimizer.zero_grad()
            step_end = min(step_start + train_batch_size, n)
            for micro_start in range(step_start, step_end, micro_batch_size):
                micro_end = micro_start + micro_batch_size

                log_probs = get_response_log_probs(
                    model, all_input_ids[micro_start:micro_end], all_labels[micro_start:micro_end]
                )["log_probs"]
                loss, step_metadata = grpo_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=all_response_mask[micro_start:micro_end],
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type="grpo_clip",
                    advantages=all_advantages[micro_start:micro_end],
                    old_log_probs=all_old_log_probs[micro_start:micro_end],
                    cliprange=cliprange,
                )
                n_clipped += step_metadata["is_clipped"].sum().item()
                n_total += step_metadata["is_clipped"].numel()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"  epoch {epoch}/{epochs_per_rollout_batch} done")

    clip_fraction = n_clipped / n_total if n_total > 0 else 0.0
    print(f"clip fraction over round: {clip_fraction:.3f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
