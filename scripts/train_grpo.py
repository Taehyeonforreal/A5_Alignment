"""Off-policy GRPO orchestrator (PDF Algorithm 3, off-policy configuration).

Alternates, each as a fresh subprocess so GPU memory is fully released between
vLLM and HF Transformers use:
  1. scripts/grpo_generate.py — sample a rollout batch with the current policy.
  2. scripts/grpo_train.py — multiple GRPO-Clip gradient steps over that batch
     (epochs_per_rollout_batch > 1), using old_log_probs frozen at the start
     of the round.
The output checkpoint of each round becomes the starting policy for the next.
"""

import shutil
import subprocess
import sys

import typer


def main(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    gsm8k_path: str = "data/gsm8k/train.jsonl",
    output_dir: str = "/workspace/grpo_model_out",
    n_grpo_steps: int = 10,
    n_prompts_per_step: int = 16,
    group_size: int = 8,
    max_gen_tokens: int = 512,
    micro_batch_size: int = 1,
    train_batch_size: int = 32,
    epochs_per_rollout_batch: int = 2,
    cliprange: float = 0.2,
    learning_rate: float = 1e-5,
    normalize_by_std: bool = True,
):
    current_checkpoint = model_id
    starting_checkpoint = model_id  # never delete: not a directory we created

    for grpo_step in range(1, n_grpo_steps + 1):
        print(f"\n=== GRPO step {grpo_step}/{n_grpo_steps} (policy: {current_checkpoint}) ===")
        rollout_path = f"data/grpo/step{grpo_step}_rollouts.jsonl"
        step_output_dir = f"{output_dir}/step{grpo_step}"

        subprocess.run(
            [
                sys.executable,
                "scripts/grpo_generate.py",
                "--model-path", current_checkpoint,
                "--gsm8k-path", gsm8k_path,
                "--output-path", rollout_path,
                "--n-prompts", str(n_prompts_per_step),
                "--group-size", str(group_size),
                "--max-tokens", str(max_gen_tokens),
                "--seed", str(grpo_step),
            ],
            check=True,
        )

        train_args = [
            sys.executable,
            "scripts/grpo_train.py",
            "--rollout-path", rollout_path,
            "--model-id", current_checkpoint,
            "--output-dir", step_output_dir,
            "--group-size", str(group_size),
            "--micro-batch-size", str(micro_batch_size),
            "--train-batch-size", str(train_batch_size),
            "--epochs-per-rollout-batch", str(epochs_per_rollout_batch),
            "--cliprange", str(cliprange),
            "--learning-rate", str(learning_rate),
            "--normalize-by-std" if normalize_by_std else "--no-normalize-by-std",
        ]
        subprocess.run(train_args, check=True)

        if current_checkpoint != starting_checkpoint:
            shutil.rmtree(current_checkpoint, ignore_errors=True)
        current_checkpoint = step_output_dir
        print(f"=== GRPO step {grpo_step} done -> {current_checkpoint} ===")

    print(f"\nFinal GRPO model: {current_checkpoint}")


if __name__ == "__main__":
    typer.run(main)
