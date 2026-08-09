"""Expert Iteration orchestrator (PDF Algorithm 2).

Alternates, each as a fresh subprocess so GPU memory is fully released between
vLLM and HF Transformers use:
  1. scripts/ei_generate.py — sample rollouts with the current policy, filter
     to verified-correct ones.
  2. scripts/train_sft.py — SFT the policy on those filtered examples.
The output checkpoint of each round becomes the starting policy for the next.
"""

import subprocess
import sys

import typer


def main(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    gsm8k_path: str = "data/gsm8k/train.jsonl",
    output_dir: str = "/workspace/ei_model_out",
    n_ei_steps: int = 3,
    n_questions_per_step: int = 64,
    group_size: int = 4,
    max_gen_tokens: int = 512,
    sft_steps: int = 20,
    sft_micro_batch_size: int = 1,
    sft_gradient_accumulation_steps: int = 8,
    sft_learning_rate: float = 2e-5,
):
    current_checkpoint = model_id

    for ei_step in range(1, n_ei_steps + 1):
        print(f"\n=== EI step {ei_step}/{n_ei_steps} (policy: {current_checkpoint}) ===")
        round_data_path = f"data/ei/round{ei_step}_sft.jsonl"
        step_output_dir = f"{output_dir}/step{ei_step}"

        subprocess.run(
            [
                sys.executable,
                "scripts/ei_generate.py",
                "--model-path", current_checkpoint,
                "--gsm8k-path", gsm8k_path,
                "--output-path", round_data_path,
                "--n-questions", str(n_questions_per_step),
                "--group-size", str(group_size),
                "--max-tokens", str(max_gen_tokens),
                "--seed", str(ei_step),
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/train_sft.py",
                "--data-path", round_data_path,
                "--model-id", current_checkpoint,
                "--output-dir", step_output_dir,
                "--n-val", "0",
                "--learning-rate", str(sft_learning_rate),
                "--micro-batch-size", str(sft_micro_batch_size),
                "--gradient-accumulation-steps", str(sft_gradient_accumulation_steps),
                "--n-steps", str(sft_steps),
            ],
            check=True,
        )

        current_checkpoint = step_output_dir
        print(f"=== EI step {ei_step} done -> {current_checkpoint} ===")

    print(f"\nFinal EI model: {current_checkpoint}")


if __name__ == "__main__":
    typer.run(main)
