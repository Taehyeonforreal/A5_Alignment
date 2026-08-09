"""Reduced-scale SFT training on data/openr1_math/sft.jsonl.

Trains with plain HuggingFace Transformers (no vLLM) and just tracks training
loss. Evaluation (zero-shot baseline and post-SFT accuracy) is a separate step
— see scripts/evaluate_math.py.
"""

import json
import random
from pathlib import Path

import torch
import typer
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.sft import sft_microbatch_train_step
from cs336_alignment.utils import get_response_log_probs, tokenize_prompt_and_output


def load_examples(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(
    data_path: str = "data/openr1_math/sft.jsonl",
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    output_dir: str = "/workspace/sft_model_out",
    n_val: int = 40,
    learning_rate: float = 2e-5,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    n_steps: int = 100,
    seed: int = 0,
):
    random.seed(seed)
    examples = load_examples(data_path)
    random.shuffle(examples)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    # Save the val split so evaluate_math.py can use the exact same held-out set.
    val_path = Path(data_path).parent / "sft_val.jsonl"
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Held out {len(val_examples)} validation examples -> {val_path}")
    print(f"Training on {len(train_examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable() # recomputation 진짜 썻아. 
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    data_idx = 0
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss_value = None
        for _ in range(gradient_accumulation_steps):
            if data_idx + micro_batch_size > len(train_examples):
                data_idx = 0
                random.shuffle(train_examples)
            batch = train_examples[data_idx : data_idx + micro_batch_size]
            data_idx += micro_batch_size

            tokenized = tokenize_prompt_and_output( # micro-batch 한 step.
                [ex["prompt"] for ex in batch], [ex["response"] for ex in batch], tokenizer
            )
            input_ids = tokenized["input_ids"].to("cuda")
            labels = tokenized["labels"].to("cuda")
            response_mask = tokenized["response_mask"].to("cuda")

            log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps)
            loss_value = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #accumulate step 이 끝나고 다 누적하고 실행.
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"step {step}/{n_steps}: loss {loss_value:.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
