"""Instruction tuning: full fine-tune of Llama 3.1 8B base on packed SFT data.

Unlike the main assignment's train_sft.py (masked loss on response tokens,
Qwen2.5-Math-1.5B), this trains next-token-prediction loss over an entire
packed token stream (no response_mask) on an 8B model -- full fine-tuning at
this scale needs an 80GB GPU even with an 8-bit optimizer.

Recommended recipe per the assignment PDF: single epoch, seq_length=512,
effective batch size 32 (via gradient accumulation), lr=2e-5 with cosine decay
and 3% linear warmup. We run this over a subsampled dataset (see
scripts/prepare_instruction_tuning_data.py) rather than the full ~210k-example
epoch (~24 H100 hrs in the original recipe) -- reduced-scale by design.
"""

import math
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from cs336_alignment.packing import PackedSFTDataset, iterate_batches


def compute_batch_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"].to("cuda")
    labels = batch["labels"].to("cuda")
    logits = model(input_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def main(
    train_path: str = "data/instruction_tuning/train.jsonl",
    val_path: str = "data/instruction_tuning/val.jsonl",
    model_id: str = "meta-llama/Llama-3.1-8B",
    output_dir: str = "/workspace/instruction_tuned_model_out",
    seq_length: int = 512,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,  # effective batch size = 32
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.03,
    log_every: int = 10,
    seed: int = 0,
):
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_dataset = PackedSFTDataset(tokenizer, train_path, seq_length=seq_length, shuffle=True)
    val_dataset = PackedSFTDataset(tokenizer, val_path, seq_length=seq_length, shuffle=False)
    print(f"train: {len(train_dataset)} packed sequences, val: {len(val_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)

    train_batches = iterate_batches(train_dataset, batch_size=micro_batch_size, shuffle=True)
    n_steps = len(train_batches) // gradient_accumulation_steps
    n_warmup_steps = max(1, math.ceil(n_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=n_steps)
    print(f"{len(train_batches)} micro-batches -> {n_steps} optimizer steps ({n_warmup_steps} warmup)")

    micro_idx = 0
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss_sum = 0.0
        for _ in range(gradient_accumulation_steps):
            loss = compute_batch_loss(model, train_batches[micro_idx]) / gradient_accumulation_steps
            loss.backward()
            loss_sum += loss.item()
            micro_idx += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step}/{n_steps}: train loss {loss_sum:.4f}, lr {scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    val_batches = iterate_batches(val_dataset, batch_size=micro_batch_size, shuffle=False)
    val_losses = []
    with torch.no_grad():
        for batch in val_batches:
            val_losses.append(compute_batch_loss(model, batch).item())
    print(f"final val loss: {sum(val_losses) / len(val_losses):.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
