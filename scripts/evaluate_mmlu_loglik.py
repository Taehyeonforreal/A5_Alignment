"""Log-likelihood MMLU scoring -- bypasses generative parsing entirely.

scripts/evaluate_mmlu_gsm8k.py lets the model generate freely, then parses the
answer letter out of the text with a regex. That two-stage pipeline conflates
"does the model know the answer" with "does the model phrase it the way the
parser expects" -- roughly half of the tuned model's apparent MMLU regression
turned out to be exactly this parsing artifact, not real capability loss.

Here there is no generation step at all: for each question we teacher-force
each candidate letter (" A"/" B"/" C"/" D") right after a bare MMLU prompt
(no system-prompt/Alpaca wrapper -- those end in a code fence or "### Response:",
not directly in "Answer:", so the very next token isn't the answer position
any more; log-likelihood scoring doesn't need a format-eliciting wrapper at
all, it just needs the prompt to end exactly where the answer goes), read out
log P(candidate | prompt) from a single forward pass, and take the argmax.
This is the standard MMLU protocol (Hendrycks et al., 2021; lm-evaluation-harness)
and the same teacher-forcing mechanism as cs336_alignment.utils.get_response_log_probs,
just applied to 4 one-token candidates instead of a full response.
"""

import json

import torch
import torch.nn.functional as F
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.metrics import build_mmlu_bare_prompt, load_mmlu_examples

_CANDIDATES = ["A", "B", "C", "D"]


def score_candidates(model, input_ids: torch.Tensor, candidate_token_ids: list[int]) -> list[float]:
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]  # next-token logits after the prompt
    log_probs = F.log_softmax(logits, dim=-1)
    return [log_probs[t].item() for t in candidate_token_ids]


def main(
    model_path: str = "meta-llama/Llama-3.1-8B",
    mmlu_dir: str = "data/mmlu/test",
    n_mmlu_examples: int = 1000,
    seed: int = 0,
    dump_path: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    model.eval()

    # Llama's BPE tokenizes " A"/" B"/" C"/" D" (leading space, matching how the
    # model would naturally continue after "Answer:") as single tokens each.
    candidate_token_ids = [tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in _CANDIDATES]

    examples = load_mmlu_examples(mmlu_dir, max_examples=n_mmlu_examples or None, seed=seed)

    n_correct = 0
    dump_records = []
    for ex in examples:
        prompt = build_mmlu_bare_prompt(ex)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        log_probs = score_candidates(model, input_ids, candidate_token_ids)
        prediction = _CANDIDATES[max(range(4), key=lambda i: log_probs[i])]

        correct = prediction == ex["answer"]
        n_correct += correct
        if dump_path and not correct:
            dump_records.append(
                {
                    "subject": ex["subject"],
                    "question": ex["question"],
                    "gold": ex["answer"],
                    "pred": prediction,
                    "log_probs": dict(zip(_CANDIDATES, log_probs)),
                }
            )

    n = len(examples)
    print(f"MMLU (log-likelihood): {n_correct}/{n} correct ({100 * n_correct / n:.1f}%)")

    if dump_path:
        with open(dump_path, "w") as f:
            for record in dump_records:
                f.write(json.dumps(record) + "\n")
        print(f"  wrote {len(dump_records)} incorrect MMLU examples to {dump_path}")


if __name__ == "__main__":
    typer.run(main)
