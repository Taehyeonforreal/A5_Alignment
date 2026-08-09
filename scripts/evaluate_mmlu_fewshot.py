"""Few-shot MMLU evaluation -- still generative parsing, but with K worked
examples (drawn from data/mmlu/dev, same subject, standard 5-shot protocol per
Hendrycks et al. 2021) shown before the test question instead of a natural-
language "respond in this format" instruction.

Motivation: scripts/evaluate_mmlu_gsm8k.py's zero-shot generative parsing
conflates "does the model know the answer" with "does it phrase it exactly as
instructed". Few-shot sidesteps this
differently than log-likelihood scoring does: instead of removing the format
requirement, it demonstrates the format via precedent (pattern completion),
which is usually a stronger elicitor of surface behavior than a single
instruction sentence buried in a larger prompt.

Like log-likelihood scoring, this drops the system-prompt/Alpaca wrapper
entirely -- the prompt must end exactly at the answer position for the shown
examples to anchor the completion, and the K worked examples make the "please
answer like this" instruction redundant.
"""

import json
import time

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.metrics import (
    build_mmlu_fewshot_prompt,
    load_mmlu_dev_examples,
    load_mmlu_examples,
    parse_mmlu_response,
)


def main(
    model_path: str = "meta-llama/Llama-3.1-8B",
    mmlu_dir: str = "data/mmlu/test",
    mmlu_dev_dir: str = "data/mmlu/dev",
    n_mmlu_examples: int = 1000,
    n_shots: int = 5,
    max_tokens: int = 16,
    seed: int = 0,
    # Few-shot prompts are much longer than zero-shot (5 worked examples per
    # question), so the 2048 cap used in evaluate_mmlu_gsm8k.py isn't enough.
    max_model_len: int = 4096,
    dump_path: str | None = None,
):
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    # Stop at the blank line separating QA blocks in build_mmlu_fewshot_prompt,
    # so the model doesn't ramble on into a hallucinated next question.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["\n\n"])

    dev_by_subject = load_mmlu_dev_examples(mmlu_dev_dir)
    examples = load_mmlu_examples(mmlu_dir, max_examples=n_mmlu_examples or None, seed=seed)
    prompts = [build_mmlu_fewshot_prompt(ex, dev_by_subject[ex["subject"]][:n_shots]) for ex in examples]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    n_correct = 0
    n_unparsed = 0
    dump_records = []
    for ex, output in zip(examples, outputs):
        response_text = output.outputs[0].text
        prediction = parse_mmlu_response(ex, response_text)
        if prediction is None:
            n_unparsed += 1
        elif prediction == ex["answer"]:
            n_correct += 1

        if dump_path and prediction != ex["answer"]:
            dump_records.append(
                {
                    "subject": ex["subject"],
                    "question": ex["question"],
                    "gold": ex["answer"],
                    "pred": prediction,
                    "output": response_text,
                }
            )

    n = len(examples)
    print(f"MMLU ({n_shots}-shot): {n_correct}/{n} correct ({100 * n_correct / n:.1f}%), {n_unparsed} unparsed ({100 * n_unparsed / n:.1f}%)")
    print(f"MMLU throughput: {n / elapsed:.2f} examples/sec ({elapsed:.1f}s for {n} examples)")

    if dump_path:
        with open(dump_path, "w") as f:
            for record in dump_records:
                f.write(json.dumps(record) + "\n")
        print(f"  wrote {len(dump_records)} incorrect MMLU examples to {dump_path}")


if __name__ == "__main__":
    typer.run(main)
