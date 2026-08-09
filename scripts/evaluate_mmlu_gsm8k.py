"""Zero-shot MMLU + GSM8K evaluation for the Instruction Tuning supplement.

Reusable for both the pre-tuning baseline and the post-tuning re-evaluation —
pass a different `model_path` against the same benchmarks for an apples-to-
apples comparison (the "alignment tax" check).

Per the PDF spec, the two runs use DIFFERENT outer prompt wrappers even though
the MMLU/GSM8K question-formatting templates stay the same:
  - baseline (untuned base model, prompt_style="system"): wrapped in the
    zero-shot system prompt, which asks the model to reply inside a ``` ... ```
    block and start a new "# Query:" turn -- so generation stops at "# Query:".
  - post-tuning (prompt_style="alpaca"): "make sure to format the inputs in
    the same instruction tuning prompt format used for training" -- i.e. the
    Alpaca template the model actually saw in scripts/train_instruction_tuning.py,
    not the system prompt. No "# Query:" turn marker in this format, so
    generation just runs to the EOS token (vLLM's default stop behavior).
"""

import json
import random
import time

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.metrics import (
    build_gsm8k_prompt,
    build_mmlu_prompt,
    load_gsm8k_examples,
    load_mmlu_examples,
    parse_gsm8k_response,
    parse_mmlu_response,
)

_STOP_STRING = "# Query:"


def evaluate_mmlu(
    llm: LLM,
    sampling_params: SamplingParams,
    mmlu_dir: str,
    prompt_style: str,
    max_examples: int | None,
    seed: int,
    n_examples_shown: int = 3,
    n_incorrect_shown: int = 10,
    dump_path: str | None = None,
) -> None:
    examples = load_mmlu_examples(mmlu_dir, max_examples=max_examples, seed=seed)
    prompts = [build_mmlu_prompt(ex, prompt_style) for ex in examples]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    n_correct = 0
    n_unparsed = 0
    unparsed_shown = 0
    incorrect_examples = []
    dump_records = []
    for ex, output in zip(examples, outputs):
        response_text = output.outputs[0].text
        prediction = parse_mmlu_response(ex, response_text)
        if prediction is None:
            n_unparsed += 1
            category = "unparsed"
            if unparsed_shown < n_examples_shown:
                print(f"  [MMLU unparsed example] model output: {response_text!r}")
                unparsed_shown += 1
            incorrect_examples.append((ex, response_text, prediction))
        elif prediction == ex["answer"]:
            n_correct += 1
            category = None
        else:
            category = "wrong"
            incorrect_examples.append((ex, response_text, prediction))

        if category is not None and dump_path:
            dump_records.append(
                {
                    "category": category,
                    "subject": ex["subject"],
                    "question": ex["question"],
                    "options": ex["options"],
                    "gold": ex["answer"],
                    "pred": prediction,
                    "output": response_text,
                }
            )

    n = len(examples)
    print(f"MMLU: {n_correct}/{n} correct ({100 * n_correct / n:.1f}%), {n_unparsed} unparsed ({100 * n_unparsed / n:.1f}%)")
    print(f"MMLU throughput: {n / elapsed:.2f} examples/sec ({elapsed:.1f}s for {n} examples)")

    if incorrect_examples and n_incorrect_shown:
        print(f"  -- sampling up to {n_incorrect_shown} incorrect MMLU examples --")
        for ex, response_text, prediction in random.Random(seed).sample(
            incorrect_examples, min(n_incorrect_shown, len(incorrect_examples))
        ):
            print(f"  [MMLU incorrect] subject={ex['subject']!r} gold={ex['answer']} pred={prediction!r} output={response_text!r}")

    if dump_path:
        with open(dump_path, "w") as f:
            for record in dump_records:
                f.write(json.dumps(record) + "\n")
        print(f"  wrote {len(dump_records)} incorrect MMLU examples to {dump_path}")


def evaluate_gsm8k(
    llm: LLM,
    sampling_params: SamplingParams,
    gsm8k_path: str,
    prompt_style: str,
    max_examples: int | None,
    seed: int,
    n_examples_shown: int = 3,
    n_incorrect_shown: int = 10,
    dump_path: str | None = None,
) -> None:
    examples = load_gsm8k_examples(gsm8k_path, max_examples=max_examples, seed=seed)
    prompts = [build_gsm8k_prompt(ex, prompt_style) for ex in examples]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    n_correct = 0
    n_unparsed = 0
    unparsed_shown = 0
    incorrect_examples = []
    dump_records = []
    for ex, output in zip(examples, outputs):
        response_text = output.outputs[0].text
        prediction = parse_gsm8k_response(response_text)
        if prediction is None:
            n_unparsed += 1
            category = "unparsed"
            if unparsed_shown < n_examples_shown:
                print(f"  [GSM8K unparsed example] model output: {response_text!r}")
                unparsed_shown += 1
            incorrect_examples.append((ex, response_text, prediction))
        elif prediction == ex["ground_truth"]:
            n_correct += 1
            category = None
        else:
            category = "wrong"
            incorrect_examples.append((ex, response_text, prediction))

        if category is not None and dump_path:
            dump_records.append(
                {
                    "category": category,
                    "question": ex["question"],
                    "gold": ex["ground_truth"],
                    "pred": prediction,
                    "output": response_text,
                }
            )

    n = len(examples)
    print(f"GSM8K: {n_correct}/{n} correct ({100 * n_correct / n:.1f}%), {n_unparsed} unparsed ({100 * n_unparsed / n:.1f}%)")
    print(f"GSM8K throughput: {n / elapsed:.2f} examples/sec ({elapsed:.1f}s for {n} examples)")

    if incorrect_examples and n_incorrect_shown:
        print(f"  -- sampling up to {n_incorrect_shown} incorrect GSM8K examples --")
        for ex, response_text, prediction in random.Random(seed).sample(
            incorrect_examples, min(n_incorrect_shown, len(incorrect_examples))
        ):
            print(f"  [GSM8K incorrect] gold={ex['ground_truth']} pred={prediction!r} output={response_text!r}")

    if dump_path:
        with open(dump_path, "w") as f:
            for record in dump_records:
                f.write(json.dumps(record) + "\n")
        print(f"  wrote {len(dump_records)} incorrect GSM8K examples to {dump_path}")


def main(
    model_path: str = "meta-llama/Llama-3.1-8B",
    mmlu_dir: str = "data/mmlu/test",
    gsm8k_path: str = "data/gsm8k/test.jsonl",
    max_tokens: int = 512,
    # Full MMLU test is 14042 examples / GSM8K test is 1319 — for a reduced-scale
    # run (consistent with the main assignment's SFT/EI/GRPO experiments) we
    # subsample by default. Pass 0 to evaluate on the full benchmark instead.
    n_mmlu_examples: int = 1000,
    n_gsm8k_examples: int = 500,
    seed: int = 0,
    # Llama 3.1's default max_model_len is 131072 (128k context) -- vLLM sizes its
    # CUDA-graph capture and KV-cache bookkeeping off this, which is wildly more
    # than our short prompts need and puts unnecessary memory pressure on a single
    # RTX 4090. Our longest prompt (system prompt + MMLU question) is well under 2k
    # tokens, so cap it hard and skip CUDA-graph capture entirely for this eval.
    max_model_len: int = 2048,
    # "system" for the untuned base-model baseline (zero_shot_system_prompt.prompt
    # wrapper, stop at "# Query:"); "alpaca" for the post-instruction-tuning
    # re-evaluation (alpaca_sft.prompt wrapper -- the format the model was
    # actually trained on -- stop at EOS).
    prompt_style: str = "system",
    mmlu_dump_path: str | None = None,
    gsm8k_dump_path: str | None = None,
):
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=[_STOP_STRING] if prompt_style == "system" else None,
    )

    evaluate_mmlu(
        llm, sampling_params, mmlu_dir, prompt_style, max_examples=n_mmlu_examples or None, seed=seed, dump_path=mmlu_dump_path
    )
    evaluate_gsm8k(
        llm, sampling_params, gsm8k_path, prompt_style, max_examples=n_gsm8k_examples or None, seed=seed, dump_path=gsm8k_dump_path
    )


if __name__ == "__main__":
    typer.run(main)
