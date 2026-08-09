"""Evaluate a model (base or fine-tuned) on r1_zero-format math problems.

Reusable for zero-shot baseline eval and post-SFT eval — pass a different
`model_path` against the same `val_path` for an apples-to-apples comparison.
"""

import json

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def extract_ground_truth_answer(response: str) -> str | None:
    """Pull the ground-truth answer out of a training example's `response` field
    (which ends with `<answer> ... </answer>` per our r1_zero-format SFT data).
    """
    if "<answer>" not in response or "</answer>" not in response:
        return None
    return response.split("<answer>")[-1].split("</answer>")[0].strip()


def load_val_examples(path: str) -> list[dict[str, str]]:
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ground_truth = extract_ground_truth_answer(ex["response"])
            if ground_truth is not None:
                examples.append({"prompt": ex["prompt"], "ground_truth": ground_truth})
    return examples


def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    val_path: str = "data/openr1_math/sft_val.jsonl",
    max_tokens: int = 1024,
    temperature: float = 1.0,
):
    val_examples = load_val_examples(val_path)
    print(f"Evaluating {model_path} on {len(val_examples)} validation examples")

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts = [ex["prompt"] for ex in val_examples]
    outputs = llm.generate(prompts, sampling_params)

    n = len(val_examples)
    n_format_ok = 0.0
    n_correct = 0.0
    for ex, output in zip(val_examples, outputs):
        response_text = output.outputs[0].text
        reward = r1_zero_reward_fn(response_text, ex["ground_truth"])
        n_format_ok += reward["format_reward"]
        n_correct += reward["answer_reward"]

    print(f"format correct: {n_format_ok:.0f}/{n} ({100 * n_format_ok / n:.1f}%)")
    print(f"answer correct: {n_correct:.0f}/{n} ({100 * n_correct / n:.1f}%)")


if __name__ == "__main__":
    typer.run(main)
