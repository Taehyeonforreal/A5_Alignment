"""One rollout-generation phase of Expert Iteration.

Samples `group_size` completions per GSM8K question with the current policy
(vLLM), keeps every completion verified correct by r1_zero_reward_fn, and
writes the resulting (prompt, response) pairs to disk as SFT training data for
the next phase (scripts/train_sft.py). Runs as its own process so vLLM's GPU
memory is fully released on exit — see scripts/train_ei.py.
"""

import json
import random
from pathlib import Path

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

_PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")


def load_gsm8k_questions(path: str, n: int, seed: int) -> list[dict[str, str]]:
    all_examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ground_truth = ex["answer"].split("####")[-1].strip()
            all_examples.append({"question": ex["question"], "ground_truth": ground_truth})
    random.seed(seed)
    return random.sample(all_examples, min(n, len(all_examples)))


def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    gsm8k_path: str = "data/gsm8k/train.jsonl",
    output_path: str = "data/ei/round_sft.jsonl",
    n_questions: int = 64,
    group_size: int = 4,
    max_tokens: int = 512,
    seed: int = 0,
):
    questions = load_gsm8k_questions(gsm8k_path, n_questions, seed)
    prompt_template = _PROMPT_PATH.read_text()
    prompts = [prompt_template.replace("{question}", q["question"]) for q in questions]

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=4,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)

    sft_examples = []
    for question, prompt, output in zip(questions, prompts, outputs):
        for completion in output.outputs:
            reward = r1_zero_reward_fn(completion.text, question["ground_truth"])
            if reward["reward"] == 1.0:
                sft_examples.append({"prompt": prompt, "response": completion.text})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"{len(sft_examples)} correct rollouts from {len(questions)} questions -> {output_path}")


if __name__ == "__main__":
    typer.run(main)
