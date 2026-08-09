# Build SFT (prompt, response) pairs from Open-R1's OpenR1-Math-220k dataset.

from __future__ import annotations

import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

SHARD_URL = "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00000-of-00010.parquet"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"


def download_shard(dest_path: str | Path) -> Path:
    # Download the first OpenR1-Math-220k parquet shard if not already present.
    
    dest_path = Path(dest_path)
    if not dest_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHARD_URL, dest_path)
    return dest_path


def _all_boxed_answers(text: str) -> list[str]:
    # All `\\boxed{...}` contents in `text`, left to right, with proper brace matching
    # a naive regex like `\\boxed\\{(.*?)\\}` breaks on nested braces e.g. `\\boxed{\\frac{1}{2}}`
    
    results = []
    search_start = 0
    while True:
        idx = text.find("\\boxed{", search_start)
        if idx == -1:
            break
        i = idx + len("\\boxed{")
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break  # unbalanced braces
        results.append(text[idx + len("\\boxed{") : i - 1])
        search_start = i
    return results


def build_sft_example(
    problem: str, generation: str, prompt_template: str, max_response_chars: int | None = None
) -> dict[str, str] | None:
    # Convert one (problem, R1 generation) pair into an r1_zero-format SFT example.
    # NuminaMath/R1 traces skew very long (median ~12k chars, i.e. ~3k tokens, in
    # the raw shard) — batching a few of these together OOMs a 24GB GPU. For our
    # reduced-scale experiments we filter to short traces via max_response_chars
    # (a cheap character-count proxy for token length; data.py has no tokenizer).

    boxed_values = _all_boxed_answers(generation)
    distinct_values = {v.strip() for v in boxed_values}
    if len(distinct_values) != 1:
        return None

    think_start = generation.find("<think>")
    think_end = generation.find("</think>")
    if think_start == -1 or think_end == -1 or think_end < think_start:
        return None

    think_body = generation[think_start + len("<think>") : think_end]
    answer = boxed_values[-1]

    response = f"{think_body}</think> <answer> {answer} </answer>"
    if max_response_chars is not None and len(response) > max_response_chars:
        return None

    prompt = prompt_template.replace("{question}", problem)
    return {"prompt": prompt, "response": response}


def build_sft_dataset(
    parquet_path: str | Path, max_examples: int | None = None, max_response_chars: int | None = 4000
) -> list[dict[str, str]]:
    # Read a local OpenR1-Math-220k parquet shard and build SFT (prompt, response) pairs.

    prompt_template = _PROMPT_PATH.read_text()

    pf = pq.ParquetFile(parquet_path)
    examples: list[dict[str, str]] = []

    for batch in pf.iter_batches(columns=["problem", "generations", "correctness_math_verify"]):
        for row in batch.to_pylist():
            correct_idxs = [i for i, ok in enumerate(row["correctness_math_verify"]) if ok]
            for idx in correct_idxs:
                example = build_sft_example(
                    row["problem"], row["generations"][idx], prompt_template, max_response_chars
                )
                if example is not None:
                    examples.append(example)
                    break  # one example per problem is enough

            if max_examples is not None and len(examples) >= max_examples:
                return examples

    return examples
