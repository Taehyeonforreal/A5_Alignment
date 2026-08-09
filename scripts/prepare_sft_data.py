"""Build data/openr1_math/sft.jsonl for the SFT step.

Output format matches the assignment's original SFT data contract: one JSON
object per line, each {"prompt": str, "response": str}.
"""

import json
from pathlib import Path

import typer

from cs336_alignment.data import build_sft_dataset, download_shard

SHARD_PATH = Path("data/openr1_math/train-00000-of-00010.parquet")
OUTPUT_PATH = Path("data/openr1_math/sft.jsonl")


def main(max_examples: int = 200, max_response_chars: int = 4000):
    download_shard(SHARD_PATH)
    examples = build_sft_dataset(SHARD_PATH, max_examples=max_examples, max_response_chars=max_response_chars)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    typer.run(main)
