"""Build data/instruction_tuning/{train,val}.jsonl for the Instruction Tuning step.

Source: safety_augmented_ultrachat_200k_single_turn (UltraChat-200k + safety data),
publicly mirrored by the course at nlp.stanford.edu. Already {"prompt", "response"}
per line -- same schema get_packed_sft_dataset expects, no conversion needed.

Reduced-scale by design: the assignment's own recipe is a full epoch over all
~210k train examples (~24 H100 hrs), which is far beyond the scope of this
self-study pass. We subsample instead, same principle used for the main
assignment's SFT/EI/GRPO experiments.
"""

import gzip
import json
import random
from pathlib import Path

import typer

TRAIN_URL = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
TEST_URL = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"


def download(url: str, dest_path: Path) -> Path:
    if not dest_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def subsample_jsonl_gz(gz_path: Path, n: int, seed: int) -> list[dict]:
    with gzip.open(gz_path, "rt") as f:
        examples = [json.loads(line) for line in f]
    return random.Random(seed).sample(examples, min(n, len(examples)))


def main(
    n_train: int = 3000,
    n_val: int = 100,
    seed: int = 0,
    raw_dir: str = "data/instruction_tuning/raw",
    output_dir: str = "data/instruction_tuning",
):
    train_gz = download(TRAIN_URL, Path(raw_dir) / "train.jsonl.gz")
    test_gz = download(TEST_URL, Path(raw_dir) / "test.jsonl.gz")

    train_examples = subsample_jsonl_gz(train_gz, n_train, seed)
    val_examples = subsample_jsonl_gz(test_gz, n_val, seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in [("train.jsonl", train_examples), ("val.jsonl", val_examples)]:
        with open(out_dir / name, "w") as f:
            for ex in examples:
                f.write(json.dumps({"prompt": ex["prompt"], "response": ex["response"]}) + "\n")

    print(f"Wrote {len(train_examples)} train / {len(val_examples)} val examples to {out_dir}")


if __name__ == "__main__":
    typer.run(main)
