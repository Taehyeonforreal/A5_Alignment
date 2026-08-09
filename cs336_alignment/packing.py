"""Packed language-modeling dataset for instruction tuning.

Unlike the main-assignment SFT (`sft.py`, masked loss on response tokens only),
instruction tuning here packs many (prompt, response) documents into one long
token stream and trains next-token-prediction loss over the whole stream —
no response_mask, no per-example padding.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Starter-provided template (cs336_alignment/prompts/alpaca_sft.prompt) ends with a
# trailing newline after {response}; strip it since the packed fixture has none.
_ALPACA_TEMPLATE = (Path(__file__).parent / "prompts" / "alpaca_sft.prompt").read_text().rstrip("\n")


def _load_documents(dataset_path: str | os.PathLike) -> list[dict[str, str]]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f]


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        documents = _load_documents(dataset_path)
        if shuffle:
            documents = documents.copy()
            random.shuffle(documents)

        all_tokens: list[int] = []
        for doc in documents:
            text = _ALPACA_TEMPLATE.format(instruction=doc["prompt"], response=doc["response"])
            all_tokens.extend(tokenizer.encode(text))
            all_tokens.append(tokenizer.eos_token_id)

        num_chunks = (len(all_tokens) - 1) // seq_length
        self._input_ids = []
        self._labels = []
        for i in range(num_chunks):
            start = i * seq_length
            self._input_ids.append(torch.tensor(all_tokens[start : start + seq_length], dtype=torch.long))
            self._labels.append(torch.tensor(all_tokens[start + 1 : start + 1 + seq_length], dtype=torch.long))

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self._input_ids[idx], "labels": self._labels[idx]}


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        examples = [dataset[i] for i in batch_indices]
        batches.append(
            {
                "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
                "labels": torch.stack([ex["labels"] for ex in examples]),
            }
        )
    return batches
