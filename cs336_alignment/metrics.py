"""MMLU/GSM8K data loading, prompt construction, and answer parsing.

Shared by scripts/evaluate_mmlu_gsm8k.py (generative parsing) and
scripts/evaluate_mmlu_loglik.py (log-likelihood scoring) -- both need the
same examples and the same prompt text, they just score the model's response
differently.

Both prompts force a specific closing phrase, so the generative parser is
just string matching, not free-form extraction:
  - MMLU:   "... The correct answer is _" -> a letter A/B/C/D.
  - GSM8K:  free-form reasoning -> the last number that occurs in the output.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

_MMLU_ANSWER_RE = re.compile(r"answer is\s*:?\s*\(?([A-Da-d])\)?(?![a-zA-Z])")
# Fallback for responses that lead with a bare letter ("D. explanation...") instead
# of "answer is D" -- found via manual error analysis that ~1/3 of "unparsed" MMLU
# responses were actually this near-miss case, not genuine free-form drift. Also
# what a few-shot-style "Answer: D" completion looks like.
_MMLU_LEADING_LETTER_RE = re.compile(r"^\s*\(?([A-Da-d])\)?(?=[\.\):\s]|$)")
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_TEMPLATE = (_PROMPTS_DIR / "zero_shot_system_prompt.prompt").read_text()
_ALPACA_TEMPLATE = (_PROMPTS_DIR / "alpaca_sft.prompt").read_text().rstrip("\n")
_MMLU_TEMPLATE = (_PROMPTS_DIR / "mmlu_zero_shot.prompt").read_text()
_GSM8K_TEMPLATE = (_PROMPTS_DIR / "gsm8k_zero_shot.prompt").read_text()


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    match = _MMLU_ANSWER_RE.search(model_output)
    if match is not None:
        return match.group(1).upper()
    match = _MMLU_LEADING_LETTER_RE.match(model_output)
    if match is not None:
        return match.group(1).upper()
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = _NUMBER_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1].replace("$", "").replace(",", "")


def load_mmlu_examples(mmlu_dir: str, max_examples: int | None = None, seed: int = 0) -> list[dict]:
    examples = []
    for csv_path in sorted(Path(mmlu_dir).glob("*_test.csv")):
        subject = csv_path.name.removesuffix("_test.csv").replace("_", " ")
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                question, option_a, option_b, option_c, option_d, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": [option_a, option_b, option_c, option_d],
                        "answer": answer,
                    }
                )
    if max_examples is not None and len(examples) > max_examples:
        examples = random.Random(seed).sample(examples, max_examples)
    return examples


def load_gsm8k_examples(gsm8k_path: str, max_examples: int | None = None, seed: int = 0) -> list[dict]:
    examples = []
    with open(gsm8k_path) as f:
        for line in f:
            ex = json.loads(line)
            ground_truth = ex["answer"].split("####")[-1].strip()
            examples.append({"question": ex["question"], "ground_truth": ground_truth})
    if max_examples is not None and len(examples) > max_examples:
        examples = random.Random(seed).sample(examples, max_examples)
    return examples


def load_mmlu_dev_examples(mmlu_dev_dir: str) -> dict[str, list[dict]]:
    # 5 examples per subject, for the standard MMLU few-shot protocol
    # (Hendrycks et al., 2021 uses 5-shot by default -- exemplars are drawn from
    # the same subject as the test question, never from the test split itself).
    by_subject: dict[str, list[dict]] = {}
    for csv_path in sorted(Path(mmlu_dev_dir).glob("*_dev.csv")):
        subject = csv_path.name.removesuffix("_dev.csv").replace("_", " ")
        examples = []
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                question, option_a, option_b, option_c, option_d, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": [option_a, option_b, option_c, option_d],
                        "answer": answer,
                    }
                )
        by_subject[subject] = examples
    return by_subject


def _format_mmlu_qa(mmlu_example: dict, answer_letter: str | None = None) -> str:
    text = (
        f"{mmlu_example['question']}\n"
        f"A. {mmlu_example['options'][0]}\n"
        f"B. {mmlu_example['options'][1]}\n"
        f"C. {mmlu_example['options'][2]}\n"
        f"D. {mmlu_example['options'][3]}\n"
        f"Answer:"
    )
    if answer_letter is not None:
        text += f" {answer_letter}"
    return text


def build_mmlu_fewshot_prompt(mmlu_example: dict, few_shot_examples: list[dict]) -> str:
    # No system-prompt/Alpaca wrapper and no "respond in this form" instruction --
    # the K worked (question, answer) pairs demonstrate the expected format
    # directly, which is both the standard protocol and, mechanically, the same
    # reason the wrapper had to be dropped for log-likelihood scoring: the
    # prompt must end exactly at the answer position.
    blocks = [f"The following are multiple choice questions (with answers) about {mmlu_example['subject']}."]
    for shot in few_shot_examples:
        blocks.append(_format_mmlu_qa(shot, shot["answer"]))
    blocks.append(_format_mmlu_qa(mmlu_example))
    return "\n\n".join(blocks)


def _wrap_instruction(instruction: str, prompt_style: str) -> str:
    if prompt_style == "system":
        return _SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction)
    elif prompt_style == "alpaca":
        # No {response} to fill in yet -- keep only the prefix up to "### Response:\n"
        # and let the model generate the response from there.
        prefix = _ALPACA_TEMPLATE.split("{response}")[0]
        return prefix.format(instruction=instruction)
    raise ValueError(f"unknown prompt_style: {prompt_style!r}")


def _mmlu_instruction(mmlu_example: dict) -> str:
    return _MMLU_TEMPLATE.format(
        subject=mmlu_example["subject"],
        question=mmlu_example["question"],
        option_a=mmlu_example["options"][0],
        option_b=mmlu_example["options"][1],
        option_c=mmlu_example["options"][2],
        option_d=mmlu_example["options"][3],
    )


def build_mmlu_prompt(mmlu_example: dict, prompt_style: str) -> str:
    return _wrap_instruction(_mmlu_instruction(mmlu_example), prompt_style)


def build_mmlu_bare_prompt(mmlu_example: dict) -> str:
    # No system-prompt/Alpaca wrapper: ends directly in "...\nAnswer:" so the
    # very next token IS the answer position -- required for log-likelihood
    # scoring (evaluate_mmlu_loglik.py), which teacher-forces a single token
    # right after the prompt and has no use for a format-eliciting wrapper.
    return _mmlu_instruction(mmlu_example)


def build_gsm8k_prompt(gsm8k_example: dict, prompt_style: str) -> str:
    instruction = _GSM8K_TEMPLATE.format(question=gsm8k_example["question"])
    return _wrap_instruction(instruction, prompt_style)
