# CS336 Spring 2025 · Assignment 5: Alignment — self-study implementation

Implementation of [Stanford CS336](https://stanford-cs336.github.io/spring2025/) Assignment 5.
The repo has two parts:

1. **Reasoning RL**: SFT → Expert Iteration → GRPO, implemented from scratch on Qwen2.5-Math-1.5B.
2. **Supplement**: instruction tuning of Llama 3.1 8B, plus an MMLU/GSM8K evaluation harness. I used them to check whether an apparent MMLU "alignment tax" was real.

Everything ran at **reduced scale on rented single GPUs** (RunPod). All numbers below are from that scale. See [Limitations](#limitations).

| | Result |
|---|---|
| SFT | answer accuracy 0% → 17.5% (40 held-out problems) |
| Expert Iteration | correct rollouts 10 → 150 → 190 (out of 256 per round) |
| GRPO (off-policy) | mean reward 3.1% → 83.6% on training rollouts; clip fraction 0.3–5.7% |
| Supplement | MMLU −9.6%p under zero-shot parsing (z = −4.32). Within noise under log-likelihood (z = +0.60) and 5-shot (z = −1.27) |

---

## Part 1. Reasoning RL: SFT → EI → GRPO

### Setup: original handout vs. this repo

| | Original handout | This repo |
|---|---|---|
| Model | Qwen2.5-Math-1.5B | same |
| SFT data | MATH + DeepSeek-R1 reasoning traces (Stanford cluster only) | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) (public R1 traces), filtered to 200 examples |
| EI / GRPO data | MATH train | GSM8K train (MATH isn't publicly distributed anymore; easier problems still give non-zero reward at small group sizes) |
| Prompt / grader | `r1_zero` prompt, `r1_zero_reward_fn` | same (staff-provided) |
| Compute | 2 GPUs (vLLM on one, training on the other) | 1× RTX 4090 24GB |
| Scale | full datasets, hundreds of steps | 100 SFT steps, 3 EI rounds, 10 GRPO rounds |

**Single-GPU design.** vLLM generation and HF training run as **separate subprocesses**. Each one exits when its phase is done, so the OS takes back all of its GPU memory before the next phase starts. This trades wall-clock time for memory: phases run one at a time instead of side by side as in the two-GPU setup.

### SFT

160 train / 40 val examples, lr 2e-5, effective batch 32 (micro-batch 1 × 32 accumulation), 100 steps (≈ 20 passes over 160 examples), gradient checkpointing.

| | Base | After SFT |
|---|---|---|
| Format correct (`<think>…</think> <answer>…</answer>`) | 4/40 (10%) | 10/40 (25%) |
| Answer correct | 0/40 (0%) | **7/40 (17.5%)** |

Validation problems are olympiad-level (NuminaMath source), sampled at temperature 1.0, n = 40.

### Expert Iteration

Each round: 64 fresh GSM8K questions × G = 4 rollouts (≤ 256). Keep only the rollouts `r1_zero_reward_fn` marks correct, then run 20 SFT steps on them. `train_ei.py` reuses `train_sft.py` unchanged.

| Round | 1 (base) | 2 | 3 |
|---|---|---|---|
| Correct rollouts / 256 | 10 | **150** | 190 |

*Hypothesis (not tested):* round 1 trained on only 10 examples, yet correct rollouts on new questions jumped to 150. That looks more like learning the `<think>/<answer>` **format** than learning math. Qwen2.5-Math was not pretrained on this format (Liu et al., 2025).

### GRPO

**Off-policy redesign.** With the handout defaults (1 epoch per rollout batch, train batch = rollout batch), there is one optimizer step per round. So π_θ = π_old, ρ ≡ 1, and clipping never fires. I switched to `epochs_per_rollout_batch=2` and `train_batch_size=32`. With 128 rollouts per round that gives 4 steps × 2 epochs = **8 optimizer steps per round**. Two things make this correct:
- `old_log_probs` is computed once at the start of the round under `torch.no_grad()` and kept frozen.
- The rollout batch is tokenized once, so padding and indices stay aligned with `old_log_probs` across steps.

Config: 10 rounds × 16 questions × G = 8, cliprange 0.2, lr 1e-5, group-std normalization on (`normalize_by_std=True`, i.e. not Dr. GRPO), `masked_mean` aggregation.

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean reward | 0.031 | 0.047 | 0.211 | 0.352 | 0.438 | 0.430 | 0.508 | 0.484 | 0.719 | **0.836** |
| Clip fraction | 0.003 | 0.011 | 0.057 | 0.021 | 0.030 | 0.028 | 0.028 | 0.037 | 0.031 | 0.019 |

Clip fraction is non-zero in every round, so the policy really moves away from π_old within a round and the clipped term gets selected. Mean reward is measured on **that round's training rollouts**, not on a held-out set.

### Unit tests

`tests/test_sft.py` and `tests/test_grpo.py`: 24/24 pass.

---

## Part 2. Supplement: instruction tuning & the "alignment tax"

**Question.** Instruction tuning targets instruction following. Does it cost base capability? I measured knowledge (MMLU) and math reasoning (GSM8K) before and after tuning.

**Scope.** I built the evaluation harness for MMLU and GSM8K only. AlpacaEval and SimpleSafetyTests need a Llama 3.3 70B judge, which was out of budget. I ran instruction tuning. **DPO was not implemented** (`tests/test_dpo.py` is not wired).

### Setup

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B` (base), full fine-tune |
| Data | `safety_augmented_ultrachat_200k_single_turn`, subsampled to 3,000 train / 100 val (handout recipe: ~210k, ~24 H100-hours) |
| Training | packed sequences of 512, effective batch 32 (2 × 16), lr 2e-5 with cosine decay and 3% warmup, 1 epoch = 96 optimizer steps, `bitsandbytes` AdamW8bit, A100 80GB |
| Loss | train 1.54 → 1.41, final val 1.506 |
| Eval | MMLU: 1,000 questions across 57 subjects. GSM8K: 500 questions. Zero-shot, generate-then-parse |
| Prompt wrapper | baseline: zero-shot system prompt. Tuned: Alpaca template (the handout says to use the format the model was trained on). The wrapper differs between the two runs, which is a confound. |

### Observed result (zero-shot, generate-then-parse)

| | Baseline | Tuned | Δ |
|---|---|---|---|
| MMLU | 60.0% (parse failure 0.0%) | 50.4% (parse failure **16.6%**) | **−9.6%p** |
| GSM8K | 16.2% (parse failure 0.6%) | 21.6% (parse failure 0.6%) | +5.4%p |

### Audit: all 496 MMLU errors of the tuned model

The MMLU parser only accepts `answer is <letter>`, which is a strict match.

| Category | Count |
|---|---|
| Answered with a leading letter (`"D. …"`) but no `"answer is"`, so the parser missed it. Correct when re-parsed | **51** |
| Same near-format, still wrong after re-parsing | 12 |
| Followed the format, wrong answer (real error) | 330 |
| No letter at all, free-form prose | 103 |

Restricted to answers that can be parsed, accuracy is 555/897 = 61.9%, close to the 60.0% baseline.

### Triangulation: two scoring methods that don't depend on output format

| Scoring | Baseline | Tuned | Δ | z | \|z\| > 1.96 |
|---|---|---|---|---|---|
| Zero-shot generate + parse | 60.0% | 50.4% | −9.6%p | **−4.32** | yes |
| Log-likelihood over A/B/C/D | 62.1% | 63.4% | +1.3%p | +0.60 | no |
| 5-shot generate + parse | 67.2% | 64.5% | −2.7%p | −1.27 | no |

- **Log-likelihood:** no generation step. It compares log P(" A" … " D" | prompt) using one forward pass on a bare prompt ending in `Answer:`. The prompt is identical for both models, which also removes the wrapper confound.
- **5-shot:** still generates, but five same-subject dev examples show the format instead of a written instruction. Parse failures were 0% for both models.
- **z-test:** two-proportion test with pooled SE = √(2p̄(1−p̄)/n), n = 1,000.

**Conclusion.** The drop is significant only under the scoring method that depends on output format. Under two methods that remove that dependence in different ways, the difference is within noise. In this setup, the MMLU "alignment tax" is mostly a **measurement artifact**: a strict parser combined with a free-form answering style learned during tuning. It is not evidence of lost knowledge.

### Left open

- **5-shot −2.7%p** with 0% parse failures: possibly instruction tuning interferes with in-context pattern completion. Not resolvable at n = 1,000.
- **Paired design:** both models answer the same 1,000 questions, so McNemar's test is the proper one. The independent-proportion z-test used here is conservative.
- **GSM8K +5.4%p** (z ≈ +2.18 under the same test) was **not audited or triangulated**. Some tuned-model errors were refusals ("there is no way to know … without additional information"), but I didn't count them.

---

## Limitations

- Reduced scale throughout: small datasets, few steps, a **single seed**, no hyperparameter sweeps.
- GRPO reward is measured on training rollouts. There is no held-out GSM8K evaluation for EI or GRPO.
- Substitute datasets (OpenR1-Math, GSM8K), so results aren't comparable to the handout's MATH numbers.
- No ablations (Dr. GRPO normalization, length normalization, prompt variants).
- Supplement: DPO, AlpacaEval and SimpleSafetyTests were not done.

---

## What I wrote vs. what came with the starter code

| Written by me | Role |
|---|---|
| `cs336_alignment/utils.py` | `tokenize_prompt_and_output`, `compute_entropy`, `get_response_log_probs`, `masked_normalize`, `masked_mean` |
| `cs336_alignment/sft.py` | `sft_microbatch_train_step` |
| `cs336_alignment/grpo.py` | group-normalized rewards, naive PG loss, GRPO-Clip loss, dispatcher, `grpo_microbatch_train_step` |
| `cs336_alignment/data.py` | OpenR1-Math → `r1_zero` SFT examples (`\boxed{}` extraction by brace-depth counting, filtering) |
| `cs336_alignment/metrics.py` | MMLU/GSM8K parsing, data loading, zero-shot / bare / few-shot prompt builders |
| `cs336_alignment/packing.py` | packed SFT dataset, batch iterator |
| `cs336_alignment/prompts/{mmlu,gsm8k}_zero_shot.prompt` | templates transcribed from the handout |
| `scripts/` (all except the two below) | data prep, training, generation and evaluation entry points |

| Starter code (unchanged or wiring only) | Notes |
|---|---|
| `cs336_alignment/drgrpo_grader.py` | `r1_zero_reward_fn`, used for all Part 1 grading |
| `cs336_alignment/prompts/{r1_zero,question_only,alpaca_sft,zero_shot_system_prompt}.prompt` | unchanged |
| `scripts/evaluate_safety.py`, `scripts/alpaca_eval_vllm_llama3_3_70b_fn/` | unchanged, unused |
| `tests/` | tests and snapshots unchanged. In `adapters.py` I only filled in calls to my implementations. In `conftest.py` I changed `model_id` to `Qwen/Qwen2.5-Math-1.5B` |

---

## Reproducing

```sh
uv sync --no-install-package flash-attn
uv sync
uv run pytest tests/test_sft.py tests/test_grpo.py tests/test_metrics.py tests/test_data.py
```

`data/` is gitignored. Copy it from the [official starter repo](https://github.com/stanford-cs336/assignment5-alignment) (GSM8K, MMLU). The prep scripts below create the rest. Llama 3.1 8B is gated, so set `HF_TOKEN`. Checkpoints default to `/workspace/...` (RunPod); override with `--output-dir`. Script defaults match the runs reported above.

```sh
# Part 1
uv run python scripts/prepare_sft_data.py
uv run python scripts/evaluate_math.py                                   # base model
uv run python scripts/train_sft.py
uv run python scripts/evaluate_math.py --model-path /workspace/sft_model_out
uv run python scripts/train_ei.py
uv run python scripts/train_grpo.py

# Part 2
uv run python scripts/prepare_instruction_tuning_data.py
uv run python scripts/evaluate_mmlu_gsm8k.py --prompt-style system       # baseline
uv run python scripts/train_instruction_tuning.py
uv run python scripts/evaluate_mmlu_gsm8k.py --model-path /workspace/instruction_tuned_model_out --prompt-style alpaca
uv run python scripts/evaluate_mmlu_loglik.py  [--model-path ...]
uv run python scripts/evaluate_mmlu_fewshot.py [--model-path ...]
```

---

## Original assignment

- Handout: [cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)
- Supplement: [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)
- Starter code: [stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment). `CHANGELOG.md` is the course staff's changelog.
