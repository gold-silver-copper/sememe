# llm-eval: small open LLMs as a compression reference

Measures how well small pretrained causal language models compress sememe's
held-out text, in bits per original byte, so sememe's numbers have an outside
reference point. This tool is not part of sememe (sememe uses no ML
libraries); it has its own Python virtualenv.

These models were trained on trillions of tokens (SmolLM2: ~2-4T, Qwen2.5:
~18T), far more than BabyLM's 10M/100M words, and FineWeb-Edu is likely in
SmolLM2's training data. They are a reference point, not a like-for-like
comparison.

## What is measured

- Input: a `.docs` file or directory of them. A document's original bytes are
  its lines, each followed by `\n` (separator lines excluded).
- Each document is scored on its own, with no context from other documents.
- Token sequence: `[start] + tokens(doc)`. SmolLM2 and Qwen2.5 add no BOS of
  their own, so `[start]` is the tokenizer's BOS, or its EOS when it has no BOS
  (`<|endoftext|>` for all three models). `[start]` is context only: it is
  not scored and not counted in `tokens`. The first real token is scored.
- No end-of-document token is scored (document lengths are free).
- Sliding window: 1024-token windows, stride 512. Each token is scored once;
  after the first window every scored token has at least 512 tokens of
  context.
- `bits` = sum of -log2 p over all document tokens; `bits_per_byte` =
  `bits / bytes` with `bytes` the original UTF-8 bytes, newlines included.
- CPU only, float32, 8 threads. `roundtrip_mismatched_docs` counts documents
  whose tokens do not decode back to the exact original text (0 means the
  tokenizer was lossless on this set).

Output: `results/llm-<model>-<set>.json` with `model, set, bytes, tokens,
bits, bits_per_byte, context, stride, seconds` (plus `docs`, `start_token`,
`roundtrip_mismatched_docs`).

## Setup (once)

```sh
cd ~/code/sememe
python3 -m venv ~/.cache/sememe-llm-venv
~/.cache/sememe-llm-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.cache/sememe-llm-venv/bin/pip install numpy transformers
```

Tested with Python 3.14, torch 2.14.1+cpu, transformers 5.19.0. Models download
from Hugging Face into `~/.cache/huggingface` on first use (no login).

## Run

All six (model, set) pairs, one after another (about 1.5 hours with 8 threads):

```sh
cd ~/code/sememe
export OMP_NUM_THREADS=8
for m in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
  for s in babylm-dev-sample:data/prepared/babylm/dev-sample \
           fineweb-heldout-sample:data/prepared/fineweb/heldout-sample.docs; do
    nice -n 10 ~/.cache/sememe-llm-venv/bin/python tools/llm-eval/llm_eval.py \
      --model "$m" --input "${s#*:}" --set "${s%%:*}" \
      --out "results/llm-$(basename "$m")-${s%%:*}.json"
  done
done
```

One pair:

```sh
cd ~/code/sememe
~/.cache/sememe-llm-venv/bin/python tools/llm-eval/llm_eval.py \
  --model HuggingFaceTB/SmolLM2-135M --input data/prepared/babylm/dev-sample \
  --set babylm-dev-sample --out results/llm-SmolLM2-135M-babylm-dev-sample.json
```

Options: `--context`, `--stride`, `--batch-tokens` (padded tokens per forward
pass; 2048 keeps Qwen2.5's 152k-vocab logits around 1.2 GB), `--limit-docs N`
(quick test).
