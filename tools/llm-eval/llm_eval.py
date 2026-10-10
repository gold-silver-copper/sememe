"""Score held-out `.docs` text with a small open causal LM, in bits per original byte.

Reference point for sememe; not part of sememe. See README.md.

Each document is scored on its own (no context across documents). The token
sequence is [start] + tokens(doc), where [start] is the tokenizer's BOS token,
or its EOS / <|endoftext|> token when it has no BOS (SmolLM2 and Qwen2.5 add no
BOS of their own). [start] is context only: it is never scored and is not
counted in `tokens`. Every document token is scored, including the first. No
end-of-document token is scored, so a document's length is not paid for.
Long documents use a sliding window: windows of `context` tokens advance by
`stride`, and each token is scored once, in the first window where it falls
past the previous window's end (so after the first window, every scored token
sees at least context - stride tokens of left context).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

THREADS = "8"
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SEP = "\x1e"


def read_docs(path: Path) -> list[str]:
    """Original text of every document in a `.docs` file or directory of them."""
    files = sorted(path.glob("*.docs")) if path.is_dir() else [path]
    if not files:
        sys.exit(f"no .docs files under {path}")
    docs = []
    for f in files:
        lines = f.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        cur: list[str] = []
        for line in lines:
            if line == SEP:
                if cur:
                    docs.append("".join(l + "\n" for l in cur))
                cur = []
            else:
                cur.append(line)
        if cur:
            docs.append("".join(l + "\n" for l in cur))
    return docs


def windows(ids: list[int], context: int, stride: int):
    """Yield (input ids, index in the input of the first scored position)."""
    prev_end = 1  # position 0 is [start], never scored
    begin = 0
    while True:
        end = min(begin + context, len(ids))
        yield ids[begin:end], prev_end - begin
        prev_end = end
        if end == len(ids):
            return
        begin += stride


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--input", required=True, type=Path, help=".docs file or directory")
    ap.add_argument("--set", required=True, help="name of the held-out set, for the output")
    ap.add_argument("--out", required=True, type=Path, help="output JSON path")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--batch-tokens", type=int, default=2048,
                    help="max padded tokens per forward pass (bounds logits memory)")
    ap.add_argument("--limit-docs", type=int, default=0, help="score only the first N docs (testing)")
    args = ap.parse_args()
    if not 0 < args.stride <= args.context - 1:
        sys.exit("need 0 < stride < context")

    torch.set_num_threads(int(THREADS))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 1 << 30
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if start is None:
        sys.exit("tokenizer has neither BOS nor EOS token")

    docs = read_docs(args.input)
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)

    jobs = []  # (input ids, first scored index)
    n_tokens = 0
    mismatched = 0
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"]
        if tok.decode(ids) != d:
            mismatched += 1
        n_tokens += len(ids)
        jobs.extend(windows([start] + ids, args.context, args.stride))
    jobs.sort(key=lambda j: len(j[0]))

    nats = 0.0
    scored = 0
    done = 0
    with torch.inference_mode():
        i = 0
        while i < len(jobs):
            # Longest-last order: grow the batch while the padded size fits.
            j = i + 1
            while j < len(jobs) and len(jobs[j][0]) * (j - i + 1) <= args.batch_tokens:
                j += 1
            batch = jobs[i:j]
            width = len(batch[-1][0])
            # Right padding needs no attention mask: causal attention keeps real
            # positions blind to the padding after them, and padding is never scored.
            x = torch.full((len(batch), width), start, dtype=torch.long)
            for r, (ids, _) in enumerate(batch):
                x[r, : len(ids)] = torch.tensor(ids)
            logits = model(input_ids=x).logits.float()
            for r, (ids, first) in enumerate(batch):
                # position p predicts token p+1; score tokens first..len-1
                pred = logits[r, first - 1 : len(ids) - 1]
                tgt = x[r, first : len(ids)]
                nats += torch.nn.functional.cross_entropy(pred, tgt, reduction="sum").item()
                scored += len(tgt)
            del logits
            done += len(batch)
            i = j
            print(f"\r{done}/{len(jobs)} windows, {scored}/{n_tokens} tokens", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    assert scored == n_tokens, (scored, n_tokens)

    bits = nats / math.log(2)
    result = {
        "model": args.model,
        "set": args.set,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "bits": round(bits, 1),
        "bits_per_byte": round(bits / n_bytes, 4),
        "context": args.context,
        "stride": args.stride,
        "seconds": round(time.time() - t0, 1),
        "docs": len(docs),
        "start_token": tok.convert_ids_to_tokens(start),
        "roundtrip_mismatched_docs": mismatched,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
