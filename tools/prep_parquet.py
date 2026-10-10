#!/usr/bin/env python3
"""Prepare FineWeb-Edu and Wikipedia parquet files as sememe `.docs` files.

Run with the virtualenv at ~/.cache/sememe-venv (it has pyarrow), from the
repository root:

    ~/.cache/sememe-venv/bin/python tools/prep_parquet.py fineweb --files 000_00000.parquet
    ~/.cache/sememe-venv/bin/python tools/prep_parquet.py wikipedia

Everything streams parquet batch by batch; nothing holds more than one
readability bucket (~64 MB of text) in memory.

The `.docs` format: documents are runs of lines separated by a line holding
exactly U+001E. Each parquet row is one document, its `text` split on "\\n"
(so the original bytes of a document are its text plus one final newline).
Rows whose text is empty or has a line that is exactly U+001E are skipped and
counted. Each `.docs` file gets a `.ids` side file: one `<row id>[\\t<score>]`
per document.

Choices by hash use `hash64(seed, row id)`: the first 8 bytes,
little-endian, of sha256(seed as 8 little-endian bytes + id), the same
function as `sememe_data::hash64` in Rust. `u = (h >> 11) / 2**53` is
uniform in [0, 1).
"""

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

SEP = "\x1e"
MB = 1_000_000


def hash64(seed: int, data: bytes) -> int:
    d = hashlib.sha256(seed.to_bytes(8, "little") + data).digest()
    return int.from_bytes(d[:8], "little")


def peak_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def unit(h: int) -> float:
    return (h >> 11) / float(1 << 53)


class DocWriter:
    """Writes a `.docs` file and its `.ids` side file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w", encoding="utf-8", newline="")
        self.ids = open(path.with_suffix(".ids"), "w", encoding="utf-8", newline="")
        self.docs = 0
        self.bytes = 0
        self.lines = 0

    def write(self, lines, doc_id, score=None):
        if self.docs:
            self.f.write(SEP + "\n")
        for line in lines:
            self.f.write(line)
            self.f.write("\n")
        self.docs += 1
        self.lines += len(lines)
        self.bytes += sum(len(l.encode("utf-8")) + 1 for l in lines)
        self.ids.write(doc_id if score is None else f"{doc_id}\t{score:.4f}")
        self.ids.write("\n")

    def close(self):
        self.f.close()
        self.ids.close()
        return {"docs": self.docs, "lines": self.lines, "bytes": self.bytes}


class ShardWriter:
    """Rolls over to `<prefix>-<k>.docs` every `shard_bytes`."""

    def __init__(self, out: Path, prefix: str, shard_bytes: int):
        self.out, self.prefix, self.shard_bytes = out, prefix, shard_bytes
        self.k = 0
        self.w = None
        self.stats = {}

    def write(self, lines, doc_id, score=None):
        if self.w is None or self.w.bytes >= self.shard_bytes:
            self._close_current()
            self.w = DocWriter(self.out / f"{self.prefix}-{self.k:03d}.docs")
            self.k += 1
        self.w.write(lines, doc_id, score)

    def _close_current(self):
        if self.w is not None:
            self.stats[self.w.path.name] = self.w.close()

    def close(self):
        self._close_current()
        self.w = None
        return self.stats


def doc_lines(text):
    """The document's lines, or None if it cannot be represented."""
    if not text:
        return None
    lines = text.split("\n")
    if any(l == SEP for l in lines):
        return None
    return lines


def text_bytes_estimate(path: Path) -> int:
    """Uncompressed size of the `text` column, from the parquet metadata."""
    md = pq.ParquetFile(path).metadata
    total = 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema == "text":
                total += c.total_uncompressed_size
    return total


def rows(path: Path, columns, max_rows=None):
    """Yields row dicts, one row group at a time. (`iter_batches` grows the
    Arrow memory pool without bound on these files; `read_row_group` stays
    flat.)"""
    f = pq.ParquetFile(path)
    n = 0
    for g in range(f.num_row_groups):
        table = f.read_row_group(g, columns=columns)
        cols = {c: table.column(c).to_pylist() for c in columns}
        del table
        for i in range(len(cols[columns[0]])):
            yield {c: cols[c][i] for c in columns}
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def pick_files(directory: Path, names):
    all_files = sorted(directory.glob("*.parquet"))
    if not names:
        return all_files
    wanted = [directory / n for n in names]
    for w in wanted:
        if not w.exists():
            sys.exit(f"no such file: {w}")
    return wanted


class HeldOut:
    """Held-out rows of the held-out file, chosen by u = unit(hash64(seed, id)).

    Rows with u < 2 * f_sample are sample candidates, kept in memory (~2x the
    sample size); at the end the lowest-u ones up to `sample_bytes` form the
    sample and the others join the held-out set. Rows with u in
    [2 * f_sample, 2 * f_sample + f_heldout) are held out. So the sample is
    ~exactly its size and the held-out set is ~its size."""

    def __init__(self, out: Path, est_bytes: int, sample_mb: float, heldout_mb: float):
        self.sample_bytes = sample_mb * MB
        self.f_sample = 2 * self.sample_bytes / est_bytes
        self.f_heldout = heldout_mb * MB / est_bytes
        self.sample_w = DocWriter(out / "heldout-sample.docs")
        self.heldout_w = DocWriter(out / "heldout.docs")
        self.candidates = []

    def offer(self, u, doc_id, lines):
        """True if the row is held out (written or kept)."""
        if u < self.f_sample:
            self.candidates.append((u, doc_id, lines))
            return True
        if u < self.f_sample + self.f_heldout:
            self.heldout_w.write(lines, doc_id)
            return True
        return False

    def is_heldout(self, u):
        return u < self.f_sample + self.f_heldout

    def close(self):
        self.candidates.sort()
        taken = 0
        for u, doc_id, lines in self.candidates:
            if taken < self.sample_bytes:
                self.sample_w.write(lines, doc_id)
                taken += sum(len(l.encode("utf-8")) + 1 for l in lines)
            else:
                self.heldout_w.write(lines, doc_id)
        return {"heldout-sample.docs": self.sample_w.close(), "heldout.docs": self.heldout_w.close()}


# ---------------------------------------------------------------- readability

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
SENTENCE_END = re.compile(r"[.!?]+(?=[\s\"'”’)\]]|$)")


def words(text):
    return [w.lower() for w in WORD.findall(text)]


def readability(text, rank, unknown_rank):
    """Lower is easier: mean log2 word rank + log2 mean sentence length.

    Words are runs of letters (with inner apostrophes), lowercased. Rank 1 is
    the most frequent word of the frequency sample; words not in it get rank
    len(sample vocabulary) + 1. Sentences end at runs of . ! ? followed by
    whitespace, a closing quote/bracket, or the end. Adding the two logs means
    doubling the mean sentence length costs as much as doubling the typical
    word rank. Documents with no words score 99 (last)."""
    ws = words(text)
    if not ws:
        return 99.0
    mean_log_rank = sum(math.log2(rank.get(w, unknown_rank)) for w in ws) / len(ws)
    sentences = max(1, len(SENTENCE_END.findall(text)))
    return mean_log_rank + math.log2(max(1.0, len(ws) / sentences))


# ------------------------------------------------------------------- fineweb


def prep_fineweb(a):
    t0 = time.time()
    src = Path(a.data) / "fineweb-edu" / "sample-10BT"
    out = Path(a.out) / "fineweb"
    tmp = out / ".tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = pick_files(src, a.files)
    heldout_file = files[0]
    est = text_bytes_estimate(heldout_file)
    held = HeldOut(out, est, a.sample_mb, a.heldout_mb)
    print(f"held-out drawn from {heldout_file.name} (~{est / 1e9:.2f} GB of text)", flush=True)
    total = a.train_mb * MB
    freq_budget = a.freq_mb * MB
    freq = {}
    freq_bytes = 0
    cand_bytes = 0
    n_cand = 0
    skipped = 0
    cand_path = tmp / "candidates.jsonl"
    with open(cand_path, "w", encoding="utf-8") as cand:
        for path in files:
            is_heldout_file = path == heldout_file
            if cand_bytes >= total and not is_heldout_file:
                break
            for r in rows(path, ["id", "text"], a.max_rows):
                u = unit(hash64(a.seed, r["id"].encode()))
                held_out = is_heldout_file and held.is_heldout(u)
                if not held_out and cand_bytes >= total:
                    continue
                lines = doc_lines(r["text"])
                if lines is None:
                    skipped += 1
                    continue
                if held_out:
                    held.offer(u, r["id"], lines)
                else:
                    n_bytes = len(r["text"].encode("utf-8")) + 1
                    cand.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False))
                    cand.write("\n")
                    cand_bytes += n_bytes
                    n_cand += 1
                    if freq_bytes < freq_budget:
                        freq_bytes += n_bytes
                        for w in words(r["text"]):
                            freq[w] = freq.get(w, 0) + 1
            print(f"  read {path.name}: {n_cand} training docs, {cand_bytes / MB:.0f} MB "
                  f"[{time.time() - t0:.0f}s, peak RSS {peak_mb()} MB]", flush=True)
    t1 = time.time()
    print(f"pass 1 (held-out, candidates, word counts): {t1 - t0:.1f}s (peak RSS {peak_mb()} MB)", flush=True)

    vocab = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {w: i + 1 for i, (w, _) in enumerate(vocab)}
    unknown_rank = len(rank) + 1
    del freq, vocab

    scores = []
    with open(cand_path, encoding="utf-8") as cand:
        for line in cand:
            scores.append(readability(json.loads(line)["text"], rank, unknown_rank))
    t2 = time.time()
    print(f"pass 2 (scores): {t2 - t1:.1f}s (peak RSS {peak_mb()} MB)", flush=True)

    # Bucket by score quantiles (~bucket_mb each), then sort each bucket.
    n_buckets = max(1, math.ceil(cand_bytes / (a.bucket_mb * MB)))
    ordered = sorted(scores)
    bounds = [ordered[len(ordered) * i // n_buckets] for i in range(1, n_buckets)] if ordered else []
    del ordered

    def bucket_of(s):
        lo, hi = 0, len(bounds)
        while lo < hi:
            mid = (lo + hi) // 2
            if s < bounds[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    bucket_files = [open(tmp / f"bucket-{b:03d}.jsonl", "w", encoding="utf-8") for b in range(n_buckets)]
    with open(cand_path, encoding="utf-8") as cand:
        for line, s in zip(cand, scores):
            row = json.loads(line)
            row["score"] = s
            bucket_files[bucket_of(s)].write(json.dumps(row, ensure_ascii=False))
            bucket_files[bucket_of(s)].write("\n")
    for f in bucket_files:
        f.close()
    os.remove(cand_path)
    t3 = time.time()
    print(f"pass 3 (bucketing into {n_buckets}): {t3 - t2:.1f}s (peak RSS {peak_mb()} MB)", flush=True)

    shards = ShardWriter(out, "train", a.shard_mb * MB)
    for b in range(n_buckets):
        bp = tmp / f"bucket-{b:03d}.jsonl"
        with open(bp, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f]
        docs.sort(key=lambda d: (d["score"], d["id"]))
        for d in docs:
            lines = doc_lines(d["text"])
            if lines is not None:
                shards.write(lines, d["id"], d["score"])
        del docs
        os.remove(bp)
    train_stats = shards.close()
    shutil.rmtree(tmp)
    t4 = time.time()
    print(f"pass 4 (sort buckets, write shards): {t4 - t3:.1f}s (peak RSS {peak_mb()} MB)", flush=True)

    manifest = {
        "params": vars(a),
        "heldout_file": heldout_file.name,
        "skipped_docs": skipped,
        "freq_sample_bytes": freq_bytes,
        "vocabulary": unknown_rank - 1,
        "score_quantiles": {q: sorted(scores)[int(q * (len(scores) - 1))] for q in (0.0, 0.1, 0.5, 0.9, 1.0)} if scores else {},
        "files": {**held.close(), **train_stats},
        "seconds": {"pass1": t1 - t0, "scores": t2 - t1, "bucket": t3 - t2, "sort_write": t4 - t3, "total": t4 - t0},
    }
    (out / "prep.json").write_text(json.dumps(manifest, indent=2, default=str))
    report(manifest)


# ----------------------------------------------------------------- wikipedia


def prep_wikipedia(a):
    t0 = time.time()
    src = Path(a.data) / "wikipedia"
    out = Path(a.out) / "wikipedia"
    titles_path = Path(a.enwik_titles)
    if not titles_path.exists():
        sys.exit(f"{titles_path} is missing: run `sememe-data enwik --only enwik9` first")
    with open(titles_path, encoding="utf-8") as f:
        enwik_titles = {l.rstrip("\n") for l in f}
    files = pick_files(src, a.files)
    heldout_file = files[0]
    held = HeldOut(out, text_bytes_estimate(heldout_file), a.sample_mb, a.heldout_mb)
    est_all = sum(text_bytes_estimate(p) for p in files)
    # Uniform sample of the remaining articles, sized for the target.
    p_train = min(1.0, a.train_mb * MB / est_all)
    print(f"{len(enwik_titles)} enwik9 titles; held-out from {heldout_file.name}; "
          f"p(train)={p_train:.3f} of ~{est_all / 1e9:.2f} GB", flush=True)

    shards = ShardWriter(out, "train", a.shard_mb * MB)
    total = a.train_mb * MB
    train_bytes = 0
    removed_enwik = 0
    skipped = 0
    for path in files:
        is_heldout_file = path == heldout_file
        if train_bytes >= total and not is_heldout_file:
            break
        for r in rows(path, ["id", "title", "text"], a.max_rows):
            u = unit(hash64(a.seed, r["id"].encode()))
            held_out = is_heldout_file and held.is_heldout(u)
            if not held_out:
                if train_bytes >= total:
                    continue
                if r["title"] in enwik_titles:
                    removed_enwik += 1
                    continue
                # A second, independent hash for the training sample.
                if unit(hash64(a.seed + 1, r["id"].encode())) >= p_train:
                    continue
            lines = doc_lines(r["text"])
            if lines is None:
                skipped += 1
                continue
            if held_out:
                held.offer(u, r["id"], lines)
            else:
                shards.write(lines, r["id"])
                train_bytes += len(r["text"].encode("utf-8")) + 1
        print(f"  read {path.name}: {train_bytes / MB:.0f} MB training [{time.time() - t0:.0f}s]", flush=True)
    manifest = {
        "params": vars(a),
        "heldout_file": heldout_file.name,
        "enwik9_titles": len(enwik_titles),
        "removed_enwik9_articles": removed_enwik,
        "skipped_docs": skipped,
        "files": {**held.close(), **shards.close()},
        "seconds": time.time() - t0,
    }
    (out / "prep.json").write_text(json.dumps(manifest, indent=2, default=str))
    report(manifest)


def report(m):
    print(f"{'file':<24} {'bytes':>12} {'lines':>10} {'docs':>8}")
    for name, s in m["files"].items():
        print(f"{name:<24} {s['bytes']:>12} {s['lines']:>10} {s['docs']:>8}")
    print(f"skipped docs: {m['skipped_docs']}; seconds: {m['seconds']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/prepared")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fineweb", "wikipedia"):
        s = sub.add_parser(name)
        s.add_argument("--files", nargs="*", help="parquet file names (default: all, sorted); "
                       "held-out rows come from the first")
        s.add_argument("--sample-mb", type=float, default=1)
        s.add_argument("--heldout-mb", type=float, default=50)
        s.add_argument("--train-mb", type=float, default=1000)
        s.add_argument("--shard-mb", type=float, default=256)
        s.add_argument("--max-rows", type=int, help="read at most this many rows per file (for trying it)")
        s.add_argument("--seed", type=int, default=0)
    sub.choices["fineweb"].add_argument("--freq-mb", type=float, default=200,
                                        help="training text used for word frequencies")
    sub.choices["fineweb"].add_argument("--bucket-mb", type=float, default=64)
    sub.choices["wikipedia"].add_argument("--enwik-titles", default="data/prepared/enwik/enwik9-titles.txt")
    a = p.parse_args()
    if a.cmd == "fineweb":
        prep_fineweb(a)
    else:
        prep_wikipedia(a)


if __name__ == "__main__":
    main()
