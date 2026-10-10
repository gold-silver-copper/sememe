#!/usr/bin/env python3
"""Charts for the report from a run's metrics and ledger.

usage: tools/report.py <run> [--out results/<run>] [--baselines results/baselines-*.json ...]
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out")
ap.add_argument("--baselines", nargs="*", default=sorted(glob.glob("results/baselines-*.json")))
a = ap.parse_args()
out = a.out or f"results/{a.run}"
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(f"runs/{a.run}/metrics.jsonl") if l.strip()]
ledger = [json.loads(l) for l in open(f"runs/{a.run}/ledger.jsonl") if l.strip()] if os.path.exists(f"runs/{a.run}/ledger.jsonl") else []

def tokens_until(ts):
    return sum(r["input"] + r["cache_write"] + r["output"] for r in ledger if r["ts"] <= ts)

xs = [tokens_until(r["ts"]) / 1e6 for r in rows]
sets = sorted({k for r in rows for k in r["heldout_bpb"]})
fig, ax = plt.subplots(figsize=(9, 5.5))
for s in sets:
    ax.plot(xs, [r["heldout_bpb"].get(s) for r in rows], marker="o", ms=3, label=f"sememe held-out ({s})")
for bf in a.baselines:
    b = json.load(open(bf))
    for key, style in [("gzip_9", ":"), ("xz_9e", "--"), ("zstd_19_dict_from_train", "-."), ("xz_conditional_on_train", (0, (1, 1)))]:
        if key in b:
            ax.axhline(b[key], ls=style, lw=1, color="gray")
            ax.text(xs[-1] if xs else 0, b[key], f" {key} ({b['name']})", fontsize=7, va="bottom", ha="right", color="gray")
prev = None
for x, r in zip(xs, rows):
    if prev is not None and r["stage"] != prev:
        ax.axvline(x, color="red", lw=0.8)
        ax.text(x, ax.get_ylim()[1], f" {r['stage']}", color="red", fontsize=8, va="top")
    prev = r["stage"]
ax.set_xlabel("Opus tokens spent (millions, input + output)")
ax.set_ylabel("held-out bits per byte (Viterbi upper bound)")
ax2 = ax.twinx()
ax2.plot(xs, [r["program_bits"] / 8 / 1024 for r in rows], color="tab:green", lw=1, alpha=0.6, label="program size (KiB)")
ax2.set_ylabel("program size (KiB)", color="tab:green")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
ax.set_title(f"sememe run {a.run}: held-out bits per byte against Opus tokens")
fig.tight_layout(); fig.savefig(f"{out}/main.png", dpi=130)

fig, ax = plt.subplots(figsize=(9, 5.5))
rounds = [r["round"] for r in rows]
phen = sorted({k for r in rows for k in r["blimp"]})
for p in phen:
    ax.plot(rounds, [r["blimp"].get(p) for r in rows], lw=1, label=p)
ax.plot(rounds, [r["blimp_overall"] for r in rows], lw=2.5, color="black", label="overall (reported paradigms)")
ax.axhline(50, color="gray", ls=":", lw=1)
ax.set_xlabel("round"); ax.set_ylabel("accuracy (%)"); ax.set_title("BLiMP (filtered, reported paradigms) per phenomenon")
ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f"{out}/blimp.png", dpi=130)

fig, ax = plt.subplots(figsize=(9, 4.5))
for key, lab in [("probe_f1", "probe agreement (F1 with teacher)"), ("paraphrase_same", "paraphrase pairs with the same meaning"), ("one_word_concepts", "share of one-word concepts"), ("teacher_agreement", "teacher self-agreement")]:
    ax.plot(rounds, [r.get("meaning", {}).get(key) for r in rows], marker=".", label=lab)
ax.set_xlabel("round"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title("Does it learn meaning?")
fig.tight_layout(); fig.savefig(f"{out}/meaning.png", dpi=130)

last = rows[-1] if rows else {}
summary = {"rounds": len(rows), "last": last, "tokens_total": tokens_until(1e18),
           "tokens_by_tier": {t: sum(r["input"] + r["cache_write"] + r["output"] for r in ledger if r["tier"] == t) for t in {r["tier"] for r in ledger}}}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print(f"wrote {out}/main.png, blimp.png, meaning.png, summary.json")
