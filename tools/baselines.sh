#!/usr/bin/env bash
# Compressor baselines on held-out text, in bits per original byte.
# usage: tools/baselines.sh <name> <train .docs dir or file> <held-out .docs dir or file> [out.json]
# Plain gzip/xz/zstd see no training text; zstd-dict uses a dictionary
# trained on the training text; xz-cond is xz(train + held-out) - xz(train).
set -euo pipefail
name=$1; train=$2; held=$3; out=${4:-results/baselines-$name.json}
tmp=${TMPDIR_SEMEME:-$HOME/.cache/claude-tmp/baselines-$name}
mkdir -p "$tmp"
raw() { if [ -d "$1" ]; then cat "$1"/*.docs; else cat "$1"; fi | grep -v $'^\x1e$' || true; }
raw "$train" > "$tmp/train.txt"
raw "$held" > "$tmp/held.txt"
n=$(stat -c %s "$tmp/held.txt")
bpb() { python3 -c "print(round($1*8/$n, 4))"; }
g=$(gzip -9 -c "$tmp/held.txt" | wc -c)
x=$(xz -9e -T1 -c "$tmp/held.txt" | wc -c)
z=$(zstd -19 -q -c "$tmp/held.txt" | wc -c)
# zstd dictionary trained on the training text (cut into 4 KB samples).
split -b 4096 -a 6 "$tmp/train.txt" "$tmp/s_"
ls "$tmp"/s_* | shuf -n 20000 --random-source=<(yes) > "$tmp/samples"
zstd -q --train $(cat "$tmp/samples") --maxdict=1000000 -o "$tmp/dict" >/dev/null
rm -f "$tmp"/s_*
zd=$(zstd -19 -q -D "$tmp/dict" -c "$tmp/held.txt" | wc -c)
dsz=$(stat -c %s "$tmp/dict")
xt=$(xz -9e -T1 -c "$tmp/train.txt" | wc -c)
xb=$(cat "$tmp/train.txt" "$tmp/held.txt" | xz -9e -T1 -c | wc -c)
cat > "$out" <<JSON
{"name": "$name", "heldout_bytes": $n,
 "gzip_9": $(bpb $g), "xz_9e": $(bpb $x), "zstd_19": $(bpb $z),
 "zstd_19_dict_from_train": $(bpb $zd), "zstd_dict_bytes": $dsz,
 "xz_conditional_on_train": $(bpb $((xb - xt)))}
JSON
cat "$out"
rm -rf "$tmp"
