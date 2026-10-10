# sememe-data

Prepares sememe's training and evaluation text. Raw downloads live in `data/`
(see `data/README.md`); everything this crate writes goes to
`data/prepared/` (gitignored). Nothing here uses NLP libraries or word lists.

## The `.docs` format

Every prepared text file is a `.docs` file: UTF-8, documents are runs of
lines, and documents are separated by a line holding exactly U+001E (record
separator). Lines keep their exact source bytes minus the trailing `\n` (a
`\r` would stay). There is no separator after the last document and no empty
document. A line that is exactly U+001E cannot be represented; the writers
refuse it (none occurs in BabyLM).

**Original bytes** of a document = sum over its lines of (length + 1): each
line plus its newline. For the BabyLM held-out files this equals the source
file's size exactly (checked: concatenating the lines with `\n` reproduces
each `dev`/`test` file byte for byte). Use it as the denominator for bits per
byte.

Rust API (`sememe_data`):

```rust
pub struct Doc { pub id: String, pub lines: Vec<String> }   // id = "<file stem>:<doc index>"
pub fn line_bytes(doc: &Doc) -> u64;                         // original bytes
pub fn lines_bytes<S: AsRef<str>>(lines: &[S]) -> u64;
pub fn word_count(line: &str) -> u64;                        // whitespace words

impl DocReader<BufReader<File>> { pub fn open(path: &Path) -> anyhow::Result<Self>; }
impl<R: BufRead> DocReader<R> { pub fn new(reader: R, stem: impl Into<String>) -> Self; }
impl<R: BufRead> Iterator for DocReader<R> { type Item = anyhow::Result<Doc>; }   // streams

impl DocWriter {
    pub fn create(path: &Path) -> anyhow::Result<Self>;
    pub fn write_doc<S: AsRef<str>>(&mut self, lines: &[S]) -> anyhow::Result<()>;
    pub fn stats(&self) -> &FileStats;
    pub fn finish(self) -> anyhow::Result<FileStats>;
}
pub struct FileStats { pub bytes: u64, pub lines: u64, pub docs: u64, pub words: u64, pub sha256: String }
pub fn file_stats(path: &Path) -> anyhow::Result<FileStats>;
pub fn hash64(seed: u64, bytes: &[u8]) -> u64;   // first 8 bytes LE of sha256(seed LE ++ bytes)
```

`hash64` is the only hash used for choices (samples, probe, splits, blocks);
`tools/prep_parquet.py` implements the same function.

## Commands

From the repository root:

```sh
cd ~/code/sememe
cargo build --release -p sememe-data -j 8
./target/release/sememe-data babylm            # ~10 s, writes data/prepared/babylm/
./target/release/sememe-data blimp             # <1 s, writes data/prepared/blimp/
./target/release/sememe-data enwik             # enwik8 + enwik9: raw, titles, cleaned text
./target/release/sememe-data unpc              # one pass over 12 GB unpacked (streams the zip)
./target/release/sememe-data manifest data/prepared/fineweb   # stats of every .docs under a dir

# parquet sources (needs the venv: python3 -m venv ~/.cache/sememe-venv && ~/.cache/sememe-venv/bin/pip install pyarrow)
~/.cache/sememe-venv/bin/python tools/prep_parquet.py fineweb                          # all 14 files, 1 GB of training text
~/.cache/sememe-venv/bin/python tools/prep_parquet.py fineweb --files 000_00000.parquet  # first file only (what was run)
./target/release/sememe-data enwik --only enwik9   # wikipedia needs the enwik9 titles first
~/.cache/sememe-venv/bin/python tools/prep_parquet.py wikipedia
```

Options: `unpc --pool-mb 100 --heldout-mb 1 --aligned-mb 10 --block-lines 200
--max-lines N`; `prep_parquet.py <fineweb|wikipedia> --files … --train-mb 1000
--sample-mb 1 --heldout-mb 50 --shard-mb 256 --max-rows N` (and `--freq-mb
200 --bucket-mb 64` for fineweb, `--enwik-titles` for wikipedia).
`--max-rows`/`--max-lines` are for trying a stage on a slice.

## Stage 1: BabyLM (`data/prepared/babylm/`)

Sources: `childes`, `bnc_spoken`, `switchboard`, `open_subtitles`,
`gutenberg`, `simple_wiki`; one `.docs` per source in each folder.

- **Documents.** A new document starts at the first non-blank line after a
  run of blank (whitespace-only) lines, and for `simple_wiki` also at a header
  `= = = Title = = =`. Blank lines stay in the document they end. Documents
  are capped at 64 lines (longer runs become consecutive 64-line documents;
  most CHILDES/subtitle documents are cap-sized, since those files have few
  blank lines).
- **`dev/`, `test/`**: BabyLM's dev and test files, every line kept.
- **`dev-sample/`, `test-sample/`**: ~1 MB each. Each source gets a share
  proportional to its bytes in that split; documents are taken in order of
  `hash64(0, "<source>:<doc index>")` until the share is reached.
- **Overlap removal.** Every train line that also occurs in dev or test is
  removed from both training sets, but only lines with ≥ 6 whitespace words
  or ≥ 40 bytes (exact byte match, via a set of 64-bit line hashes).
- **`probe/`** (meaning probe): whole documents from train_10M (after overlap
  removal), in order of `hash64(0, "train10M/<source>:<k>")`, each source
  getting a share of 3000 worded lines (lines with at least one word)
  proportional to its bytes. Documents under 200 bytes are never picked (see
  below). Removal:
  - from train_10M: the probe documents themselves, by position;
  - from train_100M: every run of consecutive lines whose content equals a
    whole probe document (matched across train_100M's own document
    boundaries, with a 64-line delay buffer). This is exact and cheap; the
    200-byte minimum keeps tiny documents like "yeah." from matching
    everywhere;
  - from both: any other line that repeats a long probe line (same ≥ 6 words
    / ≥ 40 bytes rule), so a probe sentence doesn't survive elsewhere.
- **`train10M/`, `train100M/`**: the training sets after these removals.
  Documents left empty are dropped (only train_100M documents wholly
  covered by a probe match).
- **`manifest.json`**: bytes / lines / docs / words / sha256 per file,
  removal counts per source, and the checks below.

### Findings (2024-04-08 archives = the BabyLM 2024 text)

- **Encoding.** Every source file is valid UTF-8, has `\n` line endings (no
  `\r`), no U+001E, and ends with a newline. The pipeline errors on invalid
  UTF-8 rather than handling it.
- **train_10M ⊂ train_100M: yes.** All 12,882 sampled distinct train_10M
  lines (hash % 64 == 0; 7,454 of them long) occur in train_100M. Stronger:
  for every source, train_10M is an in-order subsequence of train_100M made
  of a few dozen contiguous chunks (e.g. childes starts at line 0 of the 100M
  file; 11–78 chunks per source). Every probe document was found exactly once
  in train_100M.
- **Dev/test vs train: no wholesale leakage, but many repeated formulaic
  lines.** Long lines removed: train_10M 7,353 (0.62% of lines), train_100M
  93,834 (0.81%), 79% of them CHILDES. They are mostly recurring formulas,
  not copied text: the CHILDES speaker tag (`*INV:\t`) counts toward the 40
  bytes, so `*INV:\twhat more can you tell me?` (1,730 times in train_100M)
  qualifies; in subtitles, `- What are you talking about?`. 80% of the
  matched CHILDES lines occur once in train_100M. Dev and test share 1,962 of
  test's 444,471 distinct long lines (0.4%), also formulaic.
- Removal counts per source (10M held-out / 100M held-out / probe docs,
  lines / 10M repeats / 100M repeats): childes 5062 / 74575 / 15, 872 / 7 /
  27; bnc_spoken 123 / 797 / 5, 311 / 4 / 4; switchboard 18 / 118 / 1, 64 /
  0 / 0; open_subtitles 1817 / 15374 / 10, 638 / 45 / 49; gutenberg 240 /
  2066 / 13, 832 / 1 / 15; simple_wiki 93 / 904 / 103, 577 / 43 / 55. The
  probe has 3,188 worded lines (gutenberg lines are whole paragraphs).

## BLiMP (`data/prepared/blimp/`)

- `filtered.jsonl` (59,875 pairs, 67 paradigms), `supplement.jsonl` (5,218,
  5 paradigms), `full.jsonl` (67,000, the original BLiMP from
  `blimp-master.zip`): one row per pair `{paradigm, phenomenon, good, bad,
  pair_id}`. BLiMP's `pair_id`/`pairID` is kept.
- The supplement files have different fields (`row`, `contrast`, `quad`,
  `template`, … and no paradigm/phenomenon), so paradigm = phenomenon = file
  stem and `pair_id` = row position. `row` repeats within a file, so it is not
  an id. 473 supplement pairs (qa_congruence, turn_taking) are **two-line
  dialogues**: `good`/`bad` contain `\n`.
- 7 pairs have `good == bad` in both filtered and full BLiMP (2 in
  `passive_1`, 5 in `principle_A_case_2`): a BLiMP bug; score them as ties or
  skip them.
- `split.json`: paradigm → `gating` | `reported`. Within each phenomenon,
  paradigms are sorted by `hash64(0, paradigm)` and the first `(n + 1) / 3`
  gate (none when n < 3): 19 gating, 48 reported, plus the 5 supplement
  paradigms, all reported. The split also applies to `full.jsonl` (same 67
  paradigm names).

## Later stages

- **FineWeb-Edu** (`tools/prep_parquet.py fineweb` → `data/prepared/fineweb/`).
  One row = one document; `text` split on `\n`. Held-out rows come from the
  first listed file, by `u = hash64(0, id)`: `heldout-sample.docs` (~1 MB:
  the lowest-`u` rows up to 1 MB) and `heldout.docs` (~50 MB). Training: the
  next 1 GB of non-held-out rows in file order, ordered easiest first by a
  readability score = mean log2 rank of its words + log2 mean sentence
  length (words = letter runs, lowercased, ranked by frequency in the first
  200 MB of training docs, unknown words rank last; sentences end at
  `.!?`). Sorting is external: scores are computed in one pass, cut into ~64
  MB buckets by score quantile, then each bucket is sorted in memory and
  written to `train-000.docs`, `train-001.docs`, … (~256 MB each). Every
  `.docs` has a `.ids` side file (`<row id>\t<score>` per document) and
  `prep.json` records parameters and timings.
  **First-file run:** 90 s total (pass over the 3.5 GB of text 22 s, scoring
  56 s, bucketing 7 s, sort+write 6 s), peak RSS 323 MB; held-out sample 1.0
  MB (222 docs), held-out 51 MB (10,611 docs), training 1.0 GB in 4 shards
  (209,650 docs). pyarrow's `iter_batches` grew the Arrow memory pool without
  bound on these files (2.5 GB peak); the script reads one row group at a
  time instead.
- **Wikipedia** (`… wikipedia` → `data/prepared/wikipedia/`). Same held-out
  scheme (from the first file). Training: a uniform sample (`hash64(1, id)`
  below the rate that gives `--train-mb`) of articles across all files,
  skipping every article whose title is a non-redirect page title in enwik9,
  in shards of 256 MB. Tested on slices only.
- **enwik** (`sememe-data enwik` → `data/prepared/enwik/`): `enwik8`,
  `enwik9` (the raw bytes, verified identical to the zip), `<name>-titles.txt`
  (non-redirect page titles, XML-unescaped) and `<name>-clean.docs`: one
  document per non-redirect page, the title then the text cleaned as
  documented in `src/enwik.rs` (XML unescape; drop comments, templates,
  tables, refs, math, categories, images, interlanguage links; keep link
  labels; drop bold/italic quotes, heading `=`, list markers; decode common
  HTML entities). On enwik8: 12,346 pages, 3,664 redirects, 73 MB of clean
  text, in 1 s. A few rare leftovers remain (unusual entities, raw `<`).
- **UNPC** (`sememe-data unpc` → `data/prepared/unpc/`). The corpus is cut
  into 200-line aligned blocks; by `hash64(0, "unpc:<block>")` each block
  goes to exactly one of: `en-heldout`, `ru-heldout` (~1 MB, one language
  each), the aligned sample `aligned-en`/`aligned-ru` (~10 MB of English;
  document i of each is the same block), or (by a second hash) the English
  pool `en-train` or the Russian pool `ru-train` (~100 MB each, sampled
  uniformly over the whole corpus). No block appears in both languages
  outside the aligned sample. Tested on a 2M-line slice (2 s).
  **Surprise:** OPUS's Moses export is XML-escaped and partly tokenized:
  there is not a single literal `'` in the first 3M English lines; apostrophes
  appear as `Finland &apos; s`, quotes as `&quot;`. Lines are un-escaped
  (`Finland's`, `d'Ivoire`, `States' rights`, `"`), so UNPC text is not
  byte-identical to the zip. The zip also has `UNPC.en-ru.xml`, the
  alignment file with real document ids (`<linkGrp fromDoc=…>`); splitting by
  real documents from it would be a better unit than 200-line blocks.
