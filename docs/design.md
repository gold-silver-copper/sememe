# sememe: design

sememe learns a language model that is a readable program. Training
minimizes

> program bits + bits of the training text written down with the program

over a fixed training set T. This document says what the program is, how
bits are counted, and how the loop searches.

## The program

Text files in `program/`, one item per line with a stable ID, its doc
comment and the round that added it:

| File | Holds |
|---|---|
| `ontology.sem` | concepts (`c12 concept Girl < Person`), roles (`agent : Event -> Thing`), meaning features (`time : Event = past \| now \| future`) |
| `grammar.sem` | atomic categories (`S`, `NP`, `N`, ...), category features, the combining rules in use, inflections |
| `lexicon.sem` | entries: `e5 entry "see" (S\NP)/NP : \x y. See(experiencer: y, stimulus: x)`, and proper-name entries |
| `knowledge.sem` | named individuals, global facts, inference rules |
| `counts.tsv` | the stored counts, quantized, per table and context |

The **seed** (marked `seed`) is the smallest start that lets the loop run:
the top concept `Thing`, a `Speaker` concept for conversation
participants, three roles for unknown words (`arg`, `mod`) and
coordination (`op`), the atomic categories `S`, `NP`, `N`, and forward and
backward application. Everything else is learned.

## The engine (`crates/sememe-engine`)

The engine is fixed during a run; only the program changes.

- **Meanings** are trees of nodes (a concept, features, role children,
  an optional name), built by a small typed lambda calculus with de Bruijn
  variables (`term.rs`). Equal meanings are structurally equal: children
  with the same role are kept in a canonical order.
- **Categories** are CCG categories with features and feature variables
  (`category.rs`); combining rules: application, composition (forward,
  backward, crossed), type raising, type change and coordination. Only the
  rules listed in the program are used. Derivations are kept in Eisner
  normal form.
- **Backward (understand)**, `parse.rs`: tokenize, build leaves for every
  word (entries, inflected stems, proper names, unknowns), run a CKY chart
  within sentences with a beam, choose fragments by dynamic programming,
  then rescore the chosen analysis exactly, top-down, in decoding order,
  updating the discourse context.
- **Forward (speak)**, `generate.rs`: head-driven realization from a
  meaning, checked by parsing the words back.
- **Inside (reason)**, `reason.rs`: answer questions by matching a
  question meaning (with a `Query` node) against context facts, global
  facts and one step of inference rules.

## The code (`cost.rs`, frozen per run)

A line's bits are the code length of its cheapest analysis found (a
Viterbi path: a valid code and an upper bound on the model's cost). The
analysis is coded in decoding order:

1. **Formatting** (`text.rs`): speaker tag, then for each word its letter
   case, and the exact text between words (gaps), so every byte is covered.
   Words split at whitespace, punctuation and before an internal apostrophe
   (`it` + `'s`).
2. **Fragments**: a line is a sequence of fragments; a boundary only at
   punctuation, a speaker turn or the 12-word cap.
3. **World model** `P(meaning)`, per node: how the slot is filled (fresh,
   refer back by recency rank, restate an earlier meaning, infer), then for a
   fresh node its concept given role and parent concept, whether it carries
   a name (then spelled), its features and its role children as short
   "next" sequences.
4. **Language** `P(derivation | meaning)`, top-down: the root category;
   at each node how it expands (a leaf or a rule), the category the rule
   introduces, and how the meaning splits between the children (which role
   or feature the functor contributes); at a leaf the inflection and the
   entry given its category and meaning. An unknown word is the escape at
   the leaf: it pays its category, its slot under the world model, and its
   spelling under a fixed character model.

Every choice is a count table with a back-off chain (for concepts, up the
type hierarchy); a level without the outcome escapes to the next; the last
level is a fixed base distribution. Counts are quantized (exact to 16, then
eight levels per doubling) and cost bits in the program code (keys plus
Elias-gamma levels). Bits are fixed-point integers summed in line order.

The module's hash (with `charmodel.tsv` and `text.rs`) is recorded with
every metric; `runs/current-run.json` names the active run's hash and CI
fails if it changes. `tests/fixture/expected-bits.txt` guards the rest of
the engine: an engine change must keep the fixture's bits identical.

## The loop (`crates/sememe`)

One owner process (`trainer.rs`) holds the program and applies every change
as typed operations (`ops.rs`), logged so the log replays to the same
program. A change is scored exactly: apply it to a copy, re-parse every
training document it can affect (found through a word index), alternate
parsing and recounting two or three times (Viterbi EM: a change is judged
with its best counts), and compare the total over all of T.

- **Fast tier**: full EM passes; words learned from context (a line that
  parses except for one unknown word fixes that word's category: a new
  entry with a fresh concept); inflections that make full forms
  derivable; pruning rarely used entries.
- **Medium tier**: the Opus teacher annotates the passages the program
  explains worst, in the compact notation of `docs/meaning-guide.md`.
  Annotations are validated, retried with their errors, stored raw, and
  turned into candidate entries by GENLEX (`genlex.rs`), which reads the
  annotation's word anchors as a dependency structure and converts it to a
  CCG derivation.
- **Slow tier**: Opus coding agents in git worktrees (`agents.rs`) read the
  program and usage tables and write operations (merge and name concepts,
  build the hierarchy, generalize), which the owner scores like any change.

Changes are accepted in bulk where possible (and bisected if a bulk does
not pay). Each round ends with the BLiMP gate (gating paradigms may not drop
more than 2 points per phenomenon), held-out scoring, meaning metrics, a
metrics row and a checkpoint.

## Measured decisions

- Judging a change after one parse with guessed counts made a good bulk
  change (1,567 teacher entries) look 0.3% worse; with three EM iterations
  on the affected documents it was 11% better. Changes are judged after EM
  iterations.
- A 12-word fragment cap was both faster (2.6x) and cheaper in bits than a
  24-word cap at this stage of learning.
- Teacher calls at medium effort: at default effort about 80% of output
  tokens were thinking.
