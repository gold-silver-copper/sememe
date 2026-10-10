# Prompt: sememe, a compiler from text to a readable language model

## The idea

An LLM is trained by compressing text. Its loss is the number of bits needed
to write down the training text using the model's predictions, and training
searches for the weights that make the whole corpus cheapest to write down.

Build **sememe**: the same objective, but the model is a **program**
instead of weights. Instead of gradient descent, the search is done by
counting (EM), an Opus teacher that writes down what sentences mean, and
Opus agents that refactor the program:

> **total cost = size of the program + bits needed to write down all the
> training text using the program**

The cheapest way to write down a large corpus is to store each repeated thing
once: each word once (a lexicon), each sentence pattern once (a grammar), each
fact once however many ways it is phrased (a knowledge base), and each meaning
once across paraphrases and languages (a language-neutral ontology). So
minimizing this cost should *discover* a readable language model instead of
having one designed by hand. Text goes in; a Rust crate comes out.

Nothing runs an LLM at use time. The result is a program anyone can read:
its types are the concepts it knows, its rules are how it speaks and
understands, and every output comes with the trace that produced it. **The
learned program itself is committed to the repo as readable text**, so
people can read the model the way they read source code.

You own this fully: architecture, names, crates, the grammar formalism,
formats, tests and CLI are yours to decide. The design below is the intended
direction, not a spec. If measurements point somewhere better, follow them and
explain why in the final report. No prototypes: the work delivers a working
engine, a working training loop run at real scale, and the exported crate.

## Ground rules

This prompt is long. These are the rules that hold everywhere in it. Never
break them, whatever a later section seems to allow:

1. **Opus 5.5 only:** every LLM call is `claude -p --model claude-opus-5-5`,
   for every role. No smaller model, and no LLM at all at use time.
2. **No input from the user:** never ask them anything. Use their existing
   Claude Code login.
3. **Learned from raw text only:** no outside dictionaries, word lists,
   treebanks or annotated corpora, and no NLP or ML libraries. The only
   annotations are the ones the Opus teacher writes.
4. **Text only from meanings:** no path from text to text that skips the
   world model.
5. **The cost code is frozen** for the whole run (see "How cost is measured").
6. **Held-out text stays unseen:** never sent to the teacher, and unreadable
   by agents.
7. **The learned program is readable text,** committed to the repo, with real
   names and doc comments.
8. **Never write to `/tmp`; non-test code never panics.**
9. **PRs merge themselves** when CI is green.
10. **Everything you start stops when your session ends,** after a final
    checkpoint.

**You may improve this prompt as you work.** It lives in the repo at
`prompts/text-to-code-compiler.md`, and it's your working spec, not a frozen
contract. When you learn something it should say, edit it: a design that
measurably works better, a step that turned out wrong or impossible, a
missing detail a fresh session would need, a number that was a guess and is
now measured. Rules for doing so:

- **The ground rules above stay.** Don't loosen them. They are the user's
  decisions, not design guesses.
- **Changes to the objective or the cost code** take effect only for a new
  run, never in the middle of one (ground rule 5).
- **Say why.** Each edit goes in a commit whose message gives the reason and
  the evidence (a metric, a failed experiment, a measurement). Record it in
  the progress log too.
- **It ships like everything else:** prompt edits go into the PRs with the
  work that motivated them, and merge the same way.
- **Keep it a prompt, not a diary.** Results belong in the report and the
  progress log. The prompt says what to build and how. When you change it,
  rewrite the affected section cleanly instead of adding notes on top.

**Terms used throughout:**

| Term | Meaning |
|---|---|
| program | the learned model: lexicon, categories, combining rules in use, ontology, facts, inference rules, counts |
| engine | the fixed code that runs the program: parse, generate, reason, score |
| T | a stage's fixed training set; the objective is computed over all of it |
| stage | one rung of the data ladder (BabyLM, FineWeb-Edu, Wikipedia, UN corpus) |
| tier | fast (no LLM), medium (Opus teacher), slow (Opus agents) |
| round | one accepted program version, with one row of metrics |
| owner | the single process that applies every change to the program |

## The architecture: one generative program, run three ways

The model describes how text comes about:

```
P(text) = P(meaning) × P(text | meaning)
           world model   language
```

- **World model:** which meanings are likely. Typed concepts and relations in
  a hierarchy, facts made of them, and inference rules.
- **Language:** how a meaning becomes text in one language. One per language,
  English first.

A small, **fixed engine** runs the program three ways:

- **forward**, speaking: meaning → text;
- **backward**, understanding: text → the cheapest meaning that produces it;
- **inside**, reasoning: answer a question from facts and rules.

Like a transformer's architecture, the engine doesn't change during
training. Only the program does, the way only the weights change in an LLM.
The program is data the engine interprets, so a change takes effect instantly
with no recompiling. Keep its format as small and uniform as possible: the
format is the architecture.

### The language half: a lexicalized grammar

Start from Combinatory Categorial Grammar (CCG), or a close variant:

- **Fixed in the engine:** the *kinds* of combining rule (application,
  composition, type-raising and the like).
- **Listed in the program:** which combining rules are in use. Each costs
  program bits, so adding one is an ordinary scored change.
- **Learned, the lexicon:** almost everything about a language lives in word
  entries. Each entry pairs a word form with a category (what it combines
  with), a meaning (what it contributes, as a typed term), and a count.
- **Categories carry features**, e.g. `NP[num=pl]` or `S[fin]`. Without them,
  agreement ("the cats are" against "the cats is") and similar constraints
  have nowhere to live. The learner may split and merge categories and
  features as changes judged by cost.
- **Learned, word forms:** stems and endings ("walk" + "-ed", "child" +
  "-ren"), so related forms share one entry.

Putting everything learnable into word entries is what makes automatic
learning possible (see "The training loop"). It also suits more languages:
the same learner has induced CCG lexicons for very different languages
(Kwiatkowski et al. 2010). If measurements show a better formalism, switch,
and explain why in the report.

### The generative model, exactly

The papers that learn CCG from meanings mostly score a meaning *given* a
sentence. Sememe needs the other direction, a probability for the text
itself, so define it fully:

- **Meanings are typed terms:** trees built from concepts and relations,
  written in a small typed lambda-calculus. The engine has a type checker for
  them.
- **P(meaning) is a top-down generator.** At each node of the meaning tree,
  first choose *how* the slot is filled, then what fills it. Each choice has
  counts in the world model and costs −log2 of its estimated probability.
  The ways to fill a slot:
  - **describe it fresh:** choose a concept or relation of the slot's type,
    then its arguments;
  - **refer back:** choose an entity already in the context, weighted by
    recency;
  - **restate:** choose a fact already in the context or the global knowledge
    base;
  - **infer:** choose a fact that follows from known facts and inference rules.
    Inference is bounded (below), so the set of candidates is finite and the
    code stays valid.
- **P(text | meaning) is a generative CCG** (Hockenmaier & Steedman 2002;
  Kwiatkowski et al. 2012, which was trained on child-directed speech):
  - the derivation is generated top-down from the meaning;
  - each step chooses a combining rule and how to split the meaning, and
    ends in lexical entries;
  - only normal-form derivations are allowed (Eisner 1996), so no
    probability is wasted on equivalent derivations of the same reading.
- **State what every count is conditioned on**, e.g. a lexical choice given
  the category and the concept, or a rule given the parent category. That
  definition is part of the frozen cost code (below).
- **The code uses the single best path.** A line's cost is the bits of its
  cheapest (meaning, derivation) pair, the Viterbi path. That's a valid code
  and an upper bound on the true cost. Say so wherever bits are reported.
- **Search bounds** are part of the frozen cost code: maximum inference depth,
  maximum candidate facts per slot, and how far back the parser looks when
  searching for a referent. The model's context itself stays unbounded.

### The rules that make this work

1. **Text can only come from a meaning.** There is no path from text to text
   that skips the world model. That bottleneck is what makes the program
   semantic rather than a big table of word sequences.
2. **Unknown words are meanings too.** A word the program can't explain
   enters as an `Unknown` meaning. Each `Unknown`:
   - covers exactly one whitespace-separated token;
   - pays for its category from the learned category distribution;
   - pays for its slot type under P(meaning);
   - pays for its spelling under a fixed character model, frozen at the
     start and never trained (e.g. byte frequencies from a fixed sample of
     training text, plus an end marker).

   So an unknown word is never a free wildcard: it pays everything a real
   entry pays plus its spelling, and the rest of the sentence still parses
   and still teaches. The loss is defined from step zero, even for an empty
   program, and the bits spent on unknowns show exactly what to learn next.
3. **Real text is messy, so the model allows it, within limits.**
   - **Fragments:** a line may be several fragments, each with its own
     meaning and derivation, plus a fixed cost per fragment. A fragment
     boundary is allowed only at punctuation, a speaker turn or the
     sentence-length cap. Otherwise splitting every line into one-word
     fragments could become the cheapest path early, and lock in.
   - **A fixed formatting layer** turns tokens into exact bytes: case,
     spacing, punctuation attachment, line breaks and sentence boundaries.
     Its structure is fixed and its counts are learned. Bits per byte then
     cover every byte of the original text.
   - **Speaker tags** (`*CHI:`, `A:`, a leading `- ` in subtitles) introduce
     speakers as entities in the context, so "I" and "you" can refer to them.
4. **Context is part of the model.** A document is a sequence of meanings,
   and what has already been said makes later meanings cheaper (above). Reading
   is adding to the context. The same mechanism is how a conversation works,
   so context is a data structure with no fixed size limit.
5. **Choices carry counts on named things.** A frequency on a named rule or
   entry is readable. An anonymous vector of numbers is not, and isn't
   allowed.
6. **Everything gets a readable name.** Concepts and types get a real name and
   a doc comment from the teacher or agent that adds them. Entries the learner
   adds mechanically get generated IDs (word plus category), and the slow tier
   names and documents anything worth explaining. Names and docs cost nothing
   in the size measure.

**Keep meaning language-neutral.** Meaning types are named after concepts,
not English words. English-only distinctions (articles, English's tense set,
word order) belong in the English lexicon and grammar, not in the world model.
Represent what is meant instead (a known or new referent, when an event
happened relative to now, whether it's finished).

## How cost is measured

- **Text cost:** each line's bits through its cheapest path: the meaning, the
  derivation, the formatting, and any `Unknown`s.
- **Program cost:** serialize the program in its canonical text form (below)
  and count its bits with a fixed, stated code. For example, each symbol costs
  log2 of how many symbols could appear in that position. A new type costs
  its declared signature, each count has a stated cost, and names and doc
  comments are free. Grünwald's 2004 MDL tutorial (in `references/`) covers
  how to count model bits so they can't be gamed.
- **The cost code is frozen.** The program code, the conditioning of every
  count, the character model and the search bounds live in one module. A
  version hash of that module is recorded with every metric. Nothing changes
  it during a run, agents may not touch it, and CI fails if the hash changes
  without the run being restarted from scratch.
- **The training objective is exact.** Each stage defines a fixed training set
  T: all of BabyLM 10M, then 100M, and fixed samples for the large corpora
  (e.g. 1 GB of FineWeb-Edu). The objective is program bits plus data bits
  over all of T.
- **Scoring a change exactly.** A change is kept if it lowers that total.
  Compute the change in data bits by re-parsing every document in T it can
  affect, using an index from words, entries and concepts to the documents
  that use them. Because referring back links each line to the ones before
  it, re-parse each affected document from its first affected line to the
  end, and cap document length in T. No batch-to-corpus scaling: a saving
  measured on a sample and multiplied up turns rare words seen once into huge
  apparent gains, which accepts memorization.
  - **Changes that touch nearly everything** (function words, combining
    rules, core types) are scored on the full T, less often.
  - **Check that accepted changes generalize:** on a sample of accepted
    changes, measure the same change on held-out text, and report how often
    it still helps.
- **Bits as integers.** Store bits as fixed-point integers and sum them in
  line order, so results don't depend on thread scheduling.
- **The headline number is bits per byte on held-out text,** over the
  original bytes of the held-out files (anything the data preparation
  removes is coded too). It's the same yardstick used for LLMs and
  compressors (see "Fair comparisons").

## The training loop

**Every LLM call uses Claude Opus 5.5 through the Claude Code CLI**
(`claude -p --model claude-opus-5-5`), for every role: teacher and agents.
Never a smaller model. Use the user's existing Claude Code login. The work
needs no input from the user, so never ask them for credentials or anything
else.

Opus is the **teacher**: it reads a passage and writes down what it means.
Opus agents step in for what needs judgment. The arithmetic (parsing with the
current program, counting, scoring) is the program itself running, with no
model call. It runs at three speeds:

```
fast   (no LLM: the program itself, continuously)
       parse text with the current program → re-estimate every count with EM
       → learn new words from context → score total cost

medium (Opus as teacher, in bulk, parallel)
       the most expensive passages → the teacher writes their meaning
       → the learner proposes lexicon entries that produce it
       → keep the candidates that lower total cost

slow   (Opus agents, periodically)
       consolidate meaning types, generalize, name and document,
       prune, rebuild, fix what the learner can't express
```

### Fast: counts, and words learned from context

- **Counts by EM.** Expectation-maximization sets every count in the program
  with no LLM: how often each entry, word form, concept, fact, rule and
  formatting choice is used. Use Viterbi EM with a beam: count along each
  line's best paths. Each structural change is then judged with its best
  counts, not guessed ones. Together with the structure search, this is
  structural EM (Friedman 1998).
- **Words learned from context, without the teacher.** When a sentence parses
  except for one `Unknown` word, the rest of the parse fixes what category
  that word must have. The learner proposes a new entry for it, with a fresh
  concept of the type that slot needs. It's kept if total cost goes down (Bisk
  & Hockenmaier 2012 and Kwiatkowski et al. 2012 learn words this way).
  Without this, the lexicon could only grow as fast as Opus can annotate. The
  slow tier later names these concepts and merges them with existing ones
  where they turn out to be the same.
- **Word forms by compression.** Learn stems and endings by MDL segmentation
  (the Morfessor family: Creutz & Lagus).

### Medium: the teacher, and learning entries from meanings

- **What gets sent.** Only passages the program explains badly (high bits per
  byte, many unknowns) go to the teacher. As the program improves, the
  teacher is called less and less.
- **What the teacher writes.** For each passage, each sentence's meaning as a
  typed term in the current meaning types, with the same entity ID for every
  mention of the same thing ("Lily … she … the girl"). That's how reference
  gets learned.
  - **The ontology it sees:** while the ontology is small, all of it. Once
    it's large, the relevant slice (types used by the passage's known words
    and their neighbours in the hierarchy) plus a searchable index of type
    names and docs.
  - **New types:** it may propose a new type, with a name, a doc comment and
    a signature, when nothing fits. A new type's signature is paid in program
    bits like anything else.
- **A meaning guide keeps annotations consistent.** Separate calls will
  otherwise disagree about events against states, whether tense, number and
  definiteness appear in the meaning, and how modifiers attach. Inconsistent
  targets give many entries per word, none of which pays. So:
  - keep a versioned meaning-representation guide in the repo (event
    structure, language-neutral features for tense, number, definiteness),
    with few-shot examples taken from accepted annotations;
  - rewrite every annotation into a canonical form before learning from it;
  - double-annotate 5% of passages, and report agreement over time.
- **Validation.** Every annotation is type-checked against the ontology
  version it was written against. A rejected one goes back to the teacher
  with the error, up to 3 times, and is then dropped and counted.
- **Duplicate types.** Parallel teachers will invent the same concept under
  different names (`Girl`, `FemaleChild`). After every batch, merge proposed
  types with identical names and signatures mechanically, before scoring. The
  slow tier merges the subtler duplicates.
- **Cold start.** The ontology starts empty, so the first rounds create it from
  nothing. Expect heavy consolidation early, and run the slow tier more often
  then.
- **Learning entries from meanings.** From each (sentence, meaning) pair, the
  learner splits the meaning across the words and proposes the lexicon
  entries that would produce it (Zettlemoyer & Collins 2005; Kwiatkowski et
  al. 2010 and 2011). Candidates are scored exactly (above). The ones that
  lower total cost are kept.
- **Annotations are evidence, not truth.** An entry is kept only if it lowers
  total cost on real text, so a wrong or inconsistent annotation simply
  fails to pay.
- **Annotations are never edited.** Store the teacher's raw output with the
  ontology version, model and prompt it was written with, in a gitignored
  folder under `data/`. Type merges and renames are logged, and applied when
  annotations are read.

### Slow: agents for judgment

Headless Opus coding agents (`claude -p --model claude-opus-5-5`), each in its
own git worktree, at most 4 at once:

- **Consolidating meaning types:** merge duplicates and build the type
  hierarchy.
- **Generalizing:** replace many specific entries with one general entry or
  rule where it lowers total cost.
- **Naming and documenting** concepts, types and important entries.
  Readability is the point.
- **Pruning:** find rules that never fire, rules that fire for one line only
  (memorization), and near-duplicates. The loop also tries switching off each
  rule by its stable ID, and deletes it if total cost goes down.
- **Rebuilding:** removing one rule at a time can't undo a set of early
  mistakes that depend on each other. Periodically rebuild the lexicon from
  all stored annotations under the current ontology and categories, and keep
  the rebuild if it costs less in total.
- **Changing what the learner can't:** new combining rules (scored like any
  change), new kinds of inference rule, and engine fixes. An engine fix is
  accepted only if it builds, passes the tests, and gives identical bits on
  the frozen test sample when its new feature isn't used. The loop then
  restarts from its checkpoint.

### Measure the algorithm, don't assume it

Early on, on BabyLM 10M, run agents alone against the full three-speed loop
with the same token budget. "Agents alone" means agents that look at the most
expensive spans and edit the program directly, with no teacher or learner.
Compare held-out bits per byte, and report it. If agents alone win, say so
and rebalance.

### Facts, curriculum, and why this can work now

- **Facts.** Storing a fact from one document globally is usually not worth
  its bits. Storing one stated across many documents is. Let the cost decide
  what becomes global knowledge and what stays document context.
- **Curriculum.** Easiest corpus first, harder ones as held-out bits per byte
  level off (see "Data").
- **Why this can work now.** Earlier MDL grammar learning (de Marcken 1996,
  Grünwald) had the right objective, but searching blind was hopeless.
  Supervised CCG induction worked, but needed hand-written meanings. An Opus
  teacher supplies meanings at scale, words are also learned from context,
  and compression decides what is worth keeping.

## Running it: rounds, state and operations

### The program on disk

- **The source of truth is a canonical text format,** one item per line,
  sorted by stable ID, with its doc comment and the round that added it.
- **Structure and counts are separate files** (e.g. `program/*.sem` for
  structure, `program/counts.tsv` for counts), so a counting pass doesn't
  rewrite every line and diffs stay readable.
- **One owner process** holds the program and applies all changes as typed
  operations: add, remove, merge types, rename, set a rule. The learner and
  agents never write the program files directly.
  - An agent submits an operations file against a base version. The owner
    replays it on the current version, re-scores, and accepts or rejects.
  - Candidates accepted in the same round are re-scored together before the
    round commits, because two changes can each pass alone and fail together
    (two new entries for the same word).

### Rounds

- **A round** is one accepted program version: a counting pass, one teacher
  wave, candidate scoring, accepting. If a full counting pass over T is too
  slow to run every round, re-count only the documents affected by the
  round's changes, and do a full pass every few rounds and before every
  milestone. Each round appends one row to
  `runs/<run>/metrics.jsonl`:
  - round, stage, cost-code hash;
  - held-out bits per byte per stage, and BLiMP per phenomenon;
  - program bits;
  - tokens per tier, and lines parsed per second;
  - changes accepted and rejected.
- **"Stopped improving"** means held-out bits per byte (of the current
  stage's held-out set) fell by less than 0.5% over the last 10 rounds of the
  stage, and filtered BLiMP rose by less than 0.5 points over them. (A round
  on BabyLM 10M takes 35-40 minutes: exact scoring re-parses most of T
  several times. A 30-round window would mean 20 hours of no progress before
  moving up.)

### State, checkpoints, determinism

- **The checkpoint** is the program version, the operations log, the data
  position, the seeds and the token ledger. Keep a full snapshot every 10
  rounds and at milestones, and only the operations log in between, so disk
  use stays bounded.
- **Deterministic:** integer bits summed in line order, ties broken by stable
  ID, teacher results applied in batch order, and each round seeded with the
  base seed plus the round number.

### Calling Opus

- **All model calls go through one wrapper:**
  - one shared limit across tiers: at most 8 teacher calls and 4 agents at
    once;
  - a timeout per call (teacher about 10 minutes, agent about 2 hours), and
    backoff on errors;
  - a ledger row per call: tier, round, input, output and cache tokens,
    outcome.
- **Usage limits:** when a call reports a usage limit, pause all tiers until
  the reset time it reports, or poll hourly if it gives none.
- **Teacher calls** are plain single-turn requests: no tools, their own short
  system prompt, JSON output against a schema (check `claude --help` for the
  flags). Each call annotates about 20 passages, to spread the session
  start-up cost.
- **Agents** run with an explicit settings file: an allow-list of tools, and
  reading denied on the held-out folder. Never run them with permission
  checks skipped.
- **Tokens:** count input plus output tokens as the CLI reports them, with
  cache reads reported separately. Report tokens per byte of training text
  over time, split by tier; they should fall as the program learns.

### Running the loop inside the session

- **The loop lives and dies with your session.** It runs as a background
  process that you start, watch and stop, never as a system service, and
  nothing keeps running after the session ends.
- **The session is meant to run for a long time.** The user runs it with
  Claude Code's `/goal`, so keep working stage by stage without stopping to
  report or ask:
  - watch the metrics;
  - run the slow tier;
  - move up stages;
  - open the PRs;
  - fix what breaks.
- **Resumable:** if the session ends anyway, the next one resumes the loop
  from its last checkpoint and the progress log.
- **On start, the loop runs a self-test** with one tiny Opus call.
- **Machine caps:** at most 16 parser threads, cargo with `-j 8` and
  `--test-threads 8`, plus the call limits above.
- **How long:** the user has authorized running it as long as it keeps
  improving. Don't ask about time or token cost.
  - When a stage stops improving, move up (see "Data").
  - Stop when the last reached stage has stopped improving.
  - If you hit usage limits, wait and resume.

### Throughput

Parsing is where wall-clock time goes:

- **Bound the parser:** beam search and a cap on sentence length (longer
  lines are split into fragments at the cap).
- **Score samples, not everything:** each round, re-score fixed ~1 MB samples
  of each held-out set. Score the full held-out sets only at milestones (stage
  changes and the end).

## Data: a ladder from easiest to hardest

Raw text only: no outside dictionaries, word lists, treebanks or annotated
corpora, and no NLP or ML libraries. Everything about language is learned by
the loop. The only annotations are the ones the teacher writes.

**The data is already downloaded and verified.** It is in `data/` in the
repo, fetched by `data/fetch.sh` on 2026-10-09 (41 GB, every archive tested).
`data/README.md` records each source's version, size and license; keep it up
to date if you add data. Everything in `data/` except `fetch.sh` and
`README.md` is gitignored and never committed. If anything is missing, rerun
`data/fetch.sh` (it resumes).

What is there, still packed as downloaded:

- `babylm/train_10M.zip` (54 MB unpacked) and `train_100M.zip` (541 MB),
  plus `dev.zip` and `test.zip`. Each has one plain-text file per source:
  `childes` (child-directed speech), `bnc_spoken` and `switchboard` (spoken
  dialogue), `open_subtitles`, `gutenberg` (children's and older books) and
  `simple_wiki`. The archives are dated 2024-04-08: this is the same text as
  the 2024 BabyLM release. Say which release you compare against.
- `blimp/blimp-master.zip` (full BLiMP), and `blimp/babylm/`, BabyLM's own
  evaluation versions:
  - `blimp_filtered`: BLiMP restricted to BabyLM's vocabulary;
  - `supplement_filtered`: 5 extra tests.

  BabyLM's published scores use the filtered versions, so compare against
  them with the filtered versions.
- `fineweb-edu/sample-10BT/*.parquet` (14 files, 27 GB) and
  `wikipedia/*.parquet` (41 files, 11 GB): one document per row, text in a
  `text` column. FineWeb-Edu rows also carry an educational-quality score.
- `enwik/enwik8.zip` and `enwik9.zip`: the start of a 2006 English Wikipedia
  dump, as raw XML with markup.
- `unpc/en-ru.txt.zip` (12 GB unpacked): `UNPC.en-ru.en` and `UNPC.en-ru.ru`,
  aligned line by line.

**Preparing it** (unpacking, reading parquet, cleaning, splitting into lines
and documents, setting aside held-out text) is part of the data pipeline. It
may use ordinary non-language libraries: zip and parquet/Arrow readers in Rust,
or a small Python script with `pyarrow` in a virtualenv under `~/.cache/`. Put
prepared text in a gitignored folder under `data/`. Stream large files instead
of loading them whole: a runaway memory spike has crashed the user's terminal
before.

| Stage | Data (`data/…`) | What it should teach | Compare against |
|---|---|---|---|
| 1 | `babylm/`: strict-small (10M words), then strict (100M words). Child-directed speech, children's books, Simple Wikipedia, subtitles, dialogue | the grammar foundation, everyday words, conversation | the models BabyLM published for the same data, on filtered BLiMP |
| 2 | `fineweb-edu/`: the 10B-token sample, educational web pages, **easiest pages first** | real vocabulary, explanations, longer sentences | small open LLMs (SmolLM2, small Qwen models) on the same held-out pages |
| 3 | `wikipedia/` (2023-11-01, plain text) and `enwik/` | facts at scale, a real world model | gzip, xz, zstd, and the Hutter Prize record on enwik9 |
| 4 | `unpc/`: UN Parallel Corpus, English–Russian | a language-neutral meaning layer | an English+Russian program against two separate ones |

**Stage 1 lays the grammar foundation.** Train on BabyLM first: 10M words,
then 100M. The foundation counts as laid when both of these have stopped
improving:

- held-out bits per byte on BabyLM;
- BLiMP: sentence pairs that differ in one grammatical point ("the cats
  *are*" against "the cats *is*"). The model passes a pair when the
  grammatical sentence costs fewer bits. Report the score per phenomenon
  (agreement, questions, negation…), and don't move on while a basic
  phenomenon is clearly weak.

**Later stages build on it; they don't replace it.** Grammar is never fully
done: harder text brings constructions BabyLM rarely has, so the loop keeps
adding grammar. But:

- each later stage's training set T keeps a share of earlier stages' text;
- every stage's held-out sample, and BLiMP, are re-scored every round, and
  reported throughout;
- the BLiMP gate (below) protects basic grammar in every stage.

**Moving up:** go to the next stage when held-out bits per byte on the
current one has stopped improving. Within FineWeb-Edu, order pages by a simple
readability measure you compute yourself (for example sentence length and how
common the words are in the corpus), and start with the easiest.

**Bulk changes are fine.** Harder text brings far more vocabulary. The
learner, or an agent, may add thousands of entries in one change, as long as
total cost goes down.

**The second language** tests whether the meaning layer is really
language-neutral. Aligned translations make a joint program cheaper whatever
the ontology looks like (each Russian line can be coded given the English
line's meaning), so the test has to avoid that:

- **The real test:** train the joint program and the two separate ones on
  English and Russian text from *different* documents, and compare program
  bits plus held-out bits. A joint program that is clearly cheaper is
  evidence of shared meaning.
- **Report the aligned result separately,** as the cost of translation.

The UN corpus also exists in Chinese, Arabic, French and Spanish, aligned
across all six languages. Extending to those is a natural next step.

### Held-out text

- **Set it aside before training starts:** a fixed sample of each source, plus
  BabyLM's own `dev` and `test` sets.
- **Remove overlaps:** BabyLM 10M is a subset of 100M, so remove every
  held-out line from both. enwik9 is a 2006 Wikipedia dump and the training
  Wikipedia is from 2023, so remove enwik9's articles (by title) from the
  Wikipedia training data.
- **Keep it away from LLMs:** held-out text is never sent to the teacher, and
  it lives outside every agent's worktree, with reading it denied in the
  agents' settings.

### BLiMP

- **Split by paradigm, not by pair.** Pairs within one paradigm come from one
  template with shared words, so halves of a paradigm aren't independent.
  Within each phenomenon, some paradigms gate changes and the rest are only
  reported. Gating paradigms are never reported as results.
- **The gate, in every stage:** once per round, the round's merged changes are
  checked against the gating paradigms. The round is rejected if any
  phenomenon's gating accuracy drops by more than 2 points.
- **Unknowns and fragments distort it:**
  - when both sentences of a pair contain unknowns, the shorter one tends to
    win regardless of grammar;
  - when an ungrammatical sentence fails to parse, it falls back to
    fragments at a fixed cost, which caps how much worse it can look.

  So report the share of pairs with any unknown, the share of ungrammatical
  sentences split into more than one fragment, and the score on pairs with
  neither.

### Does it learn meaning, or relabelled words?

Compression alone doesn't guarantee real meanings. A program that gives every
word its own concept, with meaning trees that just mirror the derivation, can
satisfy rule 1 and still be a word list in disguise. Measure it:

- set aside a **meaning probe**: a few thousand sentences drawn from the
  training sources (not from the held-out text, which the teacher never sees),
  annotated by the teacher once, and removed from T. Report how often the
  program's cheapest meaning matches the teacher's (a Smatch-style overlap);
- report how often paraphrase pairs (the teacher can write them) get the same
  meaning;
- report the share of concepts with exactly one word form, over time.

### Fair comparisons

- **Compressors that may use the training text:** plain gzip, xz and zstd see
  none of it, so beating them on held-out text is easy. Also compare against
  zstd with a dictionary trained on the training text, and xz's cost of
  held-out text given the training text (compress training plus held-out,
  subtract training alone).
- **Sememe is teacher-distilled, not "strict".** BabyLM's strict tracks forbid
  outside knowledge, and Opus brings a lot of it, and has probably seen BLiMP
  and BabyLM. Label every BabyLM comparison that way.
- **The Hutter Prize comparison is total size:** the record is the whole
  compressed enwik9 plus the decompressor. Compare program bits plus data
  bits over all of enwik9, markup included. Separately report bits per byte
  on a cleaned-text version.
- **Comparison models run in a separate evaluation tool,** never inside
  sememe. Running small open models on the CPU there (e.g. with llama.cpp) is
  fine. They are what sememe is measured against, not part of training.

## Start from nothing

- Don't use, or read for ideas, the user's language crates (`english`,
  `english-core`, `english-phrase`, `english-ccg`, `church-slavonic`,
  `ruthenian`, anything in `~/code/linguistics/` or `~/code/english/`), or
  anything in `~/code/glassmind/` (a separate, earlier project).
- The seed is the engine, the kinds of combining rule, the formatting layer
  and the `Unknown` mechanism, with an empty lexicon and an empty ontology.
  Language knowledge enters only through accepted changes and EM counts. If
  the loop can't start from completely empty, add the smallest hand-written
  seed that unblocks it, mark it as seed, and report its size.

## Export: text in, Rust crate out

The trained program is exported to a Rust crate at `crates/sememe-model/`:

- **Rust types only for the ontology:** concepts become enums and structs
  with their doc comments, so the Rust compiler checks the type hierarchy.
- **Everything large is data:** the lexicon, counts, facts and rules are
  built by `build.rs` from the committed text program into a binary blob,
  loaded with `include_bytes!` and interpreted by the runtime. The blob is
  never committed. Hundreds of thousands of entries as Rust code would choke
  the compiler.
- **It depends only on a small runtime crate** (the engine). It's
  deterministic, has no LLM, and has no network access.
- **A test checks that the exported crate and the training engine give the
  same cost** on the fixture text.

The CLI on top of it:

- **talk:** read a passage into context, then answer simple questions about
  it ("who went to the market?") by reasoning over the facts;
- **explain:** show any output's trace: the words, the meaning, which entries
  and rules fired (with the round that added each), and the bits each part
  cost;
- **speak:** generate text from a meaning, or sample a short story from the
  world model, to show what it has learned;
- **translate:** once the second language is in, read in one language and
  speak in the other;
- **inspect:** browse the concepts, entries, rules and facts, with how often
  each is used.

## Research

`references/` holds related work, indexed in `references/README.md`. Most
relevant:

- **The training algorithm**, in the index's section "Learning a grammar from
  data":
  - CCG: Steedman & Baldridge's introduction;
  - generative CCG and normal-form derivations: Hockenmaier & Steedman 2002,
    Eisner 1996;
  - learning CCG from sentences paired with meanings: Zettlemoyer & Collins
    2005; Kwiatkowski et al. 2010, 2011 and 2012 (2012 learns from
    child-directed speech); Artzi & Zettlemoyer 2013;
  - learning CCG without meanings: Bisk & Hockenmaier 2012;
  - word forms by compression: Morfessor (Creutz & Lagus);
  - learning structure and counts together: structural EM (Friedman 1998);
  - the objective: language modeling is compression (Delétang et al. 2023),
    MDL language learning (de Marcken 1996), and counting model bits
    (Grünwald's 2004 tutorial).
- **DreamCoder and LILO:** growing a program by alternating solving and
  compressing, the model for the slow tier.
- **Grammatical Framework:** one typed meaning layer with a grammar per
  language.
- **Minimal Recursion Semantics, AMR and UMR:** meaning formats; MRS and UMR
  leave choices open. Useful for the meaning guide.
- **SHRDLU, Cyc and The Bitter Lesson:** what came before, and the case
  against this.

They are ideas to learn from, not code or data to reuse. Add anything else
relevant you find to the folder and the index.

## What gets delivered

- The engine (CCG combining rules, parser, generator, reasoner, formatting
  layer, `Unknown` mechanism), the program format and its type checker, the
  frozen cost code, and the owner process.
- The training loop with all three speeds, actually run on real text.
- The algorithm comparison: agents alone against the full loop on BabyLM 10M,
  with the same token budget.
- **The main result:** a chart of held-out bits per byte against Opus tokens
  spent, with stage changes marked and program size on the same timeline,
  next to the fair comparisons above. Does it keep falling as the loop spends
  more, or level off? Say it plainly.
- **The grammar result:** BLiMP per phenomenon over time (reported paradigms
  only), and filtered BLiMP against BabyLM's published models.
- **The meaning result:** agreement with held-out teacher meanings,
  paraphrase agreement, and the share of one-word concepts.
- The learned program, committed as readable text.
- The exported Rust crate and CLI, with real examples of talk, explain,
  speak and inspect.
- If reached: the English plus Russian result and translate.

**Build order:**
1. The engine, program format, type checker, formatting layer, `Unknown`
   mechanism, frozen cost code, owner process, and data pipeline, including
   the held-out splits and overlap removal. Check by hand that an empty
   program gives the expected all-unknown cost, and that a small hand-written
   lexicon lowers it as expected.
2. EM and learning words from context, then the teacher and learner, then
   the slow tier, then the algorithm comparison. Then the loop on BabyLM
   (stage 1) until the grammar foundation is laid.
3. Export and CLI.
4. The later stages in order: FineWeb-Edu, Wikipedia and enwik9, then the
   UN corpus.

**If something has to give,** cut from the bottom and say so in the report:
- **Must have:** steps 1–3, the main result chart, the BLiMP and meaning
  results, and the algorithm comparison.
- **Should have:** stage 2 (FineWeb-Edu), talk and explain on the exported
  crate.
- **Nice to have:** stage 3 (Wikipedia and enwik9), stage 4 and translate.

## How you work

- Never write to `/tmp` (it's RAM-backed). Use `~/.cache/claude-tmp/` or a
  gitignored folder in the repo.
- Non-test code must not panic: no `unwrap`, `expect`, indexing or `panic!`,
  enforced with workspace clippy `deny` lints, including in the exported
  crate.
- Few, meaningful tests, and no snapshot or exact-output tests. Useful
  properties:
  - what the engine speaks from a meaning, it understands back to the same
    meaning;
  - an empty program costs exactly the all-unknown cost;
  - the exported crate and the engine give the same cost;
  - replaying the operations log rebuilds the same program.

  CI uses a tiny committed text fixture, plus a stubbed teacher and a stubbed
  agent for one loop step, and checks the cost-code hash.
- Keep a progress log at `~/code/notes/projects/sememe/progress.md`
  (done, current numbers, decisions and why, next), updated after every
  milestone, so a fresh session can pick the work up.
- When your session ends, stop every process you started, including the
  training loop, and every tmux session, worktree and background shell. Leave
  a final checkpoint so the next session can resume.

## Repo, Git and PRs

- `~/code/sememe/` exists and holds `prompts/`, `references/` and `data/`.
  Its `.gitignore` keeps the downloaded papers and the data out of git; only
  `references/README.md`, `data/fetch.sh` and `data/README.md` are committed.
- **The learned program is committed as text at each milestone,** in
  `program/`. Keep each committed version under 50 MB; if it grows past that,
  commit the structure files only and say so. Per-round versions stay in a
  gitignored store.
- Create `gold-silver-copper/sememe` on GitHub. Push the skeleton, this
  prompt and CI to `main`, and do the work on a branch.
- **The run outlasts one PR, and PRs merge themselves.** The user wants every
  PR merged automatically as soon as it's ready, with no review step:
  - **Set up auto-merge once,** when creating the repo: enable it
    (`gh repo edit --enable-auto-merge`), and protect `main` so the CI checks
    are required (through `gh api`). Auto-merge then waits for green CI.
  - **First PR:** open it once stage 1 has stopped improving and export works.
    It includes the program at that point and the report so far. Turn on
    auto-merge right after opening it (`gh pr merge --auto --merge`).
  - **Follow-up PRs:** each later stage lands as its own PR, opened by you
    when that stage stops improving (committed program, updated report,
    metrics), with auto-merge turned on the same way.
  - **If CI fails,** the PR doesn't merge. Fix it on the same branch, and
    auto-merge completes once CI is green.
  - After each merge, the local checkout moves to the latest `main`.
- Don't publish to crates.io or create tags or releases.

## Final report

Written for the first PR, and updated in each follow-up PR:

- Repo URL and PR link.
- **The main result:** the bits-per-byte chart, how it compares on each fair
  comparison (compressors with and without the training text, BabyLM's
  models labelled as not like-for-like, small LLMs on FineWeb-Edu, the Hutter
  Prize record on enwik9), and whether it was still falling.
- **The grammar result:** BLiMP per phenomenon (reported paradigms), which
  phenomena were learned and which weren't, the shares of pairs with unknowns
  or fragments, and how the scores held up as later stages were added.
- **The meaning result:** agreement with held-out teacher meanings, paraphrase
  agreement, one-word concepts over time, and teacher self-agreement on the
  double-annotated passages.
- Which stages were reached, and when and why the loop moved up.
- **The algorithm:**
  - the comparison of agents alone against the full loop;
  - tokens per tier;
  - how many entries came from the teacher and how many from learning words
    in context;
  - how many annotations failed validation;
  - how often consolidation merged the teacher's types, and what rebuilds
    changed;
  - how often accepted changes still helped on held-out text;
  - parsing speed.
- What the program learned: how many concepts, entries, rules and facts; the
  most-used ones; examples of concepts it invented. Note anything that
  compresses well but isn't a concept a person would recognize.
- How much of the program is general rules and how much is tables (idioms,
  collocations, memorized phrases). That split is the central open question
  of the approach.
- One real `explain` trace, and real outputs from talk and speak.
- The second-language result, if reached.
- What limited progress, and what should come next.
- Where the loop stopped, and the command to resume it from its last
  checkpoint.
- How you checked it works in practice, and copy-pasteable commands starting
  with `cd ~/code/sememe` to:
  - talk to the exported crate about a passage;
  - explain an answer;
  - sample a story;
  - inspect a concept;
  - read the learned program;
  - run one round of training;
  - recompute held-out bits per byte.
