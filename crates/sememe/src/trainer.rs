//! The owner process and the training loop.
//!
//! One `Trainer` holds the program and applies every change. A change is a
//! list of typed operations; it is scored exactly: apply it to a copy of
//! the program, re-parse every training document it can affect, recount
//! (the M-step over the counts with those documents replaced) and compare
//! program bits plus data bits over all of the training set. It is kept if
//! the total goes down.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use sememe_engine::cost::{self, Bits, COST_HASH, TAG_ENTRY, Table, float};
use sememe_engine::model::Model;
use sememe_engine::program::Program;
use sememe_engine::term::{Head, Node, Term};
use sememe_engine::text::Line;
use serde::{Deserialize, Serialize};

use crate::analyze::{self, DocResult, Single};
use crate::annotate::{self, Batch, Passage};
use crate::blimp;
use crate::corpus::Corpus;
use crate::em::{self, Stats};
use crate::genlex;
use crate::learn;
use crate::ops::{self, Logged, Op};
use crate::opus::Opus;
use crate::store;

/// Settings for a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Run name (directory under `runs/`).
    pub run: String,
    /// Stage name.
    pub stage: String,
    /// Training set.
    pub train: PathBuf,
    /// Held-out samples scored every round: name -> path.
    pub heldout: BTreeMap<String, PathBuf>,
    /// Prepared BLiMP directory.
    pub blimp: PathBuf,
    /// Teacher calls per wave.
    pub calls: usize,
    /// Passages per call.
    pub passages_per_call: usize,
    /// Lines per passage (at most).
    pub lines_per_passage: usize,
    /// Full EM pass every this many rounds.
    pub em_every: u32,
    /// Stop after this round.
    pub max_rounds: u32,
    /// Use the teacher (false: words from context and EM only).
    pub teacher: bool,
    /// Run the slow tier every this many rounds (0: never).
    pub slow_every: u32,
    /// Slow-tier tasks.
    pub slow_tasks: Vec<crate::agents::Task>,
    /// The comparison baseline: agents alone edit the program each round
    /// (no teacher, no learning from context, no pruning); EM still counts.
    #[serde(default)]
    pub agents_alone: bool,
}

/// What is saved between sessions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    /// Last completed round.
    pub round: u32,
    /// Stage.
    pub stage: String,
    /// Cost-code hash the run was started with.
    pub cost_hash: String,
    /// Lines already sent to the teacher: `doc:line`.
    pub annotated: BTreeSet<String>,
    /// Base seed.
    pub seed: u64,
}

/// The result of scoring a change.
pub struct Eval {
    /// The changed program with recounted counts.
    pub program: Program,
    /// Documents re-parsed.
    pub ids: Vec<usize>,
    /// Their new analyses.
    pub results: Vec<DocResult>,
    /// New total minus old total (negative is better).
    pub delta: i128,
    /// New program bits.
    pub program_bits: Bits,
    /// New data bits.
    pub data_bits: Bits,
}

/// The owner and its loop.
pub struct Trainer {
    /// Settings.
    pub cfg: Config,
    /// Run directory.
    pub dir: PathBuf,
    /// Saved state.
    pub state: State,
    /// The program.
    pub program: Program,
    /// The training set.
    pub corpus: Corpus,
    /// Analyses of the training set.
    pub stats: Stats,
    /// Word form -> documents containing it.
    pub index: FxHashMap<String, Vec<u32>>,
    /// Word form frequencies in the training set.
    pub freq: FxHashMap<String, u64>,
    /// Program bits.
    pub program_bits: Bits,
    /// Data bits.
    pub data_bits: Bits,
    /// Single-unknown lines from the last full pass.
    pub singles: Vec<(usize, Single)>,
    /// Model calls.
    pub opus: Opus,
    /// The meaning guide.
    pub guide: String,
    /// BLiMP pairs.
    pub blimp: Vec<blimp::Pair>,
    /// Held-out sets.
    pub heldout: BTreeMap<String, Corpus>,
    /// Last gating scores.
    pub gate: Option<blimp::Results>,
    /// Parse speed of the last full pass.
    pub lines_per_sec: f64,
    /// Probe annotations (teacher meanings of held-aside training lines).
    pub probe: Vec<Batch>,
    /// Paraphrase pairs written by the teacher.
    pub paraphrases: Vec<crate::meaning::Paraphrase>,
    /// Accepted changes checked on held-out text this round, and how many
    /// still helped there.
    pub generalize: (u32, u32),
}

/// A metrics row.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// Round.
    pub round: u32,
    /// Stage.
    pub stage: String,
    /// Cost-code hash.
    pub cost_hash: String,
    /// Unix seconds.
    pub ts: u64,
    /// Held-out bits per byte by set.
    pub heldout_bpb: BTreeMap<String, f64>,
    /// Held-out unknown-word rate by set.
    pub heldout_unknown: BTreeMap<String, f64>,
    /// Training data bits per byte.
    pub train_bpb: f64,
    /// Program bits.
    pub program_bits: f64,
    /// Data bits.
    pub data_bits: f64,
    /// BLiMP reported accuracy by phenomenon.
    pub blimp: BTreeMap<String, f64>,
    /// BLiMP overall (reported, filtered).
    pub blimp_overall: f64,
    /// BLiMP on pairs with no unknowns or fragments.
    pub blimp_clean: f64,
    /// Share of pairs with an unknown word.
    pub blimp_unknown_share: f64,
    /// Share of pairs whose bad sentence split into fragments.
    pub blimp_fragment_share: f64,
    /// Tokens by tier: input+output, and cache reads.
    pub tokens: BTreeMap<String, (u64, u64)>,
    /// Lines parsed per second.
    pub lines_per_sec: f64,
    /// Changes accepted this round.
    pub accepted: u32,
    /// Changes rejected this round.
    pub rejected: u32,
    /// Program size counts.
    pub entries: usize,
    /// Concepts.
    pub concepts: usize,
    /// Rules.
    pub rules: usize,
    /// Training unknown-word rate.
    pub unknown_rate: f64,
    /// Teacher units annotated this round.
    pub units: u32,
    /// Units that failed validation.
    pub units_dropped: u32,
    /// Units GENLEX could not convert.
    pub units_unconverted: u32,
    /// Entries added from the teacher, from context, removed by pruning.
    pub from_teacher: u32,
    /// Entries added from context.
    pub from_context: u32,
    /// Entries pruned.
    pub pruned: u32,
    /// Meaning metrics.
    pub meaning: crate::meaning::MeaningMetrics,
    /// Accepted changes checked on held-out text, and how many helped there.
    #[serde(default)]
    pub generalize: (u32, u32),
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn word_index(c: &Corpus) -> (FxHashMap<String, Vec<u32>>, FxHashMap<String, u64>) {
    let mut index: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    let mut freq: FxHashMap<String, u64> = FxHashMap::default();
    for (i, d) in c.docs.iter().enumerate() {
        let mut seen: FxHashSet<String> = FxHashSet::default();
        for l in &d.lines {
            for w in Line::parse(l).words {
                *freq.entry(w.form.clone()).or_default() += 1;
                seen.insert(w.form);
            }
        }
        for w in seen {
            index.entry(w).or_default().push(i as u32);
        }
    }
    (index, freq)
}

impl Trainer {
    /// Start a new run from the seed program, or resume one.
    pub fn open(cfg: Config) -> Result<Trainer> {
        let dir = PathBuf::from("runs").join(&cfg.run);
        std::fs::create_dir_all(&dir)?;
        let state_path = dir.join("state.json");
        let (state, program) = if state_path.exists() {
            let state: State = serde_json::from_str(&std::fs::read_to_string(&state_path)?)?;
            if state.cost_hash != COST_HASH {
                anyhow::bail!(
                    "run {} was started with cost code {} but this build has {}: the cost code is frozen for a run",
                    cfg.run,
                    state.cost_hash,
                    COST_HASH
                );
            }
            let p = store::load(&dir.join("program"))?;
            eprintln!("resuming run {} at round {}", cfg.run, state.round);
            (state, p)
        } else {
            (
                State {
                    round: 0,
                    stage: cfg.stage.clone(),
                    cost_hash: COST_HASH.to_string(),
                    ..State::default()
                },
                Program::seed(),
            )
        };
        std::fs::write(dir.join("pid"), std::process::id().to_string())?;
        let corpus =
            Corpus::load(&cfg.train).with_context(|| format!("loading {}", cfg.train.display()))?;
        let (index, freq) = word_index(&corpus);
        let mut heldout = BTreeMap::new();
        for (k, path) in &cfg.heldout {
            heldout.insert(k.clone(), Corpus::load(path)?);
        }
        let blimp = blimp::load(&cfg.blimp)?;
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let opus = Opus::new(
            &dir.join("ledger.jsonl"),
            &PathBuf::from(home).join(".cache/claude-tmp/sememe-calls"),
        )?;
        let guide = std::fs::read_to_string("docs/meaning-guide.md")?;
        let mut t = Trainer {
            cfg,
            dir,
            state,
            program,
            corpus,
            stats: Stats::default(),
            index,
            freq,
            program_bits: 0,
            data_bits: 0,
            singles: Vec::new(),
            opus,
            guide,
            blimp,
            heldout,
            gate: None,
            lines_per_sec: 0.0,
            probe: Vec::new(),
            paraphrases: Vec::new(),
            generalize: (0, 0),
        };
        t.load_meaning_sets();
        t.full_em(1)?;
        Ok(t)
    }

    fn docs(&self) -> Vec<&sememe_data::Doc> {
        self.corpus.docs.iter().collect()
    }

    /// Re-parse the whole training set and recount, `iters` times.
    pub fn full_em(&mut self, iters: usize) -> Result<()> {
        for _ in 0..iters {
            let t = Instant::now();
            let model = Model::new(Arc::new(self.program.clone()));
            let docs = self.docs();
            let mut stats = Stats::default();
            let mut singles = Vec::new();
            let mut base = 0usize;
            for chunk in docs.chunks(4096) {
                let results = analyze::analyze(&model, chunk, false);
                for (k, mut r) in results.into_iter().enumerate() {
                    for s in r.singles.drain(..) {
                        singles.push((base + k, s));
                    }
                    stats.push(&self.program, r);
                }
                base += chunk.len();
            }
            let secs = t.elapsed().as_secs_f64();
            self.lines_per_sec = self.corpus.lines() as f64 / secs.max(1e-9);
            self.program.counts = em::m_step(&self.program, &stats);
            self.stats = stats;
            self.singles = singles;
            self.recompute();
            eprintln!(
                "full pass: {:.0}s ({:.0} lines/s), train {:.4} bpb (program {:.4} + data {:.4}), unknown {:.1}%",
                secs,
                self.lines_per_sec,
                self.total_bpb(),
                float(self.program_bits) / self.stats.bytes() as f64,
                float(self.data_bits) / self.stats.bytes() as f64,
                100.0 * self.stats.unknown_rate()
            );
        }
        Ok(())
    }

    fn recompute(&mut self) {
        self.program_bits = cost::program_bits(&self.program).total();
        self.data_bits = em::data_bits(&self.program, &self.stats);
    }

    /// Total bits per byte of training text.
    pub fn total_bpb(&self) -> f64 {
        float(self.program_bits + self.data_bits) / self.stats.bytes().max(1) as f64
    }

    /// The word forms whose documents a change can affect, or `None` if it
    /// can affect any document (rules, names, hierarchy and merges change
    /// how everything is read or what it costs).
    pub fn affected_forms(&self, ops_: &[Op]) -> Option<Vec<String>> {
        let mut forms = Vec::new();
        for o in ops_ {
            match o {
                Op::AddEntry { form, .. } => forms.push(form.clone()),
                Op::RemoveEntry { id } => {
                    forms.extend(self.program.entries.get(id).map(|e| e.form.clone()))
                }
                Op::AddInfl { add, .. } => forms.extend(
                    self.freq
                        .keys()
                        .filter(|w| w.ends_with(add.as_str()))
                        .cloned(),
                ),
                Op::AddConcept { .. }
                | Op::AddRole { .. }
                | Op::AddFeature { .. }
                | Op::AddValue { .. }
                | Op::SetDoc { .. }
                | Op::Rename { .. } => {}
                _ => return None,
            }
        }
        Some(forms)
    }

    /// Documents a set of word forms can affect.
    pub fn affected(&self, forms: &[String]) -> Vec<usize> {
        let mut out: BTreeSet<usize> = BTreeSet::new();
        for f in forms {
            let words: Vec<&str> = f.split(' ').collect();
            let Some(first) = words.first().and_then(|w| self.index.get(*w)) else {
                continue;
            };
            for &d in first {
                let ok = words.iter().skip(1).all(|w| {
                    self.index
                        .get(*w)
                        .is_some_and(|v| v.binary_search(&d).is_ok())
                });
                if ok {
                    out.insert(d as usize);
                }
            }
        }
        out.into_iter().collect()
    }

    /// Score a change exactly. `forms` are the word forms whose documents
    /// it can affect (`None`: all documents).
    pub fn evaluate(&self, ops: &[Op], forms: Option<&[String]>) -> Result<Eval> {
        let mut p = self.program.clone();
        let mut new_entries = Vec::new();
        for op in ops {
            if let Some(id) = ops::apply(&mut p, op, self.state.round + 1)?
                && matches!(op, Op::AddEntry { .. })
            {
                new_entries.push(id);
            }
        }
        // Provisional counts so the parser can find the new entries.
        for id in &new_entries {
            if let Some(e) = p.entries.get(id) {
                let n = e
                    .form
                    .split(' ')
                    .filter_map(|w| self.freq.get(w))
                    .min()
                    .copied()
                    .unwrap_or(1)
                    .max(1);
                let key = (
                    Table::Lex as u8,
                    [cost::cat_key(&e.cat), cost::sem_key(&e.sem)],
                );
                let cc = p.counts.tables.entry(key).or_default();
                cc.outs
                    .insert(TAG_ENTRY | u64::from(*id), cost::level_of(n));
            }
        }
        let ids: Vec<usize> = match forms {
            Some(f) => self.affected(f),
            None => (0..self.corpus.docs.len()).collect(),
        };
        let docs: Vec<&sememe_data::Doc> = ids
            .iter()
            .filter_map(|i| self.corpus.docs.get(*i))
            .collect();
        // Judge the change with its best counts: alternate parsing the
        // affected documents and recounting (Viterbi EM), more rounds for
        // bigger changes.
        let iters = if ops.len() > 20 || forms.is_none() {
            3
        } else {
            2
        };
        let mut results = Vec::new();
        let mut overlay = None;
        for _ in 0..iters {
            let model = Model::new(Arc::new(p.clone()));
            results = analyze::analyze(&model, &docs, false);
            let (o, new_syms) = self.stats.overlay(&p, &ids, &results);
            let mut syms = self.stats.syms.clone();
            syms.extend(new_syms);
            p.counts = em::m_step_events(&p, o.events.iter().map(|(e, n)| (e, *n)), &syms);
            overlay = Some(o);
        }
        let Some(overlay) = overlay else {
            anyhow::bail!("no evaluation pass")
        };
        let program_bits = cost::program_bits(&p).total();
        let data_bits = em::overlay_bits(&p, &overlay);
        let delta =
            (program_bits + data_bits) as i128 - (self.program_bits + self.data_bits) as i128;
        Ok(Eval {
            program: p,
            ids,
            results,
            delta,
            program_bits,
            data_bits,
        })
    }

    /// Keep an evaluated change.
    /// Held-out bits of a program on the first held-out set.
    fn heldout_bits(&self, p: &Program) -> Bits {
        let Some(c) = self.heldout.values().next() else {
            return 0;
        };
        let model = Model::new(Arc::new(p.clone()));
        let docs: Vec<&sememe_data::Doc> = c.docs.iter().collect();
        analyze::analyze(&model, &docs, false)
            .iter()
            .map(|r| r.bits)
            .sum()
    }

    pub fn commit(&mut self, mut eval: Eval, ops: &[Op], source: &str) -> Result<()> {
        // Check that a sample of accepted changes also helps on held-out
        // text: every bulk change, and one in five small ones.
        let sample = ops.len() > 2
            || sememe_data::hash64(u64::from(self.state.round), format!("{ops:?}").as_bytes())
                .is_multiple_of(5);
        if sample {
            let before = self.heldout_bits(&self.program);
            let after = self.heldout_bits(&eval.program);
            self.generalize.0 += 1;
            if after < before {
                self.generalize.1 += 1;
            }
        }
        // Refresh the single-unknown lines of the re-parsed documents.
        let touched: FxHashSet<usize> = eval.ids.iter().copied().collect();
        self.singles.retain(|(d, _)| !touched.contains(d));
        for (d, r) in eval.ids.iter().zip(eval.results.iter_mut()) {
            for s in r.singles.drain(..) {
                self.singles.push((*d, s));
            }
        }
        self.program = eval.program;
        self.stats.replace(&self.program, &eval.ids, eval.results);
        self.program_bits = eval.program_bits;
        self.data_bits = eval.data_bits;
        let logged: Vec<Logged> = ops
            .iter()
            .map(|op| Logged {
                round: self.state.round + 1,
                source: source.to_string(),
                op: op.clone(),
            })
            .collect();
        ops::log(&self.dir.join("ops.jsonl"), &logged)
    }

    // -----------------------------------------------------------------------
    // The teacher wave.

    /// Lines the program explains worst, weighted by how often their
    /// unknown words occur in the training set, grouped into passages.
    pub fn select_passages(&self, n: usize) -> Vec<Passage> {
        let model_forms: FxHashSet<&str> = self
            .program
            .entries
            .values()
            .map(|e| e.form.as_str())
            .collect();
        let mut scored: Vec<(u64, usize, usize)> = Vec::new();
        for (di, d) in self.corpus.docs.iter().enumerate() {
            for (li, raw) in d.lines.iter().enumerate() {
                if self.state.annotated.contains(&format!("{di}:{li}")) {
                    continue;
                }
                let line = Line::parse(raw);
                let nw = line.words.len();
                if nw == 0 || nw > 20 {
                    continue;
                }
                let mut v = 0f64;
                for w in &line.words {
                    if !model_forms.contains(w.form.as_str()) {
                        v += (1.0 + *self.freq.get(&w.form).unwrap_or(&0) as f64).ln();
                    }
                }
                // Value per word: short lines are cheaper to annotate.
                let score = v / (2.0 + nw as f64).sqrt();
                if score > 0.0 {
                    scored.push(((score * 1000.0) as u64, di, li));
                }
            }
        }
        scored.sort_unstable_by(|a, b| b.cmp(a));
        let mut out = Vec::new();
        let mut used_docs: FxHashSet<usize> = FxHashSet::default();
        for (_, di, li) in scored {
            if out.len() >= n {
                break;
            }
            if !used_docs.insert(di) {
                continue;
            }
            let Some(d) = self.corpus.docs.get(di) else {
                continue;
            };
            let first = li.saturating_sub(self.cfg.lines_per_passage.saturating_sub(1).min(2));
            let lines: Vec<String> = d
                .lines
                .iter()
                .skip(first)
                .take(self.cfg.lines_per_passage)
                .cloned()
                .collect();
            out.push(Passage {
                id: format!("p{}", out.len()),
                doc: di,
                first_line: first,
                lines,
            });
        }
        out
    }

    /// Send a wave of passages to the teacher, in parallel calls.
    pub fn teacher_wave(&mut self) -> Result<Vec<Batch>> {
        // A round interrupted after its teacher wave reuses the stored batches.
        let round = self.state.round + 1;
        let stored: Vec<Batch> =
            annotate::load_batches(&PathBuf::from("data/annotations").join(&self.cfg.run))?
                .into_iter()
                .filter(|b| b.round == round)
                .collect();
        if !stored.is_empty() {
            eprintln!(
                "round {round}: reusing {} stored teacher batches",
                stored.len()
            );
            for b in stored.iter().filter(|b| !b.double) {
                for p in &b.passages {
                    for k in 0..p.lines.len() {
                        self.state
                            .annotated
                            .insert(format!("{}:{}", p.doc, p.first_line + k));
                    }
                }
            }
            return Ok(stored);
        }
        let passages = self.select_passages(self.cfg.calls * self.cfg.passages_per_call);
        let mut groups: Vec<Vec<Passage>> = vec![Vec::new(); self.cfg.calls.max(1)];
        for (i, p) in passages.into_iter().enumerate() {
            if let Some(g) = groups.get_mut(i % self.cfg.calls.max(1)) {
                g.push(p);
            }
        }
        groups.retain(|g| !g.is_empty());
        let _ = round;
        for g in &mut groups {
            for (k, p) in g.iter_mut().enumerate() {
                p.id = format!("p{k}");
            }
        }
        // About 5% of passages are annotated twice, to measure agreement.
        let double: Vec<Passage> = groups
            .iter()
            .flat_map(|g| g.iter())
            .filter(|p| {
                sememe_data::hash64(
                    self.state.seed + u64::from(self.state.round),
                    &p.doc.to_le_bytes(),
                )
                .is_multiple_of(20)
            })
            .cloned()
            .collect();
        let round = self.state.round + 1;
        let ann_dir = PathBuf::from("data/annotations").join(&self.cfg.run);
        let p = &self.program;
        let guide = &self.guide;
        let opus = &self.opus;
        let results: Vec<Result<Batch>> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (i, g) in groups.iter().enumerate() {
                let dir = ann_dir.clone();
                let g = g.clone();
                handles.push(s.spawn(move || {
                    annotate::annotate(opus, p, guide, round, i as u32, g, false, &dir)
                }));
            }
            if !double.is_empty() {
                let dir = ann_dir.clone();
                let mut d = double.clone();
                for (k, p) in d.iter_mut().enumerate() {
                    p.id = format!("p{k}");
                }
                let i = groups.len() as u32;
                handles.push(
                    s.spawn(move || annotate::annotate(opus, p, guide, round, i, d, true, &dir)),
                );
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("teacher thread panicked")))
                })
                .collect()
        });
        let mut batches = Vec::new();
        for r in results {
            match r {
                Ok(b) => {
                    if !b.double {
                        for p in &b.passages {
                            for k in 0..p.lines.len() {
                                self.state.annotated.insert(format!(
                                    "{}:{}",
                                    p.doc,
                                    p.first_line + k
                                ));
                            }
                        }
                    }
                    batches.push(b);
                }
                Err(e) => eprintln!("teacher call failed: {e:#}"),
            }
        }
        Ok(batches)
    }

    /// Apply a batch's proposed types and convert its units to candidate
    /// entries. Returns `(ops, units converted, units failed)`.
    pub fn genlex(&mut self, batches: &[Batch]) -> (Vec<Op>, u32, u32) {
        let mut type_ops = Vec::new();
        for b in batches.iter().filter(|b| !b.double) {
            for r in &b.replies {
                type_ops.extend(learn::apply_proposals(&mut self.program, r, b.round));
            }
        }
        if !type_ops.is_empty() {
            let logged: Vec<Logged> = type_ops
                .iter()
                .map(|op| Logged {
                    round: self.state.round + 1,
                    source: "teacher".into(),
                    op: op.clone(),
                })
                .collect();
            let _ = ops::log(&self.dir.join("ops.jsonl"), &logged);
            self.recompute();
        }
        let mut cands: BTreeMap<genlex::Cand, u32> = BTreeMap::new();
        let (mut ok, mut bad) = (0, 0);
        for b in batches.iter().filter(|b| !b.double) {
            for u in &b.valid {
                let Some(pass) = b.passages.iter().find(|x| x.id == u.passage) else {
                    continue;
                };
                let Some(raw) = pass.lines.get(u.line) else {
                    continue;
                };
                let line = Line::parse(raw);
                let Ok(ann) = annotate::parse_meaning(&u.meaning) else {
                    bad += 1;
                    continue;
                };
                match genlex::convert(&self.program, u, &ann, &line) {
                    Ok(c) => {
                        ok += 1;
                        for e in c.entries {
                            *cands.entry(e).or_default() += 1;
                        }
                    }
                    Err(_) => bad += 1,
                }
            }
        }
        let existing: FxHashSet<(String, String, String)> = {
            let cn = self.program.cat_names();
            self.program
                .entries
                .values()
                .map(|e| (e.form.clone(), e.cat.show(&cn), e.sem.show(&self.program)))
                .collect()
        };
        let ops = cands
            .into_keys()
            .filter(|c| !existing.contains(&(c.form.clone(), c.cat.clone(), c.sem.clone())))
            .map(|c| Op::AddEntry {
                form: c.form,
                cat: c.cat,
                sem: c.sem,
                doc: String::new(),
            })
            .collect();
        (ops, ok, bad)
    }

    /// Candidates from lines that parse except for one unknown word: a new
    /// entry with a fresh concept, for words seen this way at least twice
    /// with the same category.
    pub fn context_candidates(&self) -> Vec<Vec<Op>> {
        let cn = self.program.cat_names();
        let mut groups: BTreeMap<(String, String), (u32, Term)> = BTreeMap::new();
        for (_, s) in &self.singles {
            let key = (s.form.clone(), s.cat.show(&cn));
            let g = groups.entry(key).or_insert((0, s.sem.clone()));
            g.0 += 1;
        }
        // The most frequent first, at most 2,000 per round.
        let mut groups: Vec<((String, String), (u32, Term))> =
            groups.into_iter().filter(|(_, (n, _))| *n >= 2).collect();
        groups.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
        groups.truncate(2000);
        let mut out = Vec::new();
        let mut taken: FxHashSet<String> = FxHashSet::default();
        for ((form, cat), (_n, sem)) in groups {
            let base: String = form
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .enumerate()
                .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
                .collect();
            if base.is_empty() {
                continue;
            }
            let mut name = format!("Ctx{base}");
            let mut k = 2;
            while self.program.concept_by_name(&name).is_some() || taken.contains(&name) {
                name = format!("Ctx{base}{k}");
                k += 1;
            }
            taken.insert(name.clone());
            let Some(c) = self.program.concept_by_name("Thing") else {
                continue;
            };
            // Replace the Unknown head with the new concept (placeholder ID
            // resolved by name after the concept op is applied).
            let sem_text = replace_unknown(&sem, &name, &self.program);
            let _ = c;
            out.push(vec![
                Op::AddConcept {
                    name: name.clone(),
                    parent: "Thing".into(),
                    doc: format!("The meaning of \"{form}\" as {cat}, learned from the contexts it occurs in (not yet named)."),
                },
                Op::AddEntry {
                    form,
                    cat,
                    sem: sem_text,
                    doc: "Learned from context.".into(),
                },
            ]);
        }
        out
    }

    /// Try to remove entries the M-step barely uses.
    pub fn prune_candidates(&self, max_uses: u64) -> Vec<(u32, String)> {
        let mut uses: FxHashMap<u32, u64> = FxHashMap::default();
        for ((t, _), cc) in &self.program.counts.tables {
            if *t != Table::Lex as u8 {
                continue;
            }
            for (o, q) in &cc.outs {
                if o & 0xffff_ffff_0000_0000 == TAG_ENTRY {
                    *uses.entry(*o as u32).or_default() += cost::level_value(*q);
                }
            }
        }
        self.program
            .entries
            .iter()
            .filter(|(id, _)| uses.get(id).copied().unwrap_or(0) <= max_uses)
            .map(|(id, e)| (*id, e.form.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Evaluation and bookkeeping.

    /// Load the probe annotations and paraphrase pairs, if they exist.
    pub fn load_meaning_sets(&mut self) {
        let base = PathBuf::from("data/annotations");
        self.probe = annotate::load_batches(&base.join(format!("{}-probe", self.cfg.run)))
            .unwrap_or_default();
        if let Ok(text) =
            std::fs::read_to_string(base.join(format!("{}-paraphrases.json", self.cfg.run)))
        {
            self.paraphrases = serde_json::from_str(&text).unwrap_or_default();
        }
    }

    /// Meaning metrics: probe agreement, paraphrase agreement, one-word
    /// concepts, teacher self-agreement.
    pub fn meaning_metrics(&self) -> crate::meaning::MeaningMetrics {
        use crate::meaning::{NTree, Overlap, overlap};
        let model = Model::new(Arc::new(self.program.clone()));
        let mut parser = sememe_engine::parse::Parser::new(&model);
        let p = &self.program;
        let mut m = crate::meaning::MeaningMetrics {
            one_word_concepts: crate::meaning::one_word_share(p),
            ..Default::default()
        };
        let mut total = Overlap::default();
        let mut by_src: BTreeMap<String, Overlap> = BTreeMap::new();
        for b in &self.probe {
            for pass in &b.passages {
                let mut ctx = sememe_engine::context::Context::new();
                for (li, raw) in pass.lines.iter().enumerate() {
                    let a = parser.parse_line(&mut ctx, raw);
                    for u in b
                        .valid
                        .iter()
                        .filter(|u| u.passage == pass.id && u.line == li)
                    {
                        let Ok(ann) = annotate::parse_meaning(&u.meaning) else {
                            continue;
                        };
                        let want = NTree::from_ann(&ann);
                        let got = a
                            .fragments
                            .iter()
                            .max_by_key(|f| f.end.min(u.end).saturating_sub(f.start.max(u.start)))
                            .and_then(|f| NTree::from_term(p, &f.meaning));
                        let o = overlap(got.as_ref(), Some(&want));
                        total.add(o);
                        let src = self
                            .corpus
                            .source
                            .get(pass.doc)
                            .cloned()
                            .unwrap_or_default();
                        by_src.entry(src).or_default().add(o);
                        m.probe_units += 1;
                    }
                }
            }
        }
        m.probe_f1 = if m.probe_units == 0 { 0.0 } else { total.f1() };
        m.probe_by_source = by_src.into_iter().map(|(k, v)| (k, v.f1())).collect();
        let (mut same, mut f1sum) = (0u32, 0f64);
        for pp in &self.paraphrases {
            let ma = parser.parse_line(&mut sememe_engine::context::Context::new(), &pp.a);
            let mb = parser.parse_line(&mut sememe_engine::context::Context::new(), &pp.b);
            let (Some(fa), Some(fb)) = (ma.fragments.first(), mb.fragments.first()) else {
                continue;
            };
            if ma.fragments.len() == 1 && mb.fragments.len() == 1 && fa.meaning == fb.meaning {
                same += 1;
            }
            let (ta, tb) = (
                NTree::from_term(p, &fa.meaning),
                NTree::from_term(p, &fb.meaning),
            );
            f1sum += overlap(ta.as_ref(), tb.as_ref()).f1();
        }
        if !self.paraphrases.is_empty() {
            m.paraphrase_same = f64::from(same) / self.paraphrases.len() as f64;
            m.paraphrase_f1 = f1sum / self.paraphrases.len() as f64;
        }
        // Teacher self-agreement: double-annotated lines against the first annotation.
        let batches =
            annotate::load_batches(&PathBuf::from("data/annotations").join(&self.cfg.run))
                .unwrap_or_default();
        let mut first: FxHashMap<(usize, usize, usize, usize), String> = FxHashMap::default();
        for b in batches.iter().filter(|b| !b.double) {
            for u in &b.valid {
                if let Some(pass) = b.passages.iter().find(|x| x.id == u.passage) {
                    first.insert(
                        (pass.doc, pass.first_line + u.line, u.start, u.end),
                        u.meaning.clone(),
                    );
                }
            }
        }
        let mut agree = Overlap::default();
        for b in batches.iter().filter(|b| b.double) {
            for u in &b.valid {
                let Some(pass) = b.passages.iter().find(|x| x.id == u.passage) else {
                    continue;
                };
                let Some(other) = first.get(&(pass.doc, pass.first_line + u.line, u.start, u.end))
                else {
                    continue;
                };
                if let (Ok(x), Ok(y)) = (
                    annotate::parse_meaning(&u.meaning),
                    annotate::parse_meaning(other),
                ) {
                    agree.add(overlap(
                        Some(&NTree::from_ann(&x)),
                        Some(&NTree::from_ann(&y)),
                    ));
                }
            }
        }
        m.teacher_agreement = agree.f1();
        m
    }

    /// Held-out bits per byte and unknown rate for each held-out set.
    pub fn heldout_scores(&self) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
        let model = Model::new(Arc::new(self.program.clone()));
        let mut bpb = BTreeMap::new();
        let mut unk = BTreeMap::new();
        for (k, c) in &self.heldout {
            let docs: Vec<&sememe_data::Doc> = c.docs.iter().collect();
            let rs = analyze::analyze(&model, &docs, false);
            let bits: Bits = rs.iter().map(|r| r.bits).sum();
            let bytes: u64 = rs.iter().map(|r| r.bytes).sum();
            let u: u64 = rs.iter().map(|r| u64::from(r.unknowns)).sum();
            let w: u64 = rs.iter().map(|r| u64::from(r.words)).sum();
            bpb.insert(k.clone(), float(bits) / bytes.max(1) as f64);
            unk.insert(k.clone(), u as f64 / w.max(1) as f64);
        }
        (bpb, unk)
    }

    fn tokens(&self) -> BTreeMap<String, (u64, u64)> {
        let mut out: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(self.dir.join("ledger.jsonl")) {
            for l in text.lines() {
                if let Ok(r) = serde_json::from_str::<crate::opus::LedgerRow>(l) {
                    let e = out.entry(r.tier.clone()).or_default();
                    e.0 += r.input + r.cache_write + r.output;
                    e.1 += r.cache_read;
                }
            }
        }
        out
    }

    /// Save the program and state (full snapshot of the program every 10
    /// rounds in the store; the current program always).
    pub fn checkpoint(&self) -> Result<()> {
        store::save(&self.program, &self.dir.join("program"))?;
        if self.state.round.is_multiple_of(10) {
            store::save(
                &self.program,
                &self
                    .dir
                    .join("store")
                    .join(format!("r{:05}", self.state.round)),
            )?;
        }
        std::fs::write(
            self.dir.join("state.json"),
            serde_json::to_string_pretty(&self.state)?,
        )?;
        Ok(())
    }

    fn append_metrics(&self, m: &Metrics) -> Result<()> {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("metrics.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(m)?)?;
        Ok(())
    }

    /// Accept a list of candidate changes: score each, keep the ones that
    /// help, then re-score the kept ones together. Returns (accepted,
    /// rejected).
    pub fn accept_each(
        &mut self,
        changes: Vec<(Vec<Op>, Vec<String>)>,
        source: &str,
    ) -> Result<(u32, u32)> {
        let mut good: Vec<(i128, Vec<Op>, Vec<String>)> = Vec::new();
        let mut rejected = 0;
        for (ops_, forms) in changes {
            match self.evaluate(&ops_, Some(&forms)) {
                Ok(e) if e.delta < 0 => good.push((e.delta, ops_, forms)),
                Ok(_) => rejected += 1,
                Err(_) => rejected += 1,
            }
        }
        good.sort_by_key(|g| g.0);
        if good.is_empty() {
            return Ok((0, rejected));
        }
        // Together.
        let all_ops: Vec<Op> = good.iter().flat_map(|g| g.1.clone()).collect();
        let all_forms: Vec<String> = good.iter().flat_map(|g| g.2.clone()).collect();
        if let Ok(e) = self.evaluate(&all_ops, Some(&all_forms))
            && e.delta < 0
        {
            let n = good.len() as u32;
            self.commit(e, &all_ops, source)?;
            return Ok((n, rejected));
        }
        // Otherwise greedily, best first.
        let mut accepted = 0;
        for (_, ops_, forms) in good {
            match self.evaluate(&ops_, Some(&forms)) {
                Ok(e) if e.delta < 0 => {
                    self.commit(e, &ops_, source)?;
                    accepted += 1;
                }
                _ => rejected += 1,
            }
        }
        Ok((accepted, rejected))
    }

    /// Add many entries as one change; if it does not pay, split it in two
    /// and try each half, recursively. Returns entries accepted.
    pub fn accept_bulk(
        &mut self,
        groups: Vec<Vec<Op>>,
        source: &str,
        depth: u32,
    ) -> Result<(u32, u32)> {
        // Drop groups that do not apply to the current program.
        let mut probe = self.program.clone();
        let total = groups.len();
        let groups: Vec<Vec<Op>> = groups
            .into_iter()
            .filter(|g| {
                // Validate in place: a failed operation leaves the program as
                // it was, and earlier operations of a failed group are harmless
                // for validating the rest.
                for op in g {
                    if let Err(e) = ops::apply(&mut probe, op, self.state.round + 1) {
                        eprintln!(
                            "  does not apply: {}",
                            format!("{op:?}: {e}").chars().take(300).collect::<String>()
                        );
                        return false;
                    }
                }
                true
            })
            .collect();
        let invalid = (total - groups.len()) as u32;
        if invalid > 0 {
            eprintln!("  dropped {invalid} changes that do not apply");
        }
        if groups.is_empty() {
            return Ok((0, invalid));
        }
        let ops_: Vec<Op> = groups.iter().flatten().cloned().collect();
        let forms = self.affected_forms(&ops_);
        let e = self.evaluate(&ops_, forms.as_deref())?;
        eprintln!(
            "  bulk of {} {source} changes: delta {:.0} bits",
            groups.len(),
            e.delta as f64 / cost::ONE_BIT as f64
        );
        if e.delta < 0 {
            let n = groups.len() as u32;
            self.commit(e, &ops_, source)?;
            return Ok((n, invalid));
        }
        if groups.len() == 1 || depth >= 6 {
            return Ok((0, groups.len() as u32 + invalid));
        }
        let mid = groups.len() / 2;
        let (a, b) = groups.split_at(mid);
        let (x1, y1) = self.accept_bulk(a.to_vec(), source, depth + 1)?;
        let (x2, y2) = self.accept_bulk(b.to_vec(), source, depth + 1)?;
        Ok((x1 + x2, y1 + y2 + invalid))
    }

    /// The slow tier: run agents in parallel (at most four), then score
    /// their operation groups. Returns (accepted, rejected).
    pub fn slow_tier(&mut self, tasks: &[crate::agents::Task]) -> Result<(u32, u32)> {
        let this: &Trainer = self;
        let outs: Vec<Vec<crate::agents::AgentOp>> = std::thread::scope(|s| {
            let hs: Vec<_> = tasks
                .iter()
                .enumerate()
                .map(|(k, task)| s.spawn(move || crate::agents::run(this, *task, k)))
                .collect();
            hs.into_iter()
                .map(|h| match h.join() {
                    Ok(Ok(v)) => v,
                    Ok(Err(e)) => {
                        eprintln!("agent failed: {e:#}");
                        Vec::new()
                    }
                    Err(_) => Vec::new(),
                })
                .collect()
        });
        let (mut acc, mut rej) = (0, 0);
        let mut renamed = false;
        for ops_ in outs {
            // Groups in first-seen order.
            let mut groups: Vec<(String, Vec<Op>)> = Vec::new();
            for o in ops_ {
                match groups.iter_mut().find(|g| g.0 == o.group) {
                    Some(g) => g.1.push(o.op),
                    None => groups.push((o.group, vec![o.op])),
                }
            }
            let mut scored = Vec::new();
            for (_, g) in groups {
                let free = g
                    .iter()
                    .all(|op| matches!(op, Op::Rename { .. } | Op::SetDoc { .. }));
                if !free {
                    scored.push(g);
                    continue;
                }
                let mut p = self.program.clone();
                if g.iter()
                    .all(|op| ops::apply(&mut p, op, self.state.round + 1).is_ok())
                {
                    self.program = p;
                    let logged: Vec<Logged> = g
                        .iter()
                        .map(|op| Logged {
                            round: self.state.round + 1,
                            source: "agent".into(),
                            op: op.clone(),
                        })
                        .collect();
                    ops::log(&self.dir.join("ops.jsonl"), &logged)?;
                    renamed |= g.iter().any(|op| matches!(op, Op::Rename { .. }));
                    acc += 1;
                } else {
                    rej += 1;
                }
            }
            eprintln!("  agent output: {} scored groups", scored.len());
            // Agent changes may re-parse everything: bisect at most twice.
            let (a, r) = self.accept_bulk(scored, "agent", 4)?;
            acc += a;
            rej += r;
        }
        if renamed {
            // Printed symbols mention names: recount from fresh analyses.
            self.full_em(1)?;
        }
        Ok((acc, rej))
    }

    /// One round: (full EM pass), teacher wave, learning from context,
    /// pruning, the BLiMP gate, metrics, checkpoint.
    pub fn round(&mut self) -> Result<Metrics> {
        let started = Instant::now();
        let round = self.state.round + 1;
        let mut m = Metrics {
            round,
            stage: self.state.stage.clone(),
            cost_hash: COST_HASH.to_string(),
            ..Metrics::default()
        };
        let snapshot = (
            self.program.clone(),
            self.stats.clone(),
            self.program_bits,
            self.data_bits,
        );
        if round.is_multiple_of(self.cfg.em_every) {
            self.full_em(1)?;
        }
        if self.cfg.agents_alone {
            let (a, r) = self.slow_tier(&[crate::agents::Task::Direct; 4])?;
            m.accepted += a;
            m.rejected += r;
            return self.finish_round(m, started, snapshot);
        }
        // Medium tier: the teacher.
        if self.cfg.teacher {
            let batches = self.teacher_wave()?;
            m.units = batches
                .iter()
                .filter(|b| !b.double)
                .map(|b| b.valid.len() as u32)
                .sum();
            m.units_dropped = batches
                .iter()
                .filter(|b| !b.double)
                .map(|b| b.dropped.len() as u32)
                .sum();
            let (ops_, _ok, bad) = self.genlex(&batches);
            m.units_unconverted = bad;
            eprintln!(
                "round {round}: {} units, {} candidate entries",
                m.units,
                ops_.len()
            );
            let groups = ops_.into_iter().map(|o| vec![o]).collect();
            let (a, r) = self.accept_bulk(groups, "teacher", 0)?;
            m.from_teacher = a;
            m.accepted += a;
            m.rejected += r;
        }
        // Fast tier: words from context.
        let ctx = self.context_candidates();
        if !ctx.is_empty() {
            eprintln!("round {round}: {} context candidates", ctx.len());
            let (a, r) = self.accept_bulk(ctx, "context", 0)?;
            m.from_context = a;
            m.accepted += a;
            m.rejected += r;
        }
        // Word forms: inflections that make full forms derivable.
        let infl = learn::infl_candidates(&self.program, 3);
        if !infl.is_empty() {
            eprintln!("round {round}: {} inflection candidates", infl.len());
            for g in infl {
                // Every word ending in the suffix can now be read with it.
                let suffix = g.iter().find_map(|o| match o {
                    Op::AddInfl { add, .. } => Some(add.clone()),
                    _ => None,
                });
                let forms: Vec<String> = match suffix {
                    Some(sfx) => self
                        .freq
                        .keys()
                        .filter(|w| w.ends_with(&sfx))
                        .cloned()
                        .collect(),
                    None => Vec::new(),
                };
                match self.evaluate(&g, Some(&forms)) {
                    Ok(e) if e.delta < 0 => {
                        self.commit(e, &g, "infl")?;
                        m.accepted += 1;
                    }
                    _ => m.rejected += 1,
                }
            }
        }
        // Slow tier.
        if self.cfg.slow_every > 0
            && round.is_multiple_of(self.cfg.slow_every)
            && !self.cfg.slow_tasks.is_empty()
        {
            let tasks = self.cfg.slow_tasks.clone();
            let (a, r) = self.slow_tier(&tasks)?;
            eprintln!("round {round}: slow tier accepted {a} groups, rejected {r}");
            m.accepted += a;
            m.rejected += r;
        }
        // Pruning rarely used entries, as one change (bisected if it fails).
        let prune: Vec<Vec<Op>> = self
            .prune_candidates(1)
            .into_iter()
            .map(|(id, _)| vec![Op::RemoveEntry { id }])
            .collect();
        let (pruned, _) = if prune.is_empty() {
            (0, 0)
        } else {
            self.accept_bulk(prune, "prune", 3)?
        };
        m.pruned = pruned;
        self.finish_round(m, started, snapshot)
    }

    #[allow(clippy::type_complexity)]
    fn finish_round(
        &mut self,
        mut m: Metrics,
        started: Instant,
        snapshot: (Program, Stats, Bits, Bits),
    ) -> Result<Metrics> {
        let round = m.round;
        // The BLiMP gate.
        let model = Model::new(Arc::new(self.program.clone()));
        let gate_now = blimp::evaluate(&model, &self.blimp, true);
        if let Some(before) = &self.gate
            && let Some(why) = blimp::gate_fails(before, &gate_now, 2.0)
        {
            eprintln!("round {round} rejected by the BLiMP gate: {why}");
            (self.program, self.stats, self.program_bits, self.data_bits) = snapshot;
            m.rejected += m.accepted;
            m.accepted = 0;
        } else {
            self.gate = Some(gate_now);
        }
        // Metrics.
        let (hb, hu) = self.heldout_scores();
        m.heldout_bpb = hb;
        m.heldout_unknown = hu;
        let model = Model::new(Arc::new(self.program.clone()));
        let b = blimp::evaluate(&model, &self.blimp, false);
        m.blimp = b
            .reported
            .iter()
            .map(|(k, v)| (k.clone(), v.acc()))
            .collect();
        m.blimp_overall = b.overall.acc();
        m.blimp_clean = 100.0 * b.overall.clean_correct / f64::from(b.overall.clean_n.max(1));
        m.blimp_unknown_share = f64::from(b.overall.with_unknown) / f64::from(b.overall.n.max(1));
        m.blimp_fragment_share =
            f64::from(b.overall.bad_fragmented) / f64::from(b.overall.n.max(1));
        m.meaning = self.meaning_metrics();
        m.generalize = std::mem::take(&mut self.generalize);
        let bytes = self.stats.bytes().max(1) as f64;
        m.train_bpb = float(self.data_bits) / bytes;
        m.program_bits = float(self.program_bits);
        m.data_bits = float(self.data_bits);
        m.tokens = self.tokens();
        m.lines_per_sec = self.lines_per_sec;
        m.entries = self.program.entries.len();
        m.concepts = self.program.concepts.len();
        m.rules = self.program.rules.len();
        m.unknown_rate = self.stats.unknown_rate();
        m.ts = now();
        self.state.round = round;
        self.append_metrics(&m)?;
        self.checkpoint()?;
        eprintln!(
            "round {round} done in {:.0}s: held-out {:?}, train total {:.4} bpb, BLiMP {:.1}, entries {}, concepts {}",
            started.elapsed().as_secs_f64(),
            m.heldout_bpb,
            self.total_bpb(),
            m.blimp_overall,
            m.entries,
            m.concepts
        );
        Ok(m)
    }
}

/// Print a meaning with the `Unknown` head replaced by a concept name.
fn replace_unknown(sem: &Term, concept: &str, p: &Program) -> String {
    // Build with a placeholder concept ID that prints as the new name.
    fn go(t: &Term) -> Term {
        match t {
            Term::Lam(b) => Term::Lam(Arc::new(go(b))),
            Term::App(a, b) => Term::App(Arc::new(go(a)), Arc::new(go(b))),
            Term::Add(a, r, b) => Term::Add(Arc::new(go(a)), *r, Arc::new(go(b))),
            Term::Feat(a, f, v) => Term::Feat(Arc::new(go(a)), *f, *v),
            Term::Node(n) => {
                let mut m: Node = (**n).clone();
                if m.head == Head::Unknown {
                    m.head = Head::Concept(u32::MAX);
                }
                for (_, c) in &mut m.roles {
                    *c = go(c);
                }
                Term::node(m)
            }
            Term::Var(_) => t.clone(),
        }
    }
    go(sem).show(p).replace("c4294967295?", concept)
}

/// Run directory helper.
pub fn run_dir(run: &str) -> PathBuf {
    Path::new("runs").join(run)
}
