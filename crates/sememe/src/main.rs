//! The sememe trainer.
#![allow(dead_code)]

mod agents;
mod analyze;
mod annotate;
mod blimp;
mod corpus;
mod em;
mod genlex;
mod learn;
mod meaning;
mod ops;
mod opus;
mod store;
mod trainer;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser as ClapParser, Subcommand};
use sememe_engine::cost::{self, float};
use sememe_engine::model::Model;
use sememe_engine::program::Program;

#[derive(ClapParser)]
#[command(about = "sememe: a compiler from text to a readable language model")]
struct Cli {
    /// Worker threads (at most 16).
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Apply GENLEX candidates from stored batches to a program and save it
    /// (for diagnosis; no scoring).
    GenlexApply {
        /// Annotation batch directory.
        batches: PathBuf,
        /// Output program directory.
        #[arg(long)]
        out: PathBuf,
    },
    /// Have the teacher annotate the meaning probe (once per run), with the
    /// run's current ontology.
    ProbeAnnotate {
        /// Run name.
        #[arg(long)]
        run: String,
        /// Probe set.
        #[arg(long, default_value = "data/prepared/babylm/probe")]
        probe: PathBuf,
        /// Lines (with words) to annotate.
        #[arg(long, default_value_t = 1000)]
        lines: usize,
    },
    /// Have the teacher write paraphrase pairs (once per run).
    Paraphrases {
        /// Run name.
        #[arg(long)]
        run: String,
        /// Pairs.
        #[arg(long, default_value_t = 200)]
        n: usize,
    },
    /// Run (or resume) the training loop.
    Run {
        /// Run name (directory under runs/).
        #[arg(long)]
        run: String,
        /// Stage name.
        #[arg(long, default_value = "babylm-10M")]
        stage: String,
        /// Training set.
        #[arg(long, default_value = "data/prepared/babylm/train10M")]
        train: PathBuf,
        /// Held-out sample.
        #[arg(long, default_value = "data/prepared/babylm/dev-sample")]
        heldout: PathBuf,
        /// Rounds to run in this session (then stop with a checkpoint).
        #[arg(long, default_value_t = 1)]
        rounds: u32,
        /// Teacher calls per wave.
        #[arg(long, default_value_t = 8)]
        calls: usize,
        /// Passages per call.
        #[arg(long, default_value_t = 20)]
        passages: usize,
        /// Disable the teacher.
        #[arg(long)]
        no_teacher: bool,
        /// Agents-alone baseline mode.
        #[arg(long)]
        agents_alone: bool,
        /// Slow tier every N rounds (0: never).
        #[arg(long, default_value_t = 3)]
        slow_every: u32,
    },
    /// Convert stored annotations to candidate entries and report.
    GenlexTrial {
        /// Annotation batch directory.
        batches: PathBuf,
    },
    /// Send a sample of passages to the teacher once and report.
    TeachSample {
        /// Training set.
        set: PathBuf,
        /// Program (seed if omitted).
        #[arg(long)]
        program: Option<PathBuf>,
        /// Passages.
        #[arg(long, default_value_t = 10)]
        passages: usize,
        /// Lines per passage.
        #[arg(long, default_value_t = 4)]
        lines: usize,
        /// Sampling seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Where to store the batch.
        #[arg(long)]
        out: PathBuf,
    },
    /// Score a document set with a program and report bits per byte.
    Score {
        /// Program directory (the seed program if omitted).
        #[arg(long)]
        program: Option<PathBuf>,
        /// Prepared `.docs` file or directory.
        set: PathBuf,
        /// Only the first N documents.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run Viterbi EM on a training set, starting from a program.
    Em {
        /// Starting program (the seed if omitted).
        #[arg(long)]
        program: Option<PathBuf>,
        /// Training set.
        set: PathBuf,
        /// Iterations.
        #[arg(long, default_value_t = 3)]
        iters: usize,
        /// Output program directory.
        #[arg(long)]
        out: PathBuf,
        /// Only the first N documents.
        #[arg(long)]
        limit: Option<usize>,
    },
}

/// Where model calls run: outside the repo, so no project instructions load.
fn dirs_workdir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/claude-tmp/sememe-calls")
}

fn load_program(p: &Option<PathBuf>) -> Result<Program> {
    match p {
        Some(dir) => store::load(dir),
        None => Ok(Program::seed()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // On Ctrl-C or SIGTERM, stop the model calls we started, then exit.
    let _ = ctrlc::set_handler(|| {
        opus::kill_children();
        std::process::exit(130);
    });
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.clamp(1, 16))
        .build_global()?;
    match cli.cmd {
        Cmd::Score {
            program,
            set,
            limit,
        } => {
            let p = Arc::new(load_program(&program)?);
            let corpus = corpus::Corpus::load(&set)?;
            let n = limit.unwrap_or(corpus.docs.len()).min(corpus.docs.len());
            let docs: Vec<_> = corpus.docs.iter().take(n).collect();
            let model = Model::new(p.clone());
            let t = Instant::now();
            let results = analyze::analyze(&model, &docs, false);
            let secs = t.elapsed().as_secs_f64();
            let bits: u64 = results.iter().map(|r| r.bits).sum();
            let bytes: u64 = results.iter().map(|r| r.bytes).sum();
            let lines: usize = docs.iter().map(|d| d.lines.len()).sum();
            let unk: u64 = results.iter().map(|r| u64::from(r.unknowns)).sum();
            let words: u64 = results.iter().map(|r| u64::from(r.words)).sum();
            println!(
                "{} docs, {lines} lines, {bytes} bytes: {:.4} bits/byte, unknown {:.1}%, {:.0} lines/s",
                docs.len(),
                float(bits) / bytes as f64,
                100.0 * unk as f64 / words.max(1) as f64,
                lines as f64 / secs
            );
        }
        Cmd::Run {
            run,
            stage,
            train,
            heldout,
            rounds,
            calls,
            passages,
            no_teacher,
            agents_alone,
            slow_every,
        } => {
            let mut held = std::collections::BTreeMap::new();
            held.insert(
                stage.split('-').next().unwrap_or("heldout").to_string(),
                heldout,
            );
            let cfg = trainer::Config {
                run,
                stage,
                train,
                heldout: held,
                blimp: PathBuf::from("data/prepared/blimp"),
                calls,
                passages_per_call: passages,
                lines_per_passage: 3,
                em_every: 5,
                max_rounds: u32::MAX,
                teacher: !no_teacher && !agents_alone,
                slow_every,
                slow_tasks: vec![
                    agents::Task::Consolidate,
                    agents::Task::Name,
                    agents::Task::Generalize,
                ],
                agents_alone,
            };
            let teacher = cfg.teacher;
            let mut t = trainer::Trainer::open(cfg)?;
            if teacher {
                opus::self_test(&t.opus)?;
                eprintln!("self-test ok");
            }
            for _ in 0..rounds {
                t.round()?;
            }
        }
        Cmd::ProbeAnnotate { run, probe, lines } => {
            let dir = trainer::run_dir(&run);
            let p = store::load(&dir.join("program"))?;
            let corpus = corpus::Corpus::load(&probe)?;
            let mut passages = Vec::new();
            let mut count = 0;
            'docs: for (di, d) in corpus.docs.iter().enumerate() {
                for (k, chunk) in d.lines.chunks(3).enumerate() {
                    if count >= lines {
                        break 'docs;
                    }
                    count += chunk
                        .iter()
                        .filter(|l| !sememe_engine::text::Line::parse(l).words.is_empty())
                        .count();
                    passages.push(annotate::Passage {
                        id: String::new(),
                        doc: di,
                        first_line: k * 3,
                        lines: chunk.to_vec(),
                    });
                }
            }
            let guide = std::fs::read_to_string("docs/meaning-guide.md")?;
            let opus = opus::Opus::new(&dir.join("ledger.jsonl"), &dirs_workdir())?;
            let out = PathBuf::from("data/annotations").join(format!("{run}-probe"));
            let groups: Vec<Vec<annotate::Passage>> = passages
                .chunks(20)
                .map(|g| {
                    g.iter()
                        .cloned()
                        .enumerate()
                        .map(|(i, mut x)| {
                            x.id = format!("p{i}");
                            x
                        })
                        .collect()
                })
                .collect();
            eprintln!("{} passages in {} calls", passages.len(), groups.len());
            let results: Vec<_> = std::thread::scope(|s| {
                let hs: Vec<_> = groups
                    .into_iter()
                    .enumerate()
                    .map(|(i, g)| {
                        let (opus, p, guide, out) = (&opus, &p, &guide, &out);
                        s.spawn(move || {
                            annotate::annotate(opus, p, guide, 0, i as u32, g, false, out)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().ok()).collect()
            });
            let ok = results.iter().filter(|r| matches!(r, Some(Ok(_)))).count();
            println!("{ok} probe batches stored in {}", out.display());
        }
        Cmd::Paraphrases { run, n } => {
            let dir = trainer::run_dir(&run);
            let opus = opus::Opus::new(&dir.join("ledger.jsonl"), &dirs_workdir())?;
            let schema = serde_json::json!({"type": "object", "properties": {"pairs": {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}}}, "required": ["pairs"]});
            let mut pairs: Vec<meaning::Paraphrase> = Vec::new();
            for k in 0..n.div_ceil(50) {
                let prompt = format!(
                    "Write 50 pairs of short English sentences (3 to 10 words) in the everyday register of conversation with \
                     children and simple stories. In each pair both sentences mean exactly the same thing but are worded \
                     differently: different words (synonyms), active and passive, a name and a description, different word \
                     order, contractions. Vary the topics. Batch {k}: do not repeat earlier batches' ideas."
                );
                let r = opus.structured(
                    opus::Tier::Teacher,
                    0,
                    "paraphrases",
                    "Reply with JSON only.",
                    &prompt,
                    &schema,
                    std::time::Duration::from_secs(600),
                )?;
                let got: Vec<meaning::Paraphrase> =
                    serde_json::from_value(r.value.get("pairs").cloned().unwrap_or_default())?;
                pairs.extend(got);
            }
            pairs.truncate(n);
            let out = PathBuf::from("data/annotations").join(format!("{run}-paraphrases.json"));
            std::fs::create_dir_all("data/annotations")?;
            std::fs::write(&out, serde_json::to_string_pretty(&pairs)?)?;
            println!("{} pairs written to {}", pairs.len(), out.display());
        }
        Cmd::GenlexApply { batches, out } => {
            let mut p = Program::seed();
            let bs = annotate::load_batches(&batches)?;
            let mut n = 0;
            for b in bs.iter().filter(|b| !b.double) {
                for r in &b.replies {
                    learn::apply_proposals(&mut p, r, b.round);
                }
            }
            for b in bs.iter().filter(|b| !b.double) {
                for u in &b.valid {
                    let Some(pass) = b.passages.iter().find(|x| x.id == u.passage) else {
                        continue;
                    };
                    let Some(raw) = pass.lines.get(u.line) else {
                        continue;
                    };
                    let line = sememe_engine::text::Line::parse(raw);
                    let Ok(ann) = annotate::parse_meaning(&u.meaning) else {
                        continue;
                    };
                    if let Ok(c) = genlex::convert(&p, u, &ann, &line) {
                        for e in c.entries {
                            let op = ops::Op::AddEntry {
                                form: e.form,
                                cat: e.cat,
                                sem: e.sem,
                                doc: String::new(),
                            };
                            if ops::apply(&mut p, &op, 1).is_ok() {
                                n += 1;
                            }
                        }
                    }
                }
            }
            store::save(&p, &out)?;
            println!("{n} entries");
        }
        Cmd::GenlexTrial { batches } => {
            let mut p = Program::seed();
            let bs = annotate::load_batches(&batches)?;
            let mut ok = 0;
            let mut fails: std::collections::BTreeMap<String, usize> = Default::default();
            let mut cands: std::collections::BTreeMap<genlex::Cand, usize> = Default::default();
            for b in &bs {
                for r in &b.replies {
                    learn::apply_proposals(&mut p, r, b.round);
                }
                for u in &b.valid {
                    let Some(pass) = b.passages.iter().find(|x| x.id == u.passage) else {
                        continue;
                    };
                    let Some(raw) = pass.lines.get(u.line) else {
                        continue;
                    };
                    let line = sememe_engine::text::Line::parse(raw);
                    let ann = match annotate::parse_meaning(&u.meaning) {
                        Ok(a) => a,
                        Err(e) => {
                            *fails.entry(e).or_default() += 1;
                            continue;
                        }
                    };
                    match genlex::convert(&p, u, &ann, &line) {
                        Ok(c) => {
                            ok += 1;
                            for e in c.entries {
                                *cands.entry(e).or_default() += 1;
                            }
                        }
                        Err(e) => {
                            println!("FAIL {}: {e}\n  {raw}\n  {}", u.passage, u.meaning);
                            *fails.entry(e.to_string()).or_default() += 1;
                        }
                    }
                }
            }
            println!("converted {ok}; failures: {fails:?}");
            for (c, n) in &cands {
                println!("{n:3} {} := {} : {}", c.form, c.cat, c.sem);
            }
        }
        Cmd::TeachSample {
            set,
            program,
            passages,
            lines,
            seed,
            out,
        } => {
            let p = load_program(&program)?;
            let corpus = corpus::Corpus::load(&set)?;
            let mut chosen = Vec::new();
            let n = corpus.docs.len().max(1);
            for k in 0..passages {
                let d = (sememe_data::hash64(seed, &k.to_le_bytes()) as usize) % n;
                let Some(doc) = corpus.docs.get(d) else {
                    continue;
                };
                let start = (sememe_data::hash64(seed + 1, &k.to_le_bytes()) as usize)
                    % doc.lines.len().max(1);
                let ls: Vec<String> = doc.lines.iter().skip(start).take(lines).cloned().collect();
                chosen.push(annotate::Passage {
                    id: format!("p{k}"),
                    doc: d,
                    first_line: start,
                    lines: ls,
                });
            }
            let guide = std::fs::read_to_string("docs/meaning-guide.md")?;
            let opus = opus::Opus::new(&out.join("ledger.jsonl"), &dirs_workdir())?;
            let t = Instant::now();
            let b = annotate::annotate(&opus, &p, &guide, 0, 0, chosen, false, &out)?;
            println!(
                "{} valid units, {} dropped, {} replies, {:.0}s",
                b.valid.len(),
                b.dropped.len(),
                b.replies.len(),
                t.elapsed().as_secs_f64()
            );
            for (u, e) in &b.dropped {
                println!(
                    "DROPPED {} {} {}..{}: {} -- {e}",
                    u.passage, u.line, u.start, u.end, u.meaning
                );
            }
        }
        Cmd::Em {
            program,
            set,
            iters,
            out,
            limit,
        } => {
            let mut p = load_program(&program)?;
            let corpus = corpus::Corpus::load(&set)?;
            let n = limit.unwrap_or(corpus.docs.len()).min(corpus.docs.len());
            let docs: Vec<_> = corpus.docs.iter().take(n).collect();
            for it in 0..iters {
                let model = Model::new(Arc::new(p.clone()));
                let t = Instant::now();
                let stats = em::Stats::build(&model, &docs);
                let parse_secs = t.elapsed().as_secs_f64();
                let bytes = stats.bytes();
                let t1 = Instant::now();
                let before_data = em::data_bits(&p, &stats);
                let before_prog = cost::program_bits(&p).total();
                let t_bits = t1.elapsed().as_secs_f64();
                let t2 = Instant::now();
                p.counts = em::m_step(&p, &stats);
                let t_m = t2.elapsed().as_secs_f64();
                let after_data = em::data_bits(&p, &stats);
                let after_prog = cost::program_bits(&p);
                println!("timing: data+program bits {t_bits:.2}s, m-step {t_m:.2}s");
                println!(
                    "iter {it}: parse {parse_secs:.0}s; before: data {:.4} + program {:.4} bpb; after M-step: data {:.4} + program {:.4} = {:.4} bpb (counts {:.0} KB); unknown {:.1}%; {} distinct events",
                    float(before_data) / bytes as f64,
                    float(before_prog) / bytes as f64,
                    float(after_data) / bytes as f64,
                    float(after_prog.total()) / bytes as f64,
                    float(after_data + after_prog.total()) / bytes as f64,
                    float(after_prog.counts) / 8192.0,
                    100.0 * stats.unknown_rate(),
                    stats.events.len(),
                );
                for (t, b) in em::bits_by_table(&p, &stats).iter().take(8) {
                    print!("{}={:.3} ", t.name(), float(*b) / bytes as f64);
                }
                println!();
            }
            store::save(&p, &out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// One loop step on the committed fixture with a stubbed teacher and a
    /// stubbed agent: the round runs, and the replayed operations log
    /// rebuilds the same program.
    #[test]
    fn one_loop_step_with_stubs() {
        crate::opus::STUB.store(true, std::sync::atomic::Ordering::Relaxed);
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::env::set_current_dir(&root).unwrap();
        let run = "ci-fixture";
        let _ = std::fs::remove_dir_all(root.join("runs").join(run));
        let mut held = std::collections::BTreeMap::new();
        held.insert(
            "fixture".to_string(),
            root.join("tests/fixture/fixture.docs"),
        );
        let cfg = crate::trainer::Config {
            run: run.into(),
            stage: "fixture".into(),
            train: root.join("tests/fixture/fixture.docs"),
            heldout: held,
            blimp: root.join("tests/fixture/blimp"),
            calls: 1,
            passages_per_call: 4,
            lines_per_passage: 3,
            em_every: 1,
            max_rounds: 1,
            teacher: true,
            slow_every: 1,
            slow_tasks: vec![crate::agents::Task::Name],
            agents_alone: false,
        };
        let mut t = crate::trainer::Trainer::open(cfg).unwrap();
        crate::opus::self_test(&t.opus).unwrap();
        let m = t.round().unwrap();
        assert_eq!(m.round, 1);
        assert!(m.units > 0);
        assert!(m.heldout_bpb.get("fixture").is_some_and(|b| *b > 0.0));
        // Replaying the log from the seed rebuilds the structure.
        let log = crate::ops::read_log(&root.join("runs").join(run).join("ops.jsonl")).unwrap();
        let mut p = sememe_engine::program::Program::seed();
        for l in &log {
            crate::ops::apply(&mut p, &l.op, l.round).unwrap();
        }
        p.counts = t.program.counts.clone();
        assert_eq!(p.to_files(), t.program.to_files());
        let _ = std::fs::remove_dir_all(root.join("runs").join(run));
        let _ = std::fs::remove_dir_all(root.join("data/annotations").join(run));
    }
}
