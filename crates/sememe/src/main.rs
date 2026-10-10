//! The sememe trainer.
#![allow(dead_code)]

mod analyze;
mod corpus;
mod em;
mod store;

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

fn load_program(p: &Option<PathBuf>) -> Result<Program> {
    match p {
        Some(dir) => store::load(dir),
        None => Ok(Program::seed()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
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
                let before_data = em::data_bits(&p, &stats);
                let before_prog = cost::program_bits(&p).total();
                p.counts = em::m_step(&p, &stats);
                let after_data = em::data_bits(&p, &stats);
                let after_prog = cost::program_bits(&p);
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
