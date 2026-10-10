//! `sememe-data`: prepares sememe's training and evaluation data.
//! Run from the repository root; see the crate README for the commands.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sememe_data::{babylm, blimp, enwik, file_stats, unpc};

const USAGE: &str = "usage: sememe-data <command> [--key value]...

commands:
  babylm    [--data data] [--out data/prepared]
  blimp     [--data data] [--out data/prepared]
  enwik     [--data data] [--out data/prepared] [--only enwik8|enwik9]
  unpc      [--data data] [--out data/prepared] [--pool-mb 100] [--heldout-mb 1]
            [--aligned-mb 10] [--block-lines 200] [--max-lines N]
  manifest  <dir>   (writes <dir>/manifest.json with stats of every .docs file under <dir>)";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        bail!("{USAGE}");
    };
    let mut opts: HashMap<String, String> = HashMap::new();
    let mut positional = Vec::new();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        if let Some(key) = a.strip_prefix("--") {
            let Some(v) = it.next() else {
                bail!("--{key} needs a value\n{USAGE}");
            };
            opts.insert(key.to_string(), v.clone());
        } else {
            positional.push(a.clone());
        }
    }
    let path = |key: &str, default: &str| PathBuf::from(opts.get(key).map_or(default, |s| s));
    let num = |key: &str| -> Result<Option<u64>> {
        opts.get(key)
            .map(|s| s.parse::<u64>().with_context(|| format!("--{key} {s}")))
            .transpose()
    };
    let data = path("data", "data");
    let out = path("out", "data/prepared");
    let start = Instant::now();

    match command.as_str() {
        "babylm" => {
            let cfg = babylm::Config::new(data.join("babylm"), out.join("babylm"));
            let manifest = babylm::prepare(&cfg)?;
            print!("{}", babylm::summary(&manifest));
            println!("wrote {}", cfg.output.join("manifest.json").display());
        }
        "blimp" => {
            let s = blimp::prepare(&data.join("blimp"), &out.join("blimp"), 0)?;
            println!(
                "pairs: filtered {}, supplement {}, full {}",
                s.filtered_pairs, s.supplement_pairs, s.full_pairs
            );
            for (phenomenon, (gating, reported)) in &s.phenomena {
                println!(
                    "{phenomenon:<26} gating {:<60} reported {}",
                    gating.join(","),
                    reported.len()
                );
            }
        }
        "enwik" => {
            let names: Vec<&str> = match opts.get("only").map(String::as_str) {
                Some(n) => vec![n],
                None => vec!["enwik8", "enwik9"],
            };
            let mut all = BTreeMap::new();
            for name in names {
                let t = Instant::now();
                let o = enwik::extract(&data.join("enwik"), &out.join("enwik"), name)?;
                println!(
                    "{name}: {} raw bytes, {} pages ({} redirects), clean {} bytes in {} docs, {:.1}s",
                    o.raw_bytes,
                    o.pages,
                    o.redirects,
                    o.clean.bytes,
                    o.clean.docs,
                    t.elapsed().as_secs_f64()
                );
                all.insert(name.to_string(), o);
            }
            write_json(&out.join("enwik/manifest.json"), &all)?;
        }
        "unpc" => {
            let mut cfg = unpc::Config::new(data.join("unpc"), out.join("unpc"));
            let mb = 1_000_000;
            if let Some(v) = num("pool-mb")? {
                cfg.pool_bytes = v * mb;
            }
            if let Some(v) = num("heldout-mb")? {
                cfg.heldout_bytes = v * mb;
            }
            if let Some(v) = num("aligned-mb")? {
                cfg.aligned_bytes = v * mb;
            }
            if let Some(v) = num("block-lines")? {
                cfg.block_lines = usize::try_from(v)?.max(1);
            }
            cfg.max_lines = num("max-lines")?;
            let m = unpc::prepare(&cfg)?;
            println!(
                "{} lines, {} blocks, {} skipped",
                m.lines_read, m.blocks, m.skipped_blocks
            );
            print_stats(&m.files);
        }
        "manifest" => {
            let Some(dir) = positional.first() else {
                bail!("{USAGE}");
            };
            let dir = Path::new(dir);
            let mut files = BTreeMap::new();
            collect_docs(dir, dir, &mut files)?;
            print_stats(&files);
            write_json(&dir.join("manifest.json"), &files)?;
        }
        _ => bail!("unknown command {command}\n{USAGE}"),
    }
    eprintln!("done in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn collect_docs(
    root: &Path,
    dir: &Path,
    files: &mut BTreeMap<String, sememe_data::FileStats>,
) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_docs(root, &p, files)?;
        } else if p.extension().is_some_and(|e| e == "docs") {
            let rel = p.strip_prefix(root).unwrap_or(&p).display().to_string();
            files.insert(rel, file_stats(&p)?);
        }
    }
    Ok(())
}

fn print_stats(files: &BTreeMap<String, sememe_data::FileStats>) {
    println!(
        "{:<34} {:>12} {:>10} {:>8} {:>12}",
        "file", "bytes", "lines", "docs", "words"
    );
    for (path, f) in files {
        println!(
            "{path:<34} {:>12} {:>10} {:>8} {:>12}",
            f.bytes, f.lines, f.docs, f.words
        );
    }
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}
