//! BLiMP: BabyLM's filtered BLiMP and supplement, and the full BLiMP, as
//! uniform jsonl, plus the gating/reported split by paradigm.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::docs::hash64;

/// One minimal pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Pair {
    pub paradigm: String,
    pub phenomenon: String,
    pub good: String,
    pub bad: String,
    pub pair_id: u64,
}

/// Whether a paradigm gates changes or is only reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Gating,
    Reported,
}

/// Parses BLiMP-format jsonl (`sentence_good`, `sentence_bad`, `UID`,
/// `linguistics_term`, `pair_id` or `pairID`).
pub fn parse_blimp<R: BufRead>(reader: R, name: &str) -> Result<Vec<Pair>> {
    let mut pairs = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading {name}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(&line).with_context(|| format!("{name}: row {i} is not JSON"))?;
        let pair_id = v
            .get("pair_id")
            .or_else(|| v.get("pairID"))
            .and_then(|x| {
                x.as_u64()
                    .or_else(|| x.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(i as u64);
        pairs.push(Pair {
            paradigm: field(&v, "UID", name, i)?,
            phenomenon: field(&v, "linguistics_term", name, i)?,
            good: field(&v, "sentence_good", name, i)?,
            bad: field(&v, "sentence_bad", name, i)?,
            pair_id,
        });
    }
    Ok(pairs)
}

/// Parses a BabyLM supplement file. These have no paradigm or phenomenon
/// fields: both are the file stem, and `pair_id` is the row's position.
pub fn parse_supplement<R: BufRead>(reader: R, paradigm: &str) -> Result<Vec<Pair>> {
    let mut pairs = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading {paradigm}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("{paradigm}: row {i} is not JSON"))?;
        pairs.push(Pair {
            paradigm: paradigm.to_string(),
            phenomenon: paradigm.to_string(),
            good: field(&v, "sentence_good", paradigm, i)?,
            bad: field(&v, "sentence_bad", paradigm, i)?,
            pair_id: i as u64,
        });
    }
    Ok(pairs)
}

fn field(v: &Value, key: &str, name: &str, row: usize) -> Result<String> {
    match v.get(key).and_then(Value::as_str) {
        Some(s) => Ok(s.to_string()),
        None => bail!("{name}: row {row} has no string field {key}"),
    }
}

/// Number of gating paradigms for a phenomenon with `n` paradigms: about a
/// third, at least one from three paradigms up, none below.
pub fn gating_count(n: usize) -> usize {
    if n < 3 { 0 } else { (n + 1) / 3 }
}

/// Splits paradigms within each phenomenon: sorted by `hash64(seed,
/// paradigm)`, the first [`gating_count`] gate, the rest are reported.
pub fn split_paradigms(pairs: &[Pair], seed: u64) -> BTreeMap<String, Role> {
    let mut by_phenomenon: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for p in pairs {
        let list = by_phenomenon.entry(&p.phenomenon).or_default();
        if !list.contains(&p.paradigm.as_str()) {
            list.push(&p.paradigm);
        }
    }
    let mut roles = BTreeMap::new();
    for paradigms in by_phenomenon.values_mut() {
        paradigms.sort_by_key(|p| (hash64(seed, p.as_bytes()), *p));
        let gating = gating_count(paradigms.len());
        for (i, p) in paradigms.iter().enumerate() {
            let role = if i < gating {
                Role::Gating
            } else {
                Role::Reported
            };
            roles.insert(p.to_string(), role);
        }
    }
    roles
}

fn jsonl_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    Ok(files)
}

fn write_pairs(path: &Path, pairs: &[Pair]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    for p in pairs {
        serde_json::to_writer(&mut w, p)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// Counts of what [`prepare`] wrote.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub filtered_pairs: usize,
    pub supplement_pairs: usize,
    pub full_pairs: usize,
    /// phenomenon → (gating paradigms, reported paradigms)
    pub phenomena: BTreeMap<String, (Vec<String>, Vec<String>)>,
}

/// Writes `filtered.jsonl`, `supplement.jsonl`, `full.jsonl` and
/// `split.json` into `output` from `input` (= `data/blimp`).
pub fn prepare(input: &Path, output: &Path, seed: u64) -> Result<Summary> {
    std::fs::create_dir_all(output).with_context(|| format!("creating {}", output.display()))?;

    let mut filtered = Vec::new();
    for path in jsonl_files(&input.join("babylm/blimp_filtered"))? {
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        filtered.extend(parse_blimp(
            BufReader::new(file),
            &path.display().to_string(),
        )?);
    }
    let mut supplement = Vec::new();
    for path in jsonl_files(&input.join("babylm/supplement_filtered"))? {
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        supplement.extend(parse_supplement(
            BufReader::new(file),
            &crate::docs::file_stem(&path),
        )?);
    }

    let zip_path = input.join("blimp-master.zip");
    let file = File::open(&zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))?;
    let mut names: Vec<String> = archive
        .file_names()
        .filter(|n| n.starts_with("blimp-master/data/") && n.ends_with(".jsonl"))
        .map(str::to_string)
        .collect();
    names.sort();
    let mut full = Vec::new();
    for name in &names {
        let mut text = String::new();
        archive.by_name(name)?.read_to_string(&mut text)?;
        full.extend(parse_blimp(text.as_bytes(), name)?);
    }

    let mut roles = split_paradigms(&filtered, seed);
    for p in &supplement {
        roles.insert(p.paradigm.clone(), Role::Reported);
    }

    write_pairs(&output.join("filtered.jsonl"), &filtered)?;
    write_pairs(&output.join("supplement.jsonl"), &supplement)?;
    write_pairs(&output.join("full.jsonl"), &full)?;
    let split_path = output.join("split.json");
    std::fs::write(&split_path, serde_json::to_string_pretty(&roles)?)
        .with_context(|| format!("writing {}", split_path.display()))?;

    let mut phenomena: BTreeMap<String, (Vec<String>, Vec<String>)> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for p in filtered.iter().chain(&supplement) {
        if !seen.insert(p.paradigm.as_str()) {
            continue;
        }
        let entry = phenomena.entry(p.phenomenon.clone()).or_default();
        match roles.get(&p.paradigm) {
            Some(Role::Gating) => entry.0.push(p.paradigm.clone()),
            _ => entry.1.push(p.paradigm.clone()),
        }
    }
    Ok(Summary {
        filtered_pairs: filtered.len(),
        supplement_pairs: supplement.len(),
        full_pairs: full.len(),
        phenomena,
    })
}
