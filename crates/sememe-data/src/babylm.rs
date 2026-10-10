//! Stage 1: BabyLM. Held-out sets and samples, overlap removal, the meaning
//! probe and the two training sets, all as `.docs` files plus a manifest.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::docs::{DocReader, DocWriter, FileStats, hash64, lines_bytes, word_count};
use crate::split::{MAX_DOC_LINES, SplitRule, for_each_doc};

/// The six BabyLM sources, one file each per split.
pub const SOURCES: [&str; 6] = [
    "childes",
    "bnc_spoken",
    "switchboard",
    "open_subtitles",
    "gutenberg",
    "simple_wiki",
];

/// Probe documents shorter than this (in original bytes) are never picked,
/// so that removing them from train_100M by sequence match cannot hit
/// unrelated text (a one-line "yeah." document would match everywhere).
pub const MIN_PROBE_DOC_BYTES: u64 = 200;

/// The split rule for a source.
pub fn rule_for(source: &str) -> SplitRule {
    if source == "simple_wiki" {
        SplitRule::BlankOrWikiHeader
    } else {
        SplitRule::Blank
    }
}

/// Lines long enough that a repeat between held-out and training text is
/// leakage rather than chance: at least 6 whitespace words or 40 bytes.
pub fn is_long_line(line: &str) -> bool {
    line.len() >= 40 || word_count(line) >= 6
}

fn has_word(line: &str) -> bool {
    line.split_whitespace().next().is_some()
}

/// Settings for [`prepare`].
#[derive(Debug, Clone, Serialize)]
pub struct Config {
    /// `data/babylm`, holding the four zips.
    pub input: PathBuf,
    /// `data/prepared/babylm`.
    pub output: PathBuf,
    /// Size of `dev-sample` and `test-sample`, in bytes.
    pub sample_bytes: u64,
    /// Lines with at least one word to put in the meaning probe.
    pub probe_lines: u64,
    /// Seed for every hash-based choice.
    pub seed: u64,
}

impl Config {
    pub fn new(input: PathBuf, output: PathBuf) -> Self {
        Self {
            input,
            output,
            sample_bytes: 1_000_000,
            probe_lines: 3000,
            seed: 0,
        }
    }
}

/// Per-source removal counts.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Removals {
    /// Long lines removed from train_10M because they occur in dev or test.
    pub train10m_heldout_lines: u64,
    pub train100m_heldout_lines: u64,
    /// Documents in the probe, and their lines.
    pub probe_docs: u64,
    pub probe_lines: u64,
    pub probe_worded_lines: u64,
    /// Lines of other train_10M documents removed because they repeat a long
    /// probe line.
    pub train10m_probe_repeat_lines: u64,
    /// Occurrences of probe documents found in train_100M (as consecutive
    /// lines), and the lines they cover.
    pub train100m_probe_matches: u64,
    pub train100m_probe_lines: u64,
    /// Other train_100M lines removed because they repeat a long probe line.
    pub train100m_probe_repeat_lines: u64,
    /// Probe documents that were not found in train_100M.
    pub probe_docs_missing_from_100m: u64,
    /// Source documents left empty by the removals (dropped).
    pub train10m_dropped_docs: u64,
    pub train100m_dropped_docs: u64,
}

/// Data checks recorded in the manifest.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Checks {
    /// Distinct train_10M lines sampled (hash % 64 == 0) and how many occur
    /// anywhere in train_100M; the same for long lines only.
    pub subset_sampled: u64,
    pub subset_found: u64,
    pub subset_sampled_long: u64,
    pub subset_found_long: u64,
    /// Distinct long lines in dev and in test, and how many test ones are
    /// also in dev.
    pub dev_long_lines: u64,
    pub test_long_lines: u64,
    pub test_long_lines_in_dev: u64,
}

/// Everything [`prepare`] produced, written as `manifest.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub params: Config,
    /// Path relative to the output directory → statistics.
    pub files: BTreeMap<String, FileStats>,
    pub removals: BTreeMap<String, Removals>,
    pub checks: Checks,
}

fn zip_name(split: &str) -> (&'static str, &'static str) {
    match split {
        "train10M" => ("train_10M", "train"),
        "train100M" => ("train_100M", "train"),
        "dev" => ("dev", "dev"),
        _ => ("test", "test"),
    }
}

fn read_source(
    input: &Path,
    split: &str,
    source: &str,
    f: impl FnMut(Vec<String>) -> Result<()>,
) -> Result<()> {
    let (dir, ext) = zip_name(split);
    let zip_path = input.join(format!("{dir}.zip"));
    let entry = format!("{dir}/{source}.{ext}");
    let file = File::open(&zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))
        .with_context(|| format!("reading {}", zip_path.display()))?;
    let reader = archive
        .by_name(&entry)
        .with_context(|| format!("{} has no {entry}", zip_path.display()))?;
    for_each_doc(
        BufReader::with_capacity(1 << 20, reader),
        rule_for(source),
        &entry,
        f,
    )
}

/// Per document: `Some((hash of its id, amount))`, or `None` if it may not
/// be picked.
type Candidates = Vec<Option<(u64, u64)>>;

/// Picks documents in hash order until `quota` is reached; `info` is
/// (hash, amount) per document, `None` for ineligible ones.
fn pick_by_hash(info: &[Option<(u64, u64)>], quota: u64) -> HashSet<usize> {
    let mut order: Vec<(u64, usize, u64)> = info
        .iter()
        .enumerate()
        .filter_map(|(i, x)| x.map(|(h, amount)| (h, i, amount)))
        .collect();
    order.sort_unstable();
    let mut picked = HashSet::new();
    let mut total = 0;
    for (_, i, amount) in order {
        if total >= quota {
            break;
        }
        total += amount;
        picked.insert(i);
    }
    picked
}

/// Proportional share of `total` for a part of `whole`.
fn share(total: u64, part: u64, whole: u64) -> u64 {
    if whole == 0 {
        return 0;
    }
    ((total as u128 * part as u128 + whole as u128 / 2) / whole as u128) as u64
}

struct Writers<'a> {
    output: &'a Path,
    files: BTreeMap<String, FileStats>,
}

impl Writers<'_> {
    fn create(&self, rel: &str) -> Result<DocWriter> {
        DocWriter::create(&self.output.join(rel))
    }

    fn finish(&mut self, rel: String, w: DocWriter) -> Result<()> {
        let stats = w.finish()?;
        self.files.insert(rel, stats);
        Ok(())
    }
}

/// Runs the whole BabyLM preparation and writes `manifest.json`.
pub fn prepare(cfg: &Config) -> Result<Manifest> {
    let seed = cfg.seed;
    let mut out = Writers {
        output: &cfg.output,
        files: BTreeMap::new(),
    };
    let mut removals: BTreeMap<String, Removals> = SOURCES
        .iter()
        .map(|s| (s.to_string(), Removals::default()))
        .collect();
    let mut checks = Checks::default();

    // 1. Held-out sets, converted as they are. Collect long-line hashes.
    let mut dev_long = HashSet::new();
    let mut test_long = HashSet::new();
    // (split, source) → per-document (hash of id, bytes).
    let mut heldout_docs: BTreeMap<(&str, &str), Candidates> = BTreeMap::new();
    for split in ["dev", "test"] {
        for source in SOURCES {
            let rel = format!("{split}/{source}.docs");
            let mut w = out.create(&rel)?;
            let mut info = Vec::new();
            read_source(&cfg.input, split, source, |lines| {
                for line in lines.iter().filter(|l| is_long_line(l)) {
                    let h = hash64(seed, line.as_bytes());
                    if split == "dev" {
                        dev_long.insert(h);
                    } else {
                        test_long.insert(h);
                    }
                }
                let id = format!("{source}:{}", info.len());
                info.push(Some((hash64(seed, id.as_bytes()), lines_bytes(&lines))));
                w.write_doc(&lines)
            })?;
            out.finish(rel, w)?;
            heldout_docs.insert((split, source), info);
        }
    }
    checks.dev_long_lines = dev_long.len() as u64;
    checks.test_long_lines = test_long.len() as u64;
    checks.test_long_lines_in_dev = test_long.intersection(&dev_long).count() as u64;
    let heldout: HashSet<u64> = dev_long.union(&test_long).copied().collect();
    drop((dev_long, test_long));

    // 2. Fixed samples of the held-out sets, proportional to source bytes.
    for split in ["dev", "test"] {
        let source_bytes = |source: &'static str| -> u64 {
            heldout_docs
                .get(&(split, source))
                .map_or(0, |v| v.iter().flatten().map(|(_, b)| b).sum())
        };
        let total: u64 = SOURCES.iter().map(|s| source_bytes(s)).sum();
        for source in SOURCES {
            let quota = share(cfg.sample_bytes, source_bytes(source), total);
            let info = heldout_docs.get(&(split, source)).map_or(&[][..], |v| v);
            let picked = pick_by_hash(info, quota);
            let rel = format!("{split}-sample/{source}.docs");
            let mut w = out.create(&rel)?;
            let path = cfg.output.join(format!("{split}/{source}.docs"));
            for (i, doc) in DocReader::open(&path)?.enumerate() {
                let doc = doc?;
                if picked.contains(&i) {
                    w.write_doc(&doc.lines)?;
                }
            }
            out.finish(rel, w)?;
        }
    }

    // Removes held-out long lines from a training document.
    let clean = |lines: Vec<String>, removed: &mut u64| -> Vec<String> {
        lines
            .into_iter()
            .filter(|l| {
                let leak = is_long_line(l) && heldout.contains(&hash64(seed, l.as_bytes()));
                *removed += u64::from(leak);
                !leak
            })
            .collect()
    };

    // 3a. train_10M after overlap removal: per-document stats, the subset
    // sample, and the probe choice.
    let mut subset_sample: HashMap<u64, bool> = HashMap::new();
    let mut probe_info: BTreeMap<&str, Candidates> = BTreeMap::new();
    let mut clean_bytes: BTreeMap<&str, u64> = BTreeMap::new();
    for source in SOURCES {
        let mut info = Vec::new();
        let mut bytes = 0;
        let mut ignored = 0;
        read_source(&cfg.input, "train10M", source, |lines| {
            for line in &lines {
                let h = hash64(seed, line.as_bytes());
                if h.is_multiple_of(64) {
                    subset_sample.insert(h, is_long_line(line));
                }
            }
            let lines = clean(lines, &mut ignored);
            if lines.is_empty() {
                return Ok(());
            }
            let b = lines_bytes(&lines);
            bytes += b;
            let id = format!("train10M/{source}:{}", info.len());
            let worded = lines.iter().filter(|l| has_word(l)).count() as u64;
            info.push((b >= MIN_PROBE_DOC_BYTES).then(|| (hash64(seed, id.as_bytes()), worded)));
            Ok(())
        })?;
        probe_info.insert(source, info);
        clean_bytes.insert(source, bytes);
    }
    let total_clean: u64 = clean_bytes.values().sum();

    // 3b. Write the probe; remember each probe document's line hashes and the
    // set of long probe lines.
    let mut probe_picks: BTreeMap<&str, HashSet<usize>> = BTreeMap::new();
    let mut probe_seqs: BTreeMap<&str, Vec<Vec<u64>>> = BTreeMap::new();
    let mut probe_long: HashSet<u64> = HashSet::new();
    for source in SOURCES {
        let quota = share(
            cfg.probe_lines,
            clean_bytes.get(source).copied().unwrap_or(0),
            total_clean,
        );
        let info = probe_info.get(source).map_or(&[][..], |v| v);
        let picked = pick_by_hash(info, quota);
        let rel = format!("probe/{source}.docs");
        let mut w = out.create(&rel)?;
        let mut seqs = Vec::new();
        let mut k = 0usize;
        let mut ignored = 0;
        let r = removals.entry(source.to_string()).or_default();
        read_source(&cfg.input, "train10M", source, |lines| {
            let lines = clean(lines, &mut ignored);
            if lines.is_empty() {
                return Ok(());
            }
            k += 1;
            if !picked.contains(&(k - 1)) {
                return Ok(());
            }
            let hashes: Vec<u64> = lines.iter().map(|l| hash64(seed, l.as_bytes())).collect();
            for (line, h) in lines.iter().zip(&hashes) {
                if is_long_line(line) {
                    probe_long.insert(*h);
                }
            }
            seqs.push(hashes);
            r.probe_docs += 1;
            r.probe_lines += lines.len() as u64;
            r.probe_worded_lines += lines.iter().filter(|l| has_word(l)).count() as u64;
            w.write_doc(&lines)
        })?;
        out.finish(rel, w)?;
        probe_picks.insert(source, picked);
        probe_seqs.insert(source, seqs);
    }

    // 3c. Write train_10M without the probe documents and long probe lines.
    for source in SOURCES {
        let picked = probe_picks.get(source).cloned().unwrap_or_default();
        let rel = format!("train10M/{source}.docs");
        let mut w = out.create(&rel)?;
        let r = removals.entry(source.to_string()).or_default();
        let mut k = 0usize;
        let mut raw_docs = 0u64;
        read_source(&cfg.input, "train10M", source, |lines| {
            raw_docs += 1;
            let lines = clean(lines, &mut r.train10m_heldout_lines);
            if lines.is_empty() {
                return Ok(());
            }
            k += 1;
            if picked.contains(&(k - 1)) {
                return Ok(());
            }
            let lines: Vec<String> = lines
                .into_iter()
                .filter(|l| {
                    let repeat =
                        is_long_line(l) && probe_long.contains(&hash64(seed, l.as_bytes()));
                    r.train10m_probe_repeat_lines += u64::from(repeat);
                    !repeat
                })
                .collect();
            if lines.is_empty() {
                return Ok(());
            }
            w.write_doc(&lines)
        })?;
        r.train10m_dropped_docs = raw_docs.saturating_sub(r.probe_docs + w.stats().docs);
        out.finish(rel, w)?;
    }

    // 4. train_100M: remove held-out long lines, every occurrence of a probe
    // document (consecutive lines with the same content, matched across
    // document boundaries), and long probe lines elsewhere.
    let mut subset_found: HashSet<u64> = HashSet::new();
    for source in SOURCES {
        let seqs = probe_seqs.get(source).map_or(&[][..], |v| v);
        let mut by_last: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, seq) in seqs.iter().enumerate() {
            if let Some(last) = seq.last() {
                by_last.entry(*last).or_default().push(i);
            }
        }
        let mut found = vec![0u64; seqs.len()];
        let rel = format!("train100M/{source}.docs");
        let mut asm = Assembler {
            writer: out.create(&rel)?,
            doc: None,
            lines: Vec::new(),
        };
        let r = removals.entry(source.to_string()).or_default();
        let mut queue: VecDeque<Pending> = VecDeque::new();
        let mut raw_docs = 0u64;
        read_source(&cfg.input, "train100M", source, |lines| {
            let doc = raw_docs;
            raw_docs += 1;
            for line in lines {
                let hash = hash64(seed, line.as_bytes());
                if subset_sample.contains_key(&hash) {
                    subset_found.insert(hash);
                }
                let long = is_long_line(&line);
                if long && heldout.contains(&hash) {
                    r.train100m_heldout_lines += 1;
                    continue;
                }
                queue.push_back(Pending {
                    line,
                    hash,
                    long,
                    doc,
                    probe: false,
                });
                for &p in by_last.get(&hash).map_or(&[][..], |v| v) {
                    let Some(seq) = seqs.get(p) else { continue };
                    if seq.len() <= queue.len()
                        && queue
                            .iter()
                            .rev()
                            .zip(seq.iter().rev())
                            .all(|(q, s)| q.hash == *s)
                    {
                        if let Some(n) = found.get_mut(p) {
                            *n += 1;
                        }
                        r.train100m_probe_matches += 1;
                        queue
                            .iter_mut()
                            .rev()
                            .take(seq.len())
                            .for_each(|q| q.probe = true);
                    }
                }
                while queue.len() > MAX_DOC_LINES {
                    if let Some(p) = queue.pop_front() {
                        asm.emit(p, &probe_long, r)?;
                    }
                }
            }
            Ok(())
        })?;
        while let Some(p) = queue.pop_front() {
            asm.emit(p, &probe_long, r)?;
        }
        asm.flush()?;
        r.probe_docs_missing_from_100m = found.iter().filter(|n| **n == 0).count() as u64;
        r.train100m_dropped_docs = raw_docs.saturating_sub(asm.writer.stats().docs);
        out.finish(rel, asm.writer)?;
    }
    checks.subset_sampled = subset_sample.len() as u64;
    checks.subset_sampled_long = subset_sample.values().filter(|l| **l).count() as u64;
    checks.subset_found = subset_found.len() as u64;
    checks.subset_found_long = subset_found
        .iter()
        .filter(|h| subset_sample.get(h).copied().unwrap_or(false))
        .count() as u64;

    let manifest = Manifest {
        params: cfg.clone(),
        files: out.files,
        removals,
        checks,
    };
    let path = cfg.output.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(manifest)
}

struct Pending {
    line: String,
    hash: u64,
    long: bool,
    doc: u64,
    probe: bool,
}

/// Regroups surviving train_100M lines into their original documents.
struct Assembler {
    writer: DocWriter,
    doc: Option<u64>,
    lines: Vec<String>,
}

impl Assembler {
    fn emit(&mut self, p: Pending, probe_long: &HashSet<u64>, r: &mut Removals) -> Result<()> {
        if p.probe {
            r.train100m_probe_lines += 1;
            return Ok(());
        }
        if p.long && probe_long.contains(&p.hash) {
            r.train100m_probe_repeat_lines += 1;
            return Ok(());
        }
        if self.doc != Some(p.doc) {
            self.flush()?;
            self.doc = Some(p.doc);
        }
        self.lines.push(p.line);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.lines.is_empty() {
            self.writer.write_doc(&self.lines)?;
            self.lines.clear();
        }
        Ok(())
    }
}

/// A plain-text summary of a manifest.
pub fn summary(m: &Manifest) -> String {
    let mut s = format!(
        "{:<34} {:>12} {:>10} {:>8} {:>12}\n",
        "file", "bytes", "lines", "docs", "words"
    );
    for (path, f) in &m.files {
        s += &format!(
            "{path:<34} {:>12} {:>10} {:>8} {:>12}\n",
            f.bytes, f.lines, f.docs, f.words
        );
    }
    s += &format!(
        "\n{:<15} {:>9} {:>9} {:>6} {:>6} {:>7} {:>8} {:>8} {:>7} {:>7}\n",
        "source",
        "10M held",
        "100M held",
        "probe",
        "lines",
        "worded",
        "10M rep",
        "100M seq",
        "100Mrep",
        "missing"
    );
    for (source, r) in &m.removals {
        s += &format!(
            "{source:<15} {:>9} {:>9} {:>6} {:>6} {:>7} {:>8} {:>8} {:>7} {:>7}\n",
            r.train10m_heldout_lines,
            r.train100m_heldout_lines,
            r.probe_docs,
            r.probe_lines,
            r.probe_worded_lines,
            r.train10m_probe_repeat_lines,
            r.train100m_probe_lines,
            r.train100m_probe_repeat_lines,
            r.probe_docs_missing_from_100m
        );
    }
    let c = &m.checks;
    s += &format!(
        "\ntrain_10M lines found in train_100M: {}/{} sampled ({}/{} long)\n\
         long test lines also in dev: {}/{} (dev has {})\n",
        c.subset_found,
        c.subset_sampled,
        c.subset_found_long,
        c.subset_sampled_long,
        c.test_long_lines_in_dev,
        c.test_long_lines,
        c.dev_long_lines
    );
    s
}
