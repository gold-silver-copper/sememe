//! Stage 4: the UN Parallel Corpus, English–Russian.
//!
//! The Moses files have no document boundaries, so the corpus is cut into
//! consecutive blocks of `block_lines` aligned lines; each block is one
//! document. Each block goes to exactly one place, chosen by
//! `hash64(seed, "unpc:<block index>")`:
//! - English held-out, Russian held-out (one language only, ~1 MB each);
//! - the aligned sample (both languages, ~`aligned_bytes` of English);
//! - otherwise a second hash bit sends it to the English pool or the Russian
//!   pool, where it is kept with the probability that yields ~`pool_bytes`.
//!
//! So no block is ever seen in both languages outside the aligned sample.
//! Lines are un-escaped from OPUS's Moses export (see [`unescape_moses`]), so
//! they are not byte-identical to the zip's text.
//! Probabilities come from the uncompressed sizes in the zip, so the whole
//! corpus is sampled uniformly in one streaming pass.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::docs::{DocWriter, FileStats, SEPARATOR, hash64};

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    /// `data/unpc`
    pub input: PathBuf,
    /// `data/prepared/unpc`
    pub output: PathBuf,
    /// Size of each single-language training pool.
    pub pool_bytes: u64,
    /// Size of each language's held-out set.
    pub heldout_bytes: u64,
    /// Size of the aligned sample (English side).
    pub aligned_bytes: u64,
    pub block_lines: usize,
    /// Stop after this many lines (for trying it on a slice).
    pub max_lines: Option<u64>,
    pub seed: u64,
}

impl Config {
    pub fn new(input: PathBuf, output: PathBuf) -> Self {
        Self {
            input,
            output,
            pool_bytes: 100_000_000,
            heldout_bytes: 1_000_000,
            aligned_bytes: 10_000_000,
            block_lines: 200,
            max_lines: None,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub params: Config,
    pub lines_read: u64,
    pub blocks: u64,
    /// Blocks skipped because a line was invalid UTF-8 or was exactly U+001E.
    pub skipped_blocks: u64,
    pub files: std::collections::BTreeMap<String, FileStats>,
}

fn open_archive(path: &std::path::Path) -> Result<zip::ZipArchive<BufReader<File>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    zip::ZipArchive::new(BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))
}

fn read_block<R: BufRead>(r: &mut R, n: usize, out: &mut Vec<Vec<u8>>) -> Result<()> {
    out.clear();
    for _ in 0..n {
        let mut buf = Vec::new();
        if r.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        if buf.last() == Some(&b'\n') {
            buf.pop();
        }
        out.push(buf);
    }
    Ok(())
}

fn to_lines(block: &[Vec<u8>]) -> Option<Vec<String>> {
    block
        .iter()
        .map(|b| {
            std::str::from_utf8(b)
                .ok()
                .map(unescape_moses)
                .filter(|s| s != SEPARATOR)
        })
        .collect()
}

/// Undoes the Moses escaping in OPUS's UNPC text: `&quot;` `&lt;` `&gt;`
/// `&#124;` `&#91;` `&#93;` `&amp;` become the characters, and the
/// apostrophe, exported as ` &apos; ` with spaces around it, is rejoined:
/// `Finland &apos; s` → `Finland's`, after a one-letter word (`d &apos;
/// Ivoire`) → `d'Ivoire`, otherwise `States &apos; rights` → `States'
/// rights`. Other spacing the export added (`&quot; ,`) is left alone.
pub fn unescape_moses(line: &str) -> String {
    const SPACED: &str = " &apos; ";
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find(SPACED) {
        let before = rest.get(..i).unwrap_or("");
        let after = rest.get(i + SPACED.len()..).unwrap_or("");
        out.push_str(before);
        let prev_word = before.rsplit(' ').next().unwrap_or("");
        let possessive_s = after
            .strip_prefix('s')
            .is_some_and(|t| t.chars().next().is_none_or(|c| !c.is_alphanumeric()));
        out.push('\'');
        if !possessive_s && prev_word.chars().count() != 1 {
            out.push(' ');
        }
        rest = after;
    }
    out.push_str(rest);
    out.replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#124;", "|")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

struct Out {
    name: &'static str,
    writer: DocWriter,
    target: u64,
}

impl Out {
    fn write(&mut self, lines: &[String]) -> Result<()> {
        if self.writer.stats().bytes < self.target {
            self.writer.write_doc(lines)?;
        }
        Ok(())
    }
}

/// Runs the UNPC preparation and writes `manifest.json`.
pub fn prepare(cfg: &Config) -> Result<Manifest> {
    let zip_path = cfg.input.join("en-ru.txt.zip");
    // Two archive handles, so both languages stream in lockstep.
    let mut en_zip = open_archive(&zip_path)?;
    let mut ru_zip = open_archive(&zip_path)?;
    let en_entry = en_zip.by_name("UNPC.en-ru.en")?;
    let ru_entry = ru_zip.by_name("UNPC.en-ru.ru")?;
    let (en_size, ru_size) = (en_entry.size(), ru_entry.size());
    let mut en = BufReader::with_capacity(1 << 20, en_entry);
    let mut ru = BufReader::with_capacity(1 << 20, ru_entry);
    let (e, r) = (en_size.max(1) as f64, ru_size.max(1) as f64);
    let p_ho_en = cfg.heldout_bytes as f64 / e;
    let p_ho_ru = cfg.heldout_bytes as f64 / r;
    let p_aligned = cfg.aligned_bytes as f64 / e;
    let p_pool_en = cfg.pool_bytes as f64 / (e / 2.0);
    let p_pool_ru = cfg.pool_bytes as f64 / (r / 2.0);

    let mk = |name: &'static str, target: u64| -> Result<Out> {
        Ok(Out {
            name,
            writer: DocWriter::create(&cfg.output.join(format!("{name}.docs")))?,
            target,
        })
    };
    let mut en_heldout = mk("en-heldout", cfg.heldout_bytes)?;
    let mut ru_heldout = mk("ru-heldout", cfg.heldout_bytes)?;
    let mut aligned_en = mk("aligned-en", cfg.aligned_bytes)?;
    let mut aligned_ru = mk("aligned-ru", u64::MAX)?;
    let mut en_train = mk("en-train", cfg.pool_bytes)?;
    let mut ru_train = mk("ru-train", cfg.pool_bytes)?;

    let (mut eb, mut rb) = (Vec::new(), Vec::new());
    let mut lines_read = 0u64;
    let mut blocks = 0u64;
    let mut skipped = 0u64;
    loop {
        if cfg.max_lines.is_some_and(|m| lines_read >= m) {
            break;
        }
        read_block(&mut en, cfg.block_lines, &mut eb)?;
        read_block(&mut ru, cfg.block_lines, &mut rb)?;
        if eb.len() != rb.len() {
            bail!(
                "UNPC files are not aligned: block {blocks} has {} English and {} Russian lines",
                eb.len(),
                rb.len()
            );
        }
        if eb.is_empty() {
            break;
        }
        lines_read += eb.len() as u64;
        let key = format!("unpc:{blocks}");
        blocks += 1;
        let (Some(el), Some(rl)) = (to_lines(&eb), to_lines(&rb)) else {
            skipped += 1;
            continue;
        };
        let h = hash64(cfg.seed, key.as_bytes());
        let u = unit(h);
        if u < p_ho_en {
            en_heldout.write(&el)?;
        } else if u < p_ho_en + p_ho_ru {
            ru_heldout.write(&rl)?;
        } else if u < p_ho_en + p_ho_ru + p_aligned {
            if aligned_en.writer.stats().bytes < aligned_en.target {
                aligned_en.writer.write_doc(&el)?;
                aligned_ru.writer.write_doc(&rl)?;
            }
        } else {
            let v = hash64(cfg.seed.wrapping_add(1), key.as_bytes());
            if v & 1 == 0 {
                if unit(v) < p_pool_en {
                    en_train.write(&el)?;
                }
            } else if unit(v) < p_pool_ru {
                ru_train.write(&rl)?;
            }
        }
    }

    let mut files = std::collections::BTreeMap::new();
    for out in [
        en_heldout, ru_heldout, aligned_en, aligned_ru, en_train, ru_train,
    ] {
        files.insert(format!("{}.docs", out.name), out.writer.finish()?);
    }
    let manifest = Manifest {
        params: cfg.clone(),
        lines_read,
        blocks,
        skipped_blocks: skipped,
        files,
    };
    let path = cfg.output.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(manifest)
}
