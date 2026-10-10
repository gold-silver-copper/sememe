//! The prepared `.docs` format.
//!
//! A `.docs` file is UTF-8 text. Documents are runs of lines, separated by a
//! line holding exactly U+001E (record separator) and nothing else. Lines keep
//! their original bytes minus the trailing `\n` (a `\r` stays). The file ends
//! with a `\n` after the last line; there is no separator after the last
//! document, and no document is empty.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The separator line between documents.
pub const SEPARATOR: &str = "\u{1e}";

/// One document: an id (`<file stem>:<doc index>`) and its lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doc {
    pub id: String,
    pub lines: Vec<String>,
}

/// Original-byte count of a document: each line plus its newline.
pub fn line_bytes(doc: &Doc) -> u64 {
    lines_bytes(&doc.lines)
}

/// Original-byte count of a run of lines: each line plus its newline.
pub fn lines_bytes<S: AsRef<str>>(lines: &[S]) -> u64 {
    lines.iter().map(|l| l.as_ref().len() as u64 + 1).sum()
}

/// Number of whitespace-separated words in a line.
pub fn word_count(line: &str) -> u64 {
    line.split_whitespace().count() as u64
}

/// Streaming reader over the documents of a `.docs` file.
pub struct DocReader<R> {
    reader: R,
    stem: String,
    index: usize,
    line_no: u64,
    done: bool,
}

impl DocReader<BufReader<File>> {
    /// Opens a `.docs` file. Document ids are `<file stem>:<doc index>`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Ok(Self::new(BufReader::new(file), file_stem(path)))
    }
}

impl<R: BufRead> DocReader<R> {
    /// Reads documents from any buffered reader; `stem` prefixes the ids.
    pub fn new(reader: R, stem: impl Into<String>) -> Self {
        Self {
            reader,
            stem: stem.into(),
            index: 0,
            line_no: 0,
            done: false,
        }
    }

    fn read_doc(&mut self) -> Result<Option<Doc>> {
        let mut lines = Vec::new();
        let mut buf = Vec::new();
        loop {
            buf.clear();
            if self.reader.read_until(b'\n', &mut buf)? == 0 {
                self.done = true;
                if lines.is_empty() {
                    if self.index > 0 {
                        bail!("{}: separator after the last document", self.stem);
                    }
                    return Ok(None);
                }
                break;
            }
            self.line_no += 1;
            if buf.last() == Some(&b'\n') {
                buf.pop();
            }
            let line = String::from_utf8(std::mem::take(&mut buf)).with_context(|| {
                format!("{}: invalid UTF-8 on line {}", self.stem, self.line_no)
            })?;
            if line == SEPARATOR {
                if lines.is_empty() {
                    bail!("{}: empty document at line {}", self.stem, self.line_no);
                }
                break;
            }
            lines.push(line);
        }
        let id = format!("{}:{}", self.stem, self.index);
        self.index += 1;
        Ok(Some(Doc { id, lines }))
    }
}

impl<R: BufRead> Iterator for DocReader<R> {
    type Item = Result<Doc>;

    fn next(&mut self) -> Option<Result<Doc>> {
        if self.done {
            return None;
        }
        match self.read_doc() {
            Ok(Some(doc)) => Some(Ok(doc)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Size and checksum of one `.docs` file, as recorded in manifests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStats {
    /// Original-byte count: sum over lines of (len + 1).
    pub bytes: u64,
    pub lines: u64,
    pub docs: u64,
    /// Whitespace-separated words.
    pub words: u64,
    /// sha256 of the `.docs` file itself.
    pub sha256: String,
}

/// Streaming writer of a `.docs` file.
pub struct DocWriter {
    out: BufWriter<File>,
    hasher: Sha256,
    stats: FileStats,
    path: PathBuf,
}

impl DocWriter {
    /// Creates (or truncates) the file, creating parent directories.
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self {
            out: BufWriter::with_capacity(1 << 20, file),
            hasher: Sha256::new(),
            stats: FileStats::default(),
            path: path.to_path_buf(),
        })
    }

    /// Appends one document. Fails on an empty document, a line containing
    /// `\n`, or a line that is exactly the separator (not representable).
    pub fn write_doc<S: AsRef<str>>(&mut self, lines: &[S]) -> Result<()> {
        if lines.is_empty() {
            bail!("{}: empty document", self.path.display());
        }
        for line in lines {
            let line = line.as_ref();
            if line == SEPARATOR || line.contains('\n') {
                bail!("{}: line not representable: {line:?}", self.path.display());
            }
        }
        if self.stats.docs > 0 {
            self.put(SEPARATOR.as_bytes())?;
            self.put(b"\n")?;
        }
        for line in lines {
            let line = line.as_ref();
            self.put(line.as_bytes())?;
            self.put(b"\n")?;
            self.stats.bytes += line.len() as u64 + 1;
            self.stats.words += word_count(line);
        }
        self.stats.lines += lines.len() as u64;
        self.stats.docs += 1;
        Ok(())
    }

    /// Statistics so far (sha256 is filled in by [`DocWriter::finish`]).
    pub fn stats(&self) -> &FileStats {
        &self.stats
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.hasher.update(bytes);
        self.out
            .write_all(bytes)
            .with_context(|| format!("writing {}", self.path.display()))
    }

    /// Flushes the file and returns its statistics.
    pub fn finish(mut self) -> Result<FileStats> {
        self.out
            .flush()
            .with_context(|| format!("flushing {}", self.path.display()))?;
        self.stats.sha256 = hex(&self.hasher.finalize());
        Ok(self.stats)
    }
}

/// Reads a `.docs` file back and computes its statistics.
pub fn file_stats(path: &Path) -> Result<FileStats> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hashing = HashingReader {
        inner: file,
        hasher: Sha256::new(),
    };
    let mut stats = FileStats::default();
    for doc in DocReader::new(BufReader::new(&mut hashing), file_stem(path)) {
        let doc = doc?;
        stats.docs += 1;
        stats.lines += doc.lines.len() as u64;
        stats.bytes += line_bytes(&doc);
        stats.words += doc.lines.iter().map(|l| word_count(l)).sum::<u64>();
    }
    stats.sha256 = hex(&hashing.hasher.finalize());
    Ok(stats)
}

struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if let Some(read) = buf.get(..n) {
            self.hasher.update(read);
        }
        Ok(n)
    }
}

/// A stable 64-bit hash: the first 8 bytes (little-endian) of
/// `sha256(seed as 8 little-endian bytes ++ bytes)`. `tools/prep_parquet.py`
/// computes the same function.
pub fn hash64(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(bytes);
    let digest = h.finalize();
    digest
        .as_slice()
        .first_chunk::<8>()
        .map_or(0, |c| u64::from_le_bytes(*c))
}

/// The file stem of a path, as a string (empty if none).
pub fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_reproduces_source_bytes() {
        // Not /tmp (RAM-backed on the dev machine): the workspace target dir.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/sememe-data-test-docs");
        let path = dir.join("held.docs");
        let source = "first line\r\n  indented \u{1e} not a separator\n\nnext doc\n";
        let docs = vec![
            vec!["first line\r", "  indented \u{1e} not a separator", ""],
            vec!["next doc"],
        ];
        let mut w = DocWriter::create(&path).unwrap();
        for d in &docs {
            w.write_doc(d).unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.bytes, source.len() as u64);
        assert_eq!(stats.docs, 2);
        assert_eq!(file_stats(&path).unwrap(), stats);

        let read: Vec<Doc> = DocReader::open(&path)
            .unwrap()
            .map(|d| d.unwrap())
            .collect();
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].id, "held:1");
        let lines: Vec<&str> = read
            .iter()
            .flat_map(|d| d.lines.iter().map(|s| s.as_str()))
            .collect();
        assert_eq!(lines, docs.concat());
        let rebuilt: String = lines.iter().map(|l| format!("{l}\n")).collect();
        assert_eq!(rebuilt, source);
        assert_eq!(
            read.iter().map(line_bytes).sum::<u64>(),
            source.len() as u64
        );

        assert!(w_rejects(&dir.join("bad.docs"), &[SEPARATOR]));
        assert!(w_rejects(&dir.join("bad.docs"), &[]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn w_rejects(path: &Path, lines: &[&str]) -> bool {
        DocWriter::create(path).unwrap().write_doc(lines).is_err()
    }
}
