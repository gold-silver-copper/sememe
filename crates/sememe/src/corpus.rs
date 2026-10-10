//! Loading prepared document sets.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use sememe_data::{Doc, DocReader};

/// A loaded set of documents.
pub struct Corpus {
    /// Documents in file order.
    pub docs: Vec<Doc>,
    /// Source file stem of each document.
    pub source: Vec<String>,
}

impl Corpus {
    /// Load every `.docs` file in a directory (sorted by name), or a single
    /// `.docs` file.
    pub fn load(path: &Path) -> Result<Corpus> {
        let mut files: Vec<PathBuf> = if path.is_dir() {
            std::fs::read_dir(path)
                .with_context(|| format!("reading {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "docs"))
                .collect()
        } else {
            vec![path.to_path_buf()]
        };
        files.sort();
        let mut docs = Vec::new();
        let mut source = Vec::new();
        for f in files {
            let stem = sememe_data::docs::file_stem(&f);
            for d in DocReader::open(&f)? {
                docs.push(d?);
                source.push(stem.clone());
            }
        }
        Ok(Corpus { docs, source })
    }

    /// Original bytes (each line plus its newline).
    pub fn bytes(&self) -> u64 {
        self.docs.iter().map(sememe_data::line_bytes).sum()
    }

    /// Number of lines.
    pub fn lines(&self) -> usize {
        self.docs.iter().map(|d| d.lines.len()).sum()
    }
}
