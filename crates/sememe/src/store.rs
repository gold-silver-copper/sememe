//! Reading and writing program directories.

use std::path::Path;

use anyhow::{Context as _, Result, anyhow};
use sememe_engine::program::{Program, ProgramFiles};

/// Read a program from a directory of `.sem` files and `counts.tsv`.
pub fn load(dir: &Path) -> Result<Program> {
    let mut files = ProgramFiles::new();
    for name in [
        "ontology.sem",
        "grammar.sem",
        "lexicon.sem",
        "knowledge.sem",
        "counts.tsv",
    ] {
        let path = dir.join(name);
        if path.exists() {
            files.insert(
                name.to_string(),
                std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?,
            );
        }
    }
    Program::from_files(&files).map_err(|e| anyhow!("{}: {e}", dir.display()))
}

/// Write a program to a directory.
pub fn save(p: &Program, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, text) in p.to_files() {
        std::fs::write(dir.join(&name), text).with_context(|| format!("writing {name}"))?;
    }
    Ok(())
}
