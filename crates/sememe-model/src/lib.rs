//! The learned sememe program, exported.
//!
//! The ontology is Rust types (module [`ontology`]): every concept is a
//! type with its doc comment, and a concept with subtypes is an enum over
//! them, so the compiler checks the hierarchy. The lexicon, grammar, counts
//! and facts are data: built from the committed text program into a blob
//! and interpreted by the engine. No LLM, no network, deterministic.

use std::sync::{Arc, OnceLock};

pub use sememe_engine;
use sememe_engine::model::Model;
use sememe_engine::program::{Program, ProgramFiles};

/// A concept type of the ontology.
pub trait Concept {
    /// Stable concept ID in the program.
    const ID: u32;
    /// Concept name.
    const NAME: &'static str;
}

/// The ontology as types.
#[allow(missing_docs, clippy::all)]
pub mod ontology {
    include!(concat!(env!("OUT_DIR"), "/ontology.rs"));
}

static BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/program.bin"));

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = b.split_at_checked(n)?;
    *b = rest;
    Some(head)
}

/// The program's text files, unpacked from the blob.
pub fn files() -> ProgramFiles {
    let mut out = ProgramFiles::new();
    let mut b = BLOB;
    while !b.is_empty() {
        let Some(n) = take(&mut b, 4)
            .and_then(|x| x.try_into().ok())
            .map(u32::from_le_bytes)
        else {
            break;
        };
        let Some(name) = take(&mut b, n as usize).map(|x| String::from_utf8_lossy(x).into_owned())
        else {
            break;
        };
        let Some(len) = take(&mut b, 8)
            .and_then(|x| x.try_into().ok())
            .map(u64::from_le_bytes)
        else {
            break;
        };
        let Some(text) =
            take(&mut b, len as usize).map(|x| String::from_utf8_lossy(x).into_owned())
        else {
            break;
        };
        out.insert(name, text);
    }
    out
}

/// The program.
pub fn program() -> Arc<Program> {
    static P: OnceLock<Arc<Program>> = OnceLock::new();
    P.get_or_init(|| Arc::new(Program::from_files(&files()).unwrap_or_else(|_| Program::seed())))
        .clone()
}

/// The compiled model.
pub fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| Model::new(program()))
}
