//! Data preparation for sememe.
//!
//! Everything downstream reads the prepared `.docs` format through
//! [`DocReader`]; see `README.md` in this crate for the format, the
//! pipeline and the commands.

pub mod babylm;
pub mod blimp;
pub mod docs;
pub mod enwik;
pub mod split;
pub mod unpc;

pub use docs::{
    Doc, DocReader, DocWriter, FileStats, SEPARATOR, file_stats, hash64, line_bytes, lines_bytes,
    word_count,
};
