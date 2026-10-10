//! Parallel analysis of documents with a model.

use rayon::prelude::*;
use sememe_data::Doc;
use sememe_engine::context::Context;
use sememe_engine::cost::{Bits, Event};
use sememe_engine::model::Model;
use sememe_engine::parse::{LineAnalysis, Parser, Sym};

/// The analysis of one document.
#[derive(Clone, Debug, Default)]
pub struct DocResult {
    /// Exact bits.
    pub bits: Bits,
    /// Original bytes.
    pub bytes: u64,
    /// Table events, in order.
    pub events: Vec<Event>,
    /// Bits outside tables.
    pub fixed: Bits,
    /// Unknown-word leaves.
    pub unknowns: u32,
    /// Words.
    pub words: u32,
    /// Line analyses, when kept.
    pub lines: Vec<LineAnalysis>,
    /// Symbol sources.
    pub syms: Vec<(u64, Sym)>,
    /// Lines whose best analysis has exactly one unknown word.
    pub singles: Vec<Single>,
}

/// A line that parses except for one unknown word: the rest of the parse
/// fixes the category that word must have.
#[derive(Clone, Debug)]
pub struct Single {
    /// Line index in the document.
    pub line: u32,
    /// The unknown word.
    pub form: String,
    /// Its category in the analysis.
    pub cat: sememe_engine::category::Cat,
    /// Its unknown meaning.
    pub sem: sememe_engine::term::Term,
}

/// Analyze one document from a fresh context.
pub fn analyze_doc(parser: &mut Parser, doc: &Doc, keep_lines: bool) -> DocResult {
    let mut ctx = Context::new();
    let mut r = DocResult {
        bytes: sememe_data::line_bytes(doc),
        ..DocResult::default()
    };
    for (li, line) in doc.lines.iter().enumerate() {
        let mut a = parser.parse_line(&mut ctx, line);
        if a.words.len() >= 2 && a.unknowns() == 1 {
            for f in &a.fragments {
                for leaf in f.deriv.leaves() {
                    if let sememe_engine::parse::Deriv::Leaf {
                        kind: sememe_engine::parse::LeafKind::Unknown { infl: None },
                        stem,
                        cat,
                        sem,
                        ..
                    } = leaf
                    {
                        r.singles.push(Single {
                            line: li as u32,
                            form: stem.clone(),
                            cat: cat.clone(),
                            sem: sem.clone(),
                        });
                    }
                }
            }
        }
        r.bits += a.bits;
        r.fixed += a.fixed;
        r.unknowns += a.unknowns() as u32;
        r.words += a.words.len() as u32;
        r.events.append(&mut a.events);
        r.syms.append(&mut a.syms);
        if keep_lines {
            r.lines.push(a);
        }
    }
    r
}

/// Analyze documents in parallel; results are in input order.
pub fn analyze(model: &Model, docs: &[&Doc], keep_lines: bool) -> Vec<DocResult> {
    docs.par_iter()
        .map_init(|| Parser::new(model), |p, d| analyze_doc(p, d, keep_lines))
        .collect()
}
