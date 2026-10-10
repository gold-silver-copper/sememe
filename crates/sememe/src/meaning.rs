//! Does the program learn meaning, or relabelled words?
//!
//! - **Probe agreement:** sentences from the training sources (removed from
//!   the training set) annotated once by the teacher; how much of the
//!   teacher's meaning the program's cheapest meaning recovers (a
//!   Smatch-style F1 over heads, features and role edges, under the best
//!   greedy tree alignment).
//! - **Paraphrase agreement:** pairs the teacher wrote to mean the same;
//!   how often the program gives both the same meaning.
//! - **One-word concepts:** the share of concepts expressed by exactly one
//!   word form.
//! - **Teacher self-agreement** on double-annotated passages.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;
use sememe_engine::program::Program;
use sememe_engine::term::{Deixis, Head, Term, TermNames};
use serde::{Deserialize, Serialize};

use crate::annotate::{AnnHead, AnnNode};

/// A meaning tree by names, so the teacher's meanings (whose concepts may
/// not be in the program) and the program's can be compared.
#[derive(Clone, Debug, PartialEq)]
pub struct NTree {
    /// Head: a concept name, `@speaker`, `@addressee`, `@other`, `Query`,
    /// `Unknown` or `Missing`.
    pub head: String,
    /// `feature=value`.
    pub feats: Vec<String>,
    /// Role children.
    pub kids: Vec<(String, NTree)>,
}

impl NTree {
    /// From a program meaning.
    pub fn from_term(p: &Program, t: &Term) -> Option<NTree> {
        let n = t.as_node()?;
        let head = match &n.head {
            Head::Concept(c) => p.concept_name(*c),
            Head::Entity(e) => p.entity_name(*e),
            Head::Unknown => "Unknown".into(),
            Head::Missing => "Missing".into(),
            Head::Query => "Query".into(),
            Head::Ref(Deixis::Speaker) => "@speaker".into(),
            Head::Ref(Deixis::Addressee) => "@addressee".into(),
            Head::Ref(Deixis::Other) => "@other".into(),
        };
        let feats = n
            .feats
            .iter()
            .map(|(f, v)| format!("{}={}", p.feat_name(*f), p.val_name(*f, *v)))
            .collect();
        let kids = n
            .roles
            .iter()
            .filter_map(|(r, c)| Some((p.role_name(*r), NTree::from_term(p, c)?)))
            .collect();
        Some(NTree { head, feats, kids })
    }

    /// From a teacher annotation.
    pub fn from_ann(a: &AnnNode) -> NTree {
        let head = match &a.head {
            AnnHead::Concept(c) => c.clone(),
            AnnHead::Query => "Query".into(),
            AnnHead::Speaker => "@speaker".into(),
            AnnHead::Addressee => "@addressee".into(),
            AnnHead::Other | AnnHead::Ref(_) => "@other".into(),
        };
        NTree {
            head,
            feats: a.feats.iter().map(|(f, v, _)| format!("{f}={v}")).collect(),
            kids: a
                .roles
                .iter()
                .map(|(r, _, c)| (r.clone(), NTree::from_ann(c)))
                .collect(),
        }
    }

    fn size(&self) -> f64 {
        1.0 + self.feats.len() as f64 + self.kids.iter().map(|(_, c)| 1.0 + c.size()).sum::<f64>()
    }
}

/// Matched and total "triples" (head, features, role edges) of two trees.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overlap {
    /// Matched triples.
    pub matched: f64,
    /// Triples in the first tree.
    pub a: f64,
    /// Triples in the second tree.
    pub b: f64,
}

impl Overlap {
    /// F1 of the overlap.
    pub fn f1(&self) -> f64 {
        if self.a + self.b == 0.0 {
            1.0
        } else {
            2.0 * self.matched / (self.a + self.b)
        }
    }
    /// Accumulate.
    pub fn add(&mut self, o: Overlap) {
        self.matched += o.matched;
        self.a += o.a;
        self.b += o.b;
    }
}

fn heads_match(a: &str, b: &str) -> bool {
    a == b || (a.starts_with('@') && b == "@other") || (b.starts_with('@') && a == "@other")
}

fn matched(a: &NTree, b: &NTree) -> f64 {
    let mut m = if heads_match(&a.head, &b.head) {
        1.0
    } else {
        0.0
    };
    m += a.feats.iter().filter(|f| b.feats.contains(f)).count() as f64;
    let mut used = vec![false; b.kids.len()];
    for (r, ac) in &a.kids {
        let mut best: Option<(f64, usize)> = None;
        for (i, (rb, bc)) in b.kids.iter().enumerate() {
            if rb != r || used.get(i).copied().unwrap_or(true) {
                continue;
            }
            let s = 1.0 + matched(ac, bc);
            if best.is_none_or(|(x, _)| s > x) {
                best = Some((s, i));
            }
        }
        if let Some((s, i)) = best {
            m += s;
            if let Some(u) = used.get_mut(i) {
                *u = true;
            }
        }
    }
    m
}

/// Smatch-style overlap of two meaning trees.
pub fn overlap(a: Option<&NTree>, b: Option<&NTree>) -> Overlap {
    Overlap {
        matched: match (a, b) {
            (Some(x), Some(y)) => matched(x, y),
            _ => 0.0,
        },
        a: a.map_or(0.0, NTree::size),
        b: b.map_or(0.0, NTree::size),
    }
}

/// The share of concepts (used by entries) expressed by exactly one word form.
pub fn one_word_share(p: &Program) -> f64 {
    let mut forms: FxHashMap<u32, Vec<String>> = FxHashMap::default();
    for e in p.entries.values() {
        if let Some(n) = e.sem.body().as_node()
            && let Head::Concept(c) = n.head
        {
            let v = forms.entry(c).or_default();
            if !v.contains(&e.form) {
                v.push(e.form.clone());
            }
        }
    }
    if forms.is_empty() {
        return 0.0;
    }
    forms.values().filter(|v| v.len() == 1).count() as f64 / forms.len() as f64
}

/// A paraphrase pair written by the teacher.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Paraphrase {
    /// First wording.
    pub a: String,
    /// Second wording.
    pub b: String,
}

/// Meaning metrics for a round.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MeaningMetrics {
    /// Probe F1 against the teacher's meanings.
    pub probe_f1: f64,
    /// Probe units scored.
    pub probe_units: u32,
    /// Share of paraphrase pairs given identical meanings.
    pub paraphrase_same: f64,
    /// Mean F1 between the meanings of paraphrase pairs.
    pub paraphrase_f1: f64,
    /// Share of concepts with exactly one word form.
    pub one_word_concepts: f64,
    /// Teacher self-agreement F1 on double annotations (cumulative).
    pub teacher_agreement: f64,
    /// By source file: probe F1.
    pub probe_by_source: BTreeMap<String, f64>,
}
