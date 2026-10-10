//! BLiMP: minimal pairs. The model passes a pair when the grammatical
//! sentence costs fewer bits. Paradigms are split into gating ones (which
//! guard every round and are never reported as results) and reported ones.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use sememe_engine::context::Context;
use sememe_engine::cost::Bits;
use sememe_engine::model::Model;
use sememe_engine::parse::{LineAnalysis, Parser};
use serde::{Deserialize, Serialize};

/// One pair.
#[derive(Clone, Debug, Deserialize)]
pub struct Pair {
    /// Paradigm (one template).
    pub paradigm: String,
    /// Phenomenon (group of paradigms).
    pub phenomenon: String,
    /// Grammatical sentence.
    pub good: String,
    /// Ungrammatical sentence.
    pub bad: String,
    /// Whether this pair's paradigm gates changes.
    #[serde(default)]
    pub gating: bool,
    /// From the supplement.
    #[serde(default)]
    pub supplement: bool,
}

/// Load the filtered BLiMP pairs and the supplement with their split.
pub fn load(dir: &Path) -> Result<Vec<Pair>> {
    let split: BTreeMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("split.json"))?)?;
    let mut out = Vec::new();
    for (file, supp) in [("filtered.jsonl", false), ("supplement.jsonl", true)] {
        let text = std::fs::read_to_string(dir.join(file))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut p: Pair = serde_json::from_str(line)?;
            if p.good == p.bad {
                continue;
            }
            p.gating = split.get(&p.paradigm).is_some_and(|r| r == "gating");
            p.supplement = supp;
            out.push(p);
        }
    }
    Ok(out)
}

/// Scores of one paradigm or phenomenon.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Score {
    /// Pairs.
    pub n: u32,
    /// Pairs passed (ties count half).
    pub correct: f64,
    /// Pairs where either sentence has an unknown word.
    pub with_unknown: u32,
    /// Pairs where the bad sentence split into more than one fragment.
    pub bad_fragmented: u32,
    /// Pairs with neither.
    pub clean_n: u32,
    /// Clean pairs passed.
    pub clean_correct: f64,
}

impl Score {
    /// Accuracy in percent.
    pub fn acc(&self) -> f64 {
        100.0 * self.correct / f64::from(self.n.max(1))
    }
    fn add(&mut self, o: &Score) {
        self.n += o.n;
        self.correct += o.correct;
        self.with_unknown += o.with_unknown;
        self.bad_fragmented += o.bad_fragmented;
        self.clean_n += o.clean_n;
        self.clean_correct += o.clean_correct;
    }
}

/// BLiMP results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Results {
    /// Reported paradigms, by phenomenon.
    pub reported: BTreeMap<String, Score>,
    /// Gating paradigms, by phenomenon.
    pub gating: BTreeMap<String, Score>,
    /// All reported filtered-BLiMP pairs together.
    pub overall: Score,
    /// The supplement, by paradigm.
    pub supplement: BTreeMap<String, Score>,
}

fn sentence(parser: &mut Parser, s: &str) -> (Bits, usize, usize) {
    let mut ctx = Context::new();
    let mut bits = 0;
    let mut unknowns = 0;
    let mut frags = 0;
    for line in s.split('\n') {
        let a: LineAnalysis = parser.parse_line(&mut ctx, line);
        bits += a.bits;
        unknowns += a.unknowns();
        frags += a.fragments.len();
    }
    (bits, unknowns, frags)
}

/// Score pairs with a model.
pub fn evaluate(model: &Model, pairs: &[Pair], gating_only: bool) -> Results {
    let scored: Vec<(&Pair, Score)> = pairs
        .par_iter()
        .filter(|p| !gating_only || p.gating)
        .map_init(
            || Parser::new(model),
            |parser, p| {
                let (g, gu, _) = sentence(parser, &p.good);
                let (b, bu, bf) = sentence(parser, &p.bad);
                let correct = if g < b {
                    1.0
                } else if g == b {
                    0.5
                } else {
                    0.0
                };
                let unk = gu + bu > 0;
                let frag = bf > 1;
                let clean = !unk && !frag;
                (
                    p,
                    Score {
                        n: 1,
                        correct,
                        with_unknown: u32::from(unk),
                        bad_fragmented: u32::from(frag),
                        clean_n: u32::from(clean),
                        clean_correct: if clean { correct } else { 0.0 },
                    },
                )
            },
        )
        .collect();
    let mut r = Results::default();
    for (p, s) in scored {
        if p.supplement {
            r.supplement.entry(p.paradigm.clone()).or_default().add(&s);
        } else if p.gating {
            r.gating.entry(p.phenomenon.clone()).or_default().add(&s);
        } else {
            r.reported.entry(p.phenomenon.clone()).or_default().add(&s);
            r.overall.add(&s);
        }
    }
    r
}

/// The gate: does any phenomenon's gating accuracy drop by more than
/// `points` from `before` to `after`?
pub fn gate_fails(before: &Results, after: &Results, points: f64) -> Option<String> {
    for (ph, b) in &before.gating {
        if let Some(a) = after.gating.get(ph)
            && a.acc() < b.acc() - points
        {
            return Some(format!("{ph}: {:.1} -> {:.1}", b.acc(), a.acc()));
        }
    }
    None
}
