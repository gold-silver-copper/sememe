//! Counting: aggregate the events of the best analyses, set the stored
//! counts (the M-step of Viterbi EM) and compute the exact total cost.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use sememe_engine::cost::{self, Bits, Event, Sizes, TAG_ENTRY, TAG_NAME, Table};
use sememe_engine::program::{Counts, CtxCounts, Program};

use crate::analyze::DocResult;

/// Event counts over a training set, with per-document event lists so a
/// document's contribution can be replaced after re-parsing it.
#[derive(Clone, Default)]
pub struct Stats {
    /// Distinct events.
    pub events: Vec<Event>,
    /// Index of each distinct event.
    pub ids: FxHashMap<Event, u32>,
    /// Count of each distinct event.
    pub counts: Vec<u64>,
    /// Bits outside tables.
    pub fixed: Bits,
    /// Per document: its event IDs, fixed bits, bytes, unknowns, words.
    pub docs: Vec<DocStats>,
    /// Printable symbol texts.
    pub syms: FxHashMap<u64, String>,
}

/// One document's contribution.
#[derive(Clone, Default)]
pub struct DocStats {
    /// Event IDs.
    pub events: Vec<u32>,
    /// Fixed bits.
    pub fixed: Bits,
    /// Original bytes.
    pub bytes: u64,
    /// Unknown-word leaves.
    pub unknowns: u32,
    /// Words.
    pub words: u32,
    /// Exact bits under the program that analyzed it.
    pub bits: Bits,
}

impl Stats {
    /// Analyze documents with a model and count, in chunks so memory stays
    /// bounded.
    pub fn build(model: &sememe_engine::model::Model, docs: &[&sememe_data::Doc]) -> Stats {
        let mut s = Stats::default();
        for chunk in docs.chunks(4096) {
            let results = crate::analyze::analyze(model, chunk, false);
            for r in results {
                let d = s.intern_doc(&model.program, r);
                s.add(&d);
                s.docs.push(d);
            }
        }
        s
    }

    /// Add one document's analysis at the end.
    pub fn push(&mut self, p: &Program, r: DocResult) {
        let d = self.intern_doc(p, r);
        self.add(&d);
        self.docs.push(d);
    }

    fn intern_doc(&mut self, p: &Program, r: DocResult) -> DocStats {
        for (k, sym) in &r.syms {
            if !self.syms.contains_key(k) {
                self.syms.insert(*k, sym.text(p));
            }
        }
        let mut ids = Vec::with_capacity(r.events.len());
        for e in r.events {
            let id = match self.ids.get(&e) {
                Some(id) => *id,
                None => {
                    let id = self.events.len() as u32;
                    self.events.push(e);
                    self.counts.push(0);
                    self.ids.insert(e, id);
                    id
                }
            };
            ids.push(id);
        }
        DocStats {
            events: ids,
            fixed: r.fixed,
            bytes: r.bytes,
            unknowns: r.unknowns,
            words: r.words,
            bits: r.bits,
        }
    }

    fn add(&mut self, d: &DocStats) {
        for id in &d.events {
            if let Some(c) = self.counts.get_mut(*id as usize) {
                *c += 1;
            }
        }
        self.fixed += d.fixed;
    }

    fn sub(&mut self, d: &DocStats) {
        for id in &d.events {
            if let Some(c) = self.counts.get_mut(*id as usize) {
                *c = c.saturating_sub(1);
            }
        }
        self.fixed = self.fixed.saturating_sub(d.fixed);
    }

    /// Replace documents' contributions with new analyses.
    pub fn replace(&mut self, p: &Program, doc_ids: &[usize], results: Vec<DocResult>) {
        for (&i, r) in doc_ids.iter().zip(results) {
            let new = self.intern_doc(p, r);
            if let Some(old) = self.docs.get(i).cloned() {
                self.sub(&old);
            }
            self.add(&new);
            if let Some(slot) = self.docs.get_mut(i) {
                *slot = new;
            }
        }
    }

    /// Total original bytes.
    pub fn bytes(&self) -> u64 {
        self.docs.iter().map(|d| d.bytes).sum()
    }

    /// Total unknown-word leaves and words.
    pub fn unknown_rate(&self) -> f64 {
        let u: u64 = self.docs.iter().map(|d| u64::from(d.unknowns)).sum();
        let w: u64 = self.docs.iter().map(|d| u64::from(d.words)).sum();
        u as f64 / w.max(1) as f64
    }
}

/// Is this a spelled-word outcome of the lexical table (never stored: an
/// unknown word always pays for its spelling)?
fn spelled_word(e: &Event) -> bool {
    if e.table != Table::Lex {
        return false;
    }
    let tag = e.out & 0xffff_ffff_0000_0000;
    tag != TAG_ENTRY && tag != TAG_NAME
}

/// The M-step: choose which outcomes are stored in which contexts, and
/// their quantized counts. An outcome is stored at the most specific
/// context of its back-off chain where it was used at least twice, or at
/// the last level if it was used at all; the rest escape to the next level.
/// Spelled words are never stored.
pub fn m_step(p: &Program, stats: &Stats) -> Counts {
    m_step_events(
        p,
        stats.events.iter().zip(stats.counts.iter().copied()),
        &stats.syms,
    )
}

/// The M-step over any event counts.
pub fn m_step_events<'a>(
    p: &Program,
    events: impl Iterator<Item = (&'a Event, u64)>,
    syms: &FxHashMap<u64, String>,
) -> Counts {
    // Pending (table, context, outcome) -> count, starting at level 0.
    let mut pending: FxHashMap<(Table, [u64; 2], u64), u64> = FxHashMap::default();
    let mut spelled: FxHashMap<(Table, [u64; 2]), u64> = FxHashMap::default();
    for (e, n) in events {
        if n == 0 {
            continue;
        }
        if spelled_word(e) {
            *spelled.entry((e.table, e.ctx)).or_default() += n;
            continue;
        }
        *pending.entry((e.table, e.ctx, e.out)).or_default() += n;
    }
    let mut stored: FxHashMap<(Table, [u64; 2]), CtxCounts> = FxHashMap::default();
    let mut escapes: FxHashMap<(Table, [u64; 2]), u64> = FxHashMap::default();
    for ((t, ctx), n) in spelled {
        *escapes.entry((t, ctx)).or_default() += n;
    }
    let mut next_cache: FxHashMap<(Table, [u64; 2]), Option<[u64; 2]>> = FxHashMap::default();
    while !pending.is_empty() {
        let mut next: FxHashMap<(Table, [u64; 2], u64), u64> = FxHashMap::default();
        let mut keys: Vec<_> = pending.into_iter().collect();
        keys.sort_unstable();
        for ((t, ctx, out), n) in keys {
            let nxt = *next_cache.entry((t, ctx)).or_insert_with(|| {
                let e = Event {
                    table: t,
                    ctx,
                    out,
                    spell: 0,
                };
                cost::chain(p, &e).get(1).copied()
            });
            let store = n >= 2 || nxt.is_none();
            if store {
                stored
                    .entry((t, ctx))
                    .or_default()
                    .outs
                    .insert(out, cost::level_of(n));
            } else {
                *escapes.entry((t, ctx)).or_default() += n;
                if let Some(c) = nxt {
                    *next.entry((t, c, out)).or_default() += n;
                }
            }
        }
        pending = next;
    }
    for ((t, ctx), n) in escapes {
        if let Some(cc) = stored.get_mut(&(t, ctx)) {
            cc.esc = cost::level_of(n);
        }
    }
    let mut counts = Counts::default();
    for ((t, ctx), cc) in stored {
        for k in [ctx[0], ctx[1]].into_iter().chain(cc.outs.keys().copied()) {
            if let Some(s) = syms.get(&k) {
                counts.syms.insert(k, s.clone());
            }
        }
        counts.tables.insert((t as u8, ctx), cc);
    }
    counts
}

/// Exact data bits of the stored analyses under a program's counts.
pub fn data_bits(p: &Program, stats: &Stats) -> Bits {
    let sizes = Sizes::of(p);
    let table: Bits = stats
        .events
        .par_iter()
        .zip(stats.counts.par_iter())
        .map(|(e, n)| {
            if *n == 0 {
                0
            } else {
                n * cost::event_bits(p, &sizes, e)
            }
        })
        .sum();
    table + stats.fixed
}

/// Bits per table (for reports).
pub fn bits_by_table(p: &Program, stats: &Stats) -> Vec<(Table, Bits)> {
    let sizes = Sizes::of(p);
    let mut by: FxHashMap<Table, Bits> = FxHashMap::default();
    for (e, n) in stats.events.iter().zip(stats.counts.iter()) {
        if *n > 0 {
            *by.entry(e.table).or_default() += n * cost::event_bits(p, &sizes, e);
        }
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.1));
    v
}

/// Event counts with some documents' analyses replaced: the base counts
/// plus a delta, as a flat list (for scoring a candidate change).
pub struct Overlay {
    /// Events and their counts.
    pub events: Vec<(Event, u64)>,
    /// Fixed bits.
    pub fixed: Bits,
}

impl Stats {
    /// The counts that would result from replacing documents `ids` with
    /// `results`, without changing `self`.
    pub fn overlay(
        &self,
        p: &Program,
        ids: &[usize],
        results: &[DocResult],
    ) -> (Overlay, FxHashMap<u64, String>) {
        let mut delta: FxHashMap<Event, i64> = FxHashMap::default();
        let mut fixed = self.fixed as i128;
        let mut syms: FxHashMap<u64, String> = FxHashMap::default();
        for (&i, r) in ids.iter().zip(results) {
            if let Some(d) = self.docs.get(i) {
                for id in &d.events {
                    if let Some(e) = self.events.get(*id as usize) {
                        *delta.entry(*e).or_default() -= 1;
                    }
                }
                fixed -= d.fixed as i128;
            }
            for e in &r.events {
                *delta.entry(*e).or_default() += 1;
            }
            fixed += r.fixed as i128;
            for (k, s) in &r.syms {
                if !self.syms.contains_key(k) && !syms.contains_key(k) {
                    syms.insert(*k, s.text(p));
                }
            }
        }
        let mut events: Vec<(Event, u64)> = Vec::with_capacity(self.events.len() + delta.len());
        for (e, n) in self.events.iter().zip(self.counts.iter()) {
            let d = delta.remove(e).unwrap_or(0);
            let v = (*n as i64 + d).max(0) as u64;
            if v > 0 {
                events.push((*e, v));
            }
        }
        for (e, d) in delta {
            if d > 0 {
                events.push((e, d as u64));
            }
        }
        (
            Overlay {
                events,
                fixed: fixed.max(0) as Bits,
            },
            syms,
        )
    }
}

/// Exact data bits of overlay counts under a program.
pub fn overlay_bits(p: &Program, o: &Overlay) -> Bits {
    let sizes = Sizes::of(p);
    let table: Bits = o
        .events
        .par_iter()
        .map(|(e, n)| n * cost::event_bits(p, &sizes, e))
        .sum();
    table + o.fixed
}
