//! The world model: the cost of a meaning given the context.
//!
//! A meaning tree is generated top-down. For each slot (the root, or a role
//! of a parent node) it first chooses how the slot is filled:
//!
//! - **fresh**: a head (concept, entity, `Unknown`, `Missing`, `Query`),
//!   whether it carries a name (then the name is spelled), its features and
//!   its role children;
//! - **refer back**: an entity in the context, by recency rank (rank 0 is
//!   the speaker, 1 the addressee), then its features and children;
//! - **restate**: (at the root) an earlier fragment meaning or a global
//!   fact, uniformly among the candidates;
//! - **infer**: (at the root) a fact that follows by one inference step.
//!
//! The cheapest choice is taken for each node.

use std::sync::Arc;

use crate::context::Context;
use crate::cost::{
    self, Bits, Event, FACT_LOOKBACK, K_ABSENT, K_ANY, K_NONE, K_ROOT, MAX_FACT_CANDIDATES,
    REF_LOOKBACK, Table, featval_key, head_key, lg, meaning_key,
};
use crate::model::Scorer;
use crate::program::TOP;
use crate::term::{Deixis, Head, Node, Term};

/// How a slot was filled.
pub const MODE_FRESH: u64 = 0;
/// Referred back to a context entity.
pub const MODE_REF: u64 = 1;
/// Restated an earlier meaning or fact.
pub const MODE_RESTATE: u64 = 2;
/// Inferred from known facts.
pub const MODE_INFER: u64 = 3;

/// Rank used for a reference with nothing to point at in the context.
pub const RANK_EXOPHORIC: u64 = REF_LOOKBACK as u64 + 2;

/// An entity a meaning mentions, for updating the context.
#[derive(Clone, Debug, PartialEq)]
pub enum Mention {
    /// A new entity.
    New {
        /// Head key of its concept.
        concept: u64,
        /// Its name.
        name: Option<Arc<str>>,
    },
    /// An existing context entity.
    Old(u32),
}

/// The world-model cost of a meaning.
#[derive(Clone, Debug, Default)]
pub struct WorldOut {
    /// Total bits.
    pub bits: Bits,
    /// Table events.
    pub events: Vec<Event>,
    /// Bits outside tables (spelled names, uniform choices).
    pub fixed: Bits,
    /// Entities mentioned, in order.
    pub mentions: Vec<Mention>,
    /// How each node was filled (pre-order).
    pub modes: Vec<u64>,
}

impl WorldOut {
    fn push(&mut self, s: &Scorer, table: Table, ctx: [u64; 2], out: u64) {
        let e = Event {
            table,
            ctx,
            out,
            spell: 0,
        };
        self.bits += s.bits(&e);
        self.events.push(e);
    }

    fn fixed(&mut self, b: Bits) {
        self.bits += b;
        self.fixed += b;
    }
}

/// Cost a fragment meaning (a closed tree) in a context.
pub fn world_cost(s: &Scorer, ctx: &Context, meaning: &Term) -> Option<WorldOut> {
    let node = meaning.as_node()?;
    let mut full = WorldOut::default();
    code_node(s, ctx, node, K_ROOT, K_NONE, TOP, &mut full)?;

    // Restating an earlier meaning or a global fact.
    let key = meaning_key(meaning);
    let kb = s.model.fact_keys.len().min(MAX_FACT_CANDIDATES);
    let in_ctx = ctx.facts.iter().take(FACT_LOOKBACK).any(|f| f.key == key);
    if in_ctx || s.model.fact_keys.contains(&key) {
        let mut alt = WorldOut::default();
        alt.push(s, Table::Mode, [K_ROOT, 0], MODE_RESTATE);
        let candidates = ctx.facts.len().min(FACT_LOOKBACK) + kb;
        alt.fixed(lg(candidates as u64));
        alt.modes.push(MODE_RESTATE);
        collect_mentions(ctx, node, &mut alt.mentions);
        if alt.bits < full.bits {
            return Some(alt);
        }
    }
    Some(full)
}

fn collect_mentions(ctx: &Context, n: &Node, out: &mut Vec<Mention>) {
    match &n.head {
        Head::Unknown | Head::Missing | Head::Query => {}
        Head::Ref(d) => {
            let id = match d {
                Deixis::Speaker => Some(ctx.speaker),
                Deixis::Addressee => Some(ctx.addressee),
                Deixis::Other => ctx.recent().next().map(|e| e.id),
            };
            if let Some(id) = id {
                out.push(Mention::Old(id));
            }
        }
        h => out.push(Mention::New {
            concept: head_key(h),
            name: n.name.clone(),
        }),
    }
    for (_, c) in &n.roles {
        if let Some(cn) = c.as_node() {
            collect_mentions(ctx, cn, out);
        }
    }
}

/// Code one node in slot `(role, parent)`. `range` is the role's range
/// type, used to pick a compatible referent for pronouns.
fn code_node(
    s: &Scorer,
    ctx: &Context,
    n: &Node,
    role: u64,
    parent: u64,
    range: u32,
    out: &mut WorldOut,
) -> Option<()> {
    let p = &s.model.program;
    let concept: u64;
    match &n.head {
        Head::Ref(d) => {
            let (rank, ent) = match d {
                Deixis::Speaker => (0u64, ctx.entity(ctx.speaker)),
                Deixis::Addressee => (1u64, ctx.entity(ctx.addressee)),
                Deixis::Other => {
                    let mut best: Option<(Bits, u64, u32)> = None;
                    for (i, e) in ctx.recent().enumerate() {
                        let ok = range == TOP
                            || u32::try_from(e.concept).is_ok_and(|c| p.is_a(c, range));
                        if !ok {
                            continue;
                        }
                        let r = i as u64 + 2;
                        let b = s.ev(Table::RefRank, [role, 0], r, 0);
                        if best.is_none_or(|(bb, _, _)| b < bb) {
                            best = Some((b, r, e.id));
                        }
                    }
                    match best {
                        Some((_, r, id)) => (r, ctx.entity(id)),
                        None => (RANK_EXOPHORIC, None),
                    }
                }
            };
            out.push(s, Table::Mode, [role, 0], MODE_REF);
            out.push(s, Table::RefRank, [role, 0], rank);
            out.modes.push(MODE_REF);
            concept = ent.map_or(K_NONE, |e| e.concept);
            if let Some(e) = ent {
                out.mentions.push(Mention::Old(e.id));
            }
        }
        h => {
            let hk = head_key(h);
            concept = hk;
            // Fresh.
            let mut fresh = WorldOut::default();
            fresh.push(s, Table::Mode, [role, 0], MODE_FRESH);
            fresh.push(s, Table::Concept, [role, parent], hk);
            if matches!(h, Head::Concept(_) | Head::Entity(_)) {
                fresh.push(s, Table::HasName, [hk, 0], u64::from(n.name.is_some()));
                if let Some(name) = &n.name {
                    fresh.fixed(cost::spell(name.as_bytes()));
                }
            }
            // Refer back to an entity with the same concept (and name).
            let mut referent = None;
            if matches!(h, Head::Concept(_) | Head::Entity(_)) {
                let found = ctx
                    .recent()
                    .enumerate()
                    .find(|(_, e)| e.concept == hk && (n.name.is_none() || e.name == n.name));
                if let Some((i, e)) = found {
                    let mut r = WorldOut::default();
                    r.push(s, Table::Mode, [role, 0], MODE_REF);
                    r.push(s, Table::RefRank, [role, 0], i as u64 + 2);
                    if r.bits < fresh.bits {
                        referent = Some((r, e.id));
                    }
                } else if n.name.is_some() {
                    // A named speaker can be referred to by name.
                    for id in [ctx.speaker, ctx.addressee] {
                        if let Some(e) = ctx.entity(id)
                            && e.name == n.name
                        {
                            let mut r = WorldOut::default();
                            r.push(s, Table::Mode, [role, 0], MODE_REF);
                            let rank = if id == ctx.speaker { 0 } else { 1 };
                            r.push(s, Table::RefRank, [role, 0], rank);
                            referent = Some((r, id));
                            break;
                        }
                    }
                }
            }
            let (part, mention, mode) = match referent {
                Some((r, id)) => (r, Mention::Old(id), MODE_REF),
                None => (
                    fresh,
                    Mention::New {
                        concept: hk,
                        name: n.name.clone(),
                    },
                    MODE_FRESH,
                ),
            };
            out.bits += part.bits;
            out.fixed += part.fixed;
            out.events.extend(part.events);
            out.modes.push(mode);
            if !matches!(h, Head::Unknown | Head::Missing | Head::Query) {
                out.mentions.push(mention);
            }
        }
    }

    // Features in feature-ID order, then stop.
    let mut prev = K_ROOT;
    for (f, v) in &n.feats {
        out.push(s, Table::FeatNext, [prev, concept], featval_key(*f, *v));
        prev = u64::from(*f);
    }
    out.push(s, Table::FeatNext, [prev, concept], K_ABSENT);

    // Role children in role-ID order, then stop.
    let mut prev = K_ROOT;
    for (r, c) in &n.roles {
        out.push(s, Table::RoleNext, [prev, concept], u64::from(*r));
        let child = c.as_node()?;
        let range = p.roles.get(r).map_or(TOP, |x| x.range);
        code_node(s, ctx, child, u64::from(*r), concept, range, out)?;
        prev = u64::from(*r);
    }
    out.push(s, Table::RoleNext, [prev, concept], K_ABSENT);
    Some(())
}

/// A cheap estimate of the world cost of the nodes a leaf meaning creates
/// (their heads under the most general context), used to rank analyses
/// during the chart search.
pub fn leaf_estimate(s: &Scorer, sem: &Term) -> Bits {
    let mut b = 0;
    sem.for_each_node(&mut |n| {
        if !matches!(n.head, Head::Ref(_)) {
            b += s.ev(Table::Concept, [K_ANY, K_ANY], head_key(&n.head), 0);
        }
    });
    b
}
