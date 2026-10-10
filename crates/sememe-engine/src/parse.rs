//! Backward mode: text to the cheapest analysis, and its exact cost.
//!
//! For each line: tokenize, build leaf analyses for every word (lexical
//! entries, inflected stems, proper names, unknowns), run a beam-limited
//! CKY chart over spans within sentences, pick the cheapest split into
//! fragments, then rescore the chosen analysis exactly, top-down, in the
//! order a decoder would read it, updating the context as it goes.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::category::{Bindings, Cat, Dir, substitute, unify};
use crate::context::Context;
use crate::cost::{
    self, BEAM, Bits, Event, K_ABSENT, K_LEX, K_NEW, MAX_ARITY, MAX_FRAGMENT_WORDS, TAG_ENTRY,
    TAG_NAME, Table, cat_key, featset_key, label_key, meaning_key, sem_key, str_key,
};
use crate::model::{Model, Scorer};
use crate::program::{ROLE_OP, RuleKind};
use crate::term::{Head, Label, Node, Term};
use crate::text::{Case, GAP_BOUNDARY, GAP_END, GAP_INTERNAL, GAP_START, GAP_WHOLE, Line, TagKind};
use crate::world::{Mention, WorldOut, world_cost};

/// What a leaf of a derivation is.
#[derive(Clone, Debug, PartialEq)]
pub enum LeafKind {
    /// A lexical entry, possibly inflected.
    Entry {
        /// Entry ID.
        id: u32,
        /// Inflection ID.
        infl: Option<u32>,
    },
    /// A proper name.
    Name {
        /// Name-entry ID.
        id: u32,
    },
    /// An unknown word (spelled), possibly with a known inflection.
    Unknown {
        /// Inflection ID.
        infl: Option<u32>,
    },
}

/// A derivation tree.
#[derive(Clone, Debug)]
pub enum Deriv {
    /// A word.
    Leaf {
        /// Word index in the line.
        word: usize,
        /// What realizes it.
        kind: LeafKind,
        /// The stem that was looked up or spelled.
        stem: String,
        /// Category.
        cat: Cat,
        /// Meaning.
        sem: Term,
    },
    /// A unary rule.
    Unary {
        /// Rule ID.
        rule: u32,
        /// The child.
        child: Arc<Deriv>,
        /// Category.
        cat: Cat,
        /// Meaning.
        sem: Term,
    },
    /// A binary rule.
    Binary {
        /// Rule ID.
        rule: u32,
        /// Rule kind code.
        kind: u32,
        /// Left child.
        left: Arc<Deriv>,
        /// Right child.
        right: Arc<Deriv>,
        /// The category the rule introduces.
        newcat: Cat,
        /// How the meaning is split.
        label: Label,
        /// Category.
        cat: Cat,
        /// Meaning.
        sem: Term,
    },
    /// Coordination `X CONJ X`.
    Coord {
        /// Rule ID.
        rule: u32,
        /// Left conjunct.
        left: Arc<Deriv>,
        /// The conjunction word.
        conj: Arc<Deriv>,
        /// Right conjunct.
        right: Arc<Deriv>,
        /// Category.
        cat: Cat,
        /// Meaning.
        sem: Term,
    },
}

impl Deriv {
    /// The node's category.
    pub fn cat(&self) -> &Cat {
        match self {
            Deriv::Leaf { cat, .. }
            | Deriv::Unary { cat, .. }
            | Deriv::Binary { cat, .. }
            | Deriv::Coord { cat, .. } => cat,
        }
    }

    /// The node's meaning.
    pub fn sem(&self) -> &Term {
        match self {
            Deriv::Leaf { sem, .. }
            | Deriv::Unary { sem, .. }
            | Deriv::Binary { sem, .. }
            | Deriv::Coord { sem, .. } => sem,
        }
    }

    /// The leaves in order.
    pub fn leaves(&self) -> Vec<&Deriv> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a Deriv>) {
        match self {
            Deriv::Leaf { .. } => out.push(self),
            Deriv::Unary { child, .. } => child.collect_leaves(out),
            Deriv::Binary { left, right, .. } => {
                left.collect_leaves(out);
                right.collect_leaves(out);
            }
            Deriv::Coord {
                left, conj, right, ..
            } => {
                left.collect_leaves(out);
                conj.collect_leaves(out);
                right.collect_leaves(out);
            }
        }
    }
}

/// One fragment of an analyzed line.
#[derive(Clone, Debug)]
pub struct Fragment {
    /// First word.
    pub start: usize,
    /// One past the last word.
    pub end: usize,
    /// Derivation.
    pub deriv: Arc<Deriv>,
    /// The fragment's meaning (functor roots applied to `Missing`).
    pub meaning: Term,
    /// Language bits (derivation and root category).
    pub lang_bits: Bits,
    /// World-model bits.
    pub world_bits: Bits,
    /// How each meaning node was filled.
    pub modes: Vec<u64>,
}

/// The analysis of one line.
#[derive(Clone, Debug, Default)]
pub struct LineAnalysis {
    /// Exact bits of the line (Viterbi path found by the bounded search).
    pub bits: Bits,
    /// Table events, in decoding order.
    pub events: Vec<Event>,
    /// Bits outside tables.
    pub fixed: Bits,
    /// Fragments.
    pub fragments: Vec<Fragment>,
    /// The tokenized line.
    pub words: Vec<String>,
    /// Formatting bits (tag, case, gaps).
    pub format_bits: Bits,
    /// Printable sources of the hashed keys the events use.
    pub syms: Vec<(u64, Sym)>,
}

/// The source of a hashed key, for printing counts.
#[derive(Clone, Debug)]
pub enum Sym {
    /// A category.
    Cat(Cat),
    /// A lexical meaning.
    Term(Term),
    /// A string.
    Str(String),
    /// A set of meaning features.
    Feats(Vec<(u32, u16)>),
}

impl Sym {
    /// Printable text (the form `counts.tsv` uses).
    pub fn text(&self, p: &crate::program::Program) -> String {
        match self {
            Sym::Cat(c) => c.show(&p.cat_names()),
            Sym::Term(t) => t.show(p),
            Sym::Str(s) => s.clone(),
            Sym::Feats(fs) => fs
                .iter()
                .map(|(f, v)| {
                    use crate::term::TermNames;
                    format!("{}={}", p.feat_name(*f), p.val_name(*f, *v))
                })
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

impl LineAnalysis {
    /// Number of unknown-word leaves.
    pub fn unknowns(&self) -> usize {
        self.fragments
            .iter()
            .flat_map(|f| f.deriv.leaves())
            .filter(|l| {
                matches!(
                    l,
                    Deriv::Leaf {
                        kind: LeafKind::Unknown { .. },
                        ..
                    }
                )
            })
            .count()
    }
}

/// Chosen fragments: `(start, end, root item)`.
type Segmentation = Vec<(usize, usize, Item)>;

/// Normal-form flags (Eisner 1996).
const NF_NONE: u8 = 0;
const NF_FCOMP: u8 = 1;
const NF_BCOMP: u8 = 2;

#[derive(Clone, Debug)]
struct Item {
    cat: Cat,
    sem: Term,
    score: Bits,
    world_est: Bits,
    deriv: Arc<Deriv>,
    nf: u8,
    /// A coordination half `CONJ X` waiting for its left conjunct.
    conj: Option<(u32, Arc<Deriv>, Arc<Deriv>)>,
    unary: bool,
    /// What the item does with its first argument, if it is a functor.
    label: Label,
    /// `meaning_key` of `sem`.
    mkey: u64,
    /// `cat_key` of `cat`.
    ckey: u64,
}

/// Feature-free keys of an item's category, its result and its argument.
fn bare_keys(c: &Cat) -> (u64, u64, u64) {
    let b = cat_key(&c.bare());
    match c.as_slash() {
        Some(s) => (b, cat_key(&s.res.bare()), cat_key(&s.arg.bare())),
        None => (b, 0, 0),
    }
}

/// The items of one chart cell, indexed for combination.
#[derive(Default)]
struct Cell {
    items: Vec<Item>,
    /// Bare category key -> item indices.
    by_cat: FxHashMap<u64, Vec<usize>>,
    /// Bare result key (functors only) -> item indices.
    by_res: FxHashMap<u64, Vec<usize>>,
    /// Bare argument key of each item (0 for atoms).
    arg: Vec<u64>,
}

impl Cell {
    fn new(items: Vec<Item>) -> Cell {
        let mut c = Cell::default();
        for (i, it) in items.iter().enumerate() {
            let (b, r, a) = bare_keys(&it.cat);
            c.by_cat.entry(b).or_default().push(i);
            if r != 0 {
                c.by_res.entry(r).or_default().push(i);
            }
            c.arg.push(a);
        }
        c.items = items;
        c
    }
}

/// The parser.
pub struct Parser<'m> {
    /// Event costs.
    pub scorer: Scorer<'m>,
    rule_cache: std::cell::RefCell<FxHashMap<(u64, u32, u64, Label), Bits>>,
    #[allow(clippy::type_complexity)]
    sem_memo: std::cell::RefCell<FxHashMap<(u64, u64, u32), Option<(Term, u64, Label)>>>,
    /// How many fragment candidates per span get an exact world score.
    pub roots_per_span: usize,
}

fn chain_cost(
    s: &Scorer,
    parent: &Cat,
    rule: u32,
    kind: u32,
    newcat: Option<&Cat>,
    label: Option<Label>,
) -> Bits {
    let pk = cat_key(parent);
    let mut b = s.ev(Table::Expand, [pk, 0], u64::from(rule), 0);
    if let Some(nc) = newcat {
        b += s.ev(
            Table::NewCat,
            [u64::from(rule), pk],
            cat_key(nc),
            cost::cat_prior(&s.model.program, nc),
        );
    }
    if let Some(l) = label {
        b += s.ev(Table::Split, [u64::from(kind), pk], label_key(l), 0);
    }
    b
}

fn merge(a: &Cat, b: &Cat) -> Cat {
    let mut bind = Bindings::default();
    if unify(a, 0, b, 1, &mut bind) {
        let x = substitute(a, 0, &bind);
        let y = substitute(b, 1, &bind);
        union_feats(&x, &y)
    } else {
        a.clone()
    }
}

fn union_feats(a: &Cat, b: &Cat) -> Cat {
    if !b.has_feats() {
        return a.clone();
    }
    match (a, b) {
        (Cat::Atom(x), Cat::Atom(y)) => {
            let mut feats = x.feats.clone();
            for f in &y.feats {
                if !feats.iter().any(|(d, _)| *d == f.0) {
                    feats.push(*f);
                }
            }
            feats.sort();
            Cat::Atom(crate::category::Atom { id: x.id, feats })
        }
        (Cat::Slash(x), Cat::Slash(y)) => Cat::slash(
            union_feats(&x.res, &y.res),
            x.dir,
            union_feats(&x.arg, &y.arg),
        ),
        _ => a.clone(),
    }
}

/// Try a binary rule on two items at the category level: `(category, new
/// category, label, normal-form flag)`. The meaning is built later, only
/// for candidates that survive the beam.
fn combine_cat(kind: &RuleKind, l: &Item, r: &Item) -> Option<(Cat, Cat, Label, u8)> {
    match kind {
        RuleKind::FApp => {
            if l.nf == NF_FCOMP {
                return None;
            }
            let ls = l.cat.as_slash().filter(|s| s.dir == Dir::Fwd)?;
            let mut b = Bindings::default();
            if !unify(&ls.arg, 0, &r.cat, 1, &mut b) {
                return None;
            }
            let cat = substitute(&ls.res, 0, &b);
            let newcat = union_feats(&substitute(&r.cat, 1, &b), &substitute(&ls.arg, 0, &b));
            Some((cat, newcat, l.label, NF_NONE))
        }
        RuleKind::BApp => {
            if r.nf == NF_BCOMP {
                return None;
            }
            let rs = r.cat.as_slash().filter(|s| s.dir == Dir::Bwd)?;
            let mut b = Bindings::default();
            if !unify(&rs.arg, 0, &l.cat, 1, &mut b) {
                return None;
            }
            let cat = substitute(&rs.res, 0, &b);
            let newcat = union_feats(&substitute(&l.cat, 1, &b), &substitute(&rs.arg, 0, &b));
            Some((cat, newcat, r.label, NF_NONE))
        }
        RuleKind::FComp | RuleKind::FXComp => {
            if l.nf == NF_FCOMP {
                return None;
            }
            let ls = l.cat.as_slash().filter(|s| s.dir == Dir::Fwd)?;
            let want = if *kind == RuleKind::FComp {
                Dir::Fwd
            } else {
                Dir::Bwd
            };
            let rs = r.cat.as_slash().filter(|s| s.dir == want)?;
            let mut b = Bindings::default();
            if !unify(&ls.arg, 0, &rs.res, 1, &mut b) {
                return None;
            }
            let cat = Cat::slash(substitute(&ls.res, 0, &b), want, substitute(&rs.arg, 1, &b));
            if cat.arity() > MAX_ARITY {
                return None;
            }
            let newcat = union_feats(&substitute(&rs.res, 1, &b), &substitute(&ls.arg, 0, &b));
            Some((cat, newcat, l.label, NF_FCOMP))
        }
        RuleKind::BComp | RuleKind::BXComp => {
            if r.nf == NF_BCOMP {
                return None;
            }
            let rs = r.cat.as_slash().filter(|s| s.dir == Dir::Bwd)?;
            let want = if *kind == RuleKind::BComp {
                Dir::Bwd
            } else {
                Dir::Fwd
            };
            let ls = l.cat.as_slash().filter(|s| s.dir == want)?;
            let mut b = Bindings::default();
            if !unify(&rs.arg, 0, &ls.res, 1, &mut b) {
                return None;
            }
            let cat = Cat::slash(substitute(&rs.res, 0, &b), want, substitute(&ls.arg, 1, &b));
            if cat.arity() > MAX_ARITY {
                return None;
            }
            let newcat = union_feats(&substitute(&ls.res, 1, &b), &substitute(&rs.arg, 0, &b));
            Some((cat, newcat, r.label, NF_BCOMP))
        }
        _ => None,
    }
}

/// The meaning of a binary combination.
fn combine_sem(kind: &RuleKind, l: &Item, r: &Item) -> Option<Term> {
    match kind {
        RuleKind::FApp => l.sem.apply(&r.sem),
        RuleKind::BApp => r.sem.apply(&l.sem),
        RuleKind::FComp | RuleKind::FXComp => Term::compose(&l.sem, &r.sem),
        RuleKind::BComp | RuleKind::BXComp => Term::compose(&r.sem, &l.sem),
        _ => None,
    }
}

/// A binary combination scored but not yet built.
struct Cand<'a> {
    score: Bits,
    ckey: u64,
    l: &'a Item,
    r: &'a Item,
    rule: u32,
    kind: &'a RuleKind,
    cat: Cat,
    newcat: Cat,
    label: Label,
    nf: u8,
}

fn featset_of(sem: &Term) -> Vec<(u32, u16)> {
    sem.leaf_head().map(|n| n.feats.clone()).unwrap_or_default()
}

impl<'m> Parser<'m> {
    /// A parser for a model.
    pub fn new(model: &'m Model) -> Parser<'m> {
        Parser {
            scorer: Scorer::new(model),
            rule_cache: std::cell::RefCell::new(FxHashMap::default()),
            sem_memo: std::cell::RefCell::new(FxHashMap::default()),
            roots_per_span: 3,
        }
    }

    /// Bits of a binary step: expansion, new category, split label.
    #[allow(clippy::too_many_arguments)]
    fn rule_cost_keys(
        &self,
        cat: &Cat,
        cat_k: u64,
        rule: u32,
        kind: u32,
        newcat: &Cat,
        newcat_k: u64,
        label: Label,
    ) -> Bits {
        let key = (cat_k, rule, newcat_k, label);
        if let Some(b) = self.rule_cache.borrow().get(&key) {
            return *b;
        }
        let b = chain_cost(&self.scorer, cat, rule, kind, Some(newcat), Some(label));
        let mut c = self.rule_cache.borrow_mut();
        if c.len() > 1_000_000 {
            c.clear();
        }
        c.insert(key, b);
        b
    }

    fn model(&self) -> &'m Model {
        self.scorer.model
    }

    /// Bits of a leaf: expansion to a leaf, inflection, lexical choice.
    fn leaf_bits(&self, cat: &Cat, sem: &Term, kind: &LeafKind, stem: &str) -> Bits {
        let s = &self.scorer;
        let ck = cat_key(cat);
        let mut b = s.ev(Table::Expand, [ck, 0], K_LEX, 0);
        let feats = featset_of(sem);
        let infl_out = match kind {
            LeafKind::Entry { infl: Some(i), .. } | LeafKind::Unknown { infl: Some(i) } => {
                u64::from(*i)
            }
            _ => K_ABSENT,
        };
        b += s.ev(Table::Infl, [ck, featset_key(&feats)], infl_out, 0);
        let base = self.base_sem(sem, kind);
        let (out, spell) = match kind {
            LeafKind::Entry { id, .. } => {
                (TAG_ENTRY | u64::from(*id), cost::spell(stem.as_bytes()))
            }
            LeafKind::Name { id } => (TAG_NAME | u64::from(*id), 0),
            LeafKind::Unknown { .. } => (str_key(stem), cost::spell(stem.as_bytes())),
        };
        b += s.ev(Table::Lex, [ck, sem_key(&base)], out, spell);
        b
    }

    fn base_sem(&self, sem: &Term, kind: &LeafKind) -> Term {
        let infl = match kind {
            LeafKind::Entry { infl: Some(i), .. } | LeafKind::Unknown { infl: Some(i) } => Some(*i),
            _ => None,
        };
        match infl.and_then(|i| self.model().program.infls.get(&i)) {
            Some(inf) => sem.without_head_feats(&inf.feats),
            None => sem.clone(),
        }
    }

    fn leaves(&self, word: usize, form: &str, written_case: Case) -> Vec<Item> {
        let m = self.model();
        let mut out = Vec::new();
        let mut push = |cat: Cat, sem: Term, kind: LeafKind, stem: &str| {
            let score = self.leaf_bits(&cat, &sem, &kind, stem);
            let world_est = crate::world::leaf_estimate(&self.scorer, &sem);
            let label = sem.classify();
            let mkey = meaning_key(&sem);
            let ckey = cat_key(&cat);
            out.push(Item {
                mkey,
                ckey,
                deriv: Arc::new(Deriv::Leaf {
                    word,
                    kind,
                    stem: stem.to_string(),
                    cat: cat.clone(),
                    sem: sem.clone(),
                }),
                cat,
                sem,
                score: score + world_est,
                world_est,
                nf: NF_NONE,
                conj: None,
                unary: false,
                label,
            });
        };
        if let Some(es) = m.by_form.get(form) {
            for e in es {
                push(
                    e.cat.clone(),
                    e.sem.clone(),
                    LeafKind::Entry {
                        id: e.id,
                        infl: None,
                    },
                    &e.form,
                );
            }
        }
        for (iid, inf) in &m.infls {
            let Some(base) = form.strip_suffix(inf.add.as_str()) else {
                continue;
            };
            if base.is_empty() {
                continue;
            }
            let stem = format!("{base}{}", inf.strip);
            if let Some(es) = m.by_form.get(&stem) {
                for e in es {
                    if !e.cat.same_shape(&inf.cat) {
                        continue;
                    }
                    if let Some(sem) = e.sem.with_head_feats(&inf.feats) {
                        push(
                            e.cat.clone(),
                            sem,
                            LeafKind::Entry {
                                id: e.id,
                                infl: Some(*iid),
                            },
                            &stem,
                        );
                    }
                }
            }
            for u in &m.unknown_cats {
                if u.cat.same_shape(&inf.cat)
                    && let Some(sem) = u.sem.with_head_feats(&inf.feats)
                {
                    push(
                        u.cat.clone(),
                        sem,
                        LeafKind::Unknown { infl: Some(*iid) },
                        &stem,
                    );
                }
            }
        }
        if written_case != Case::Lower {
            for (nid, cat, concept) in &m.names {
                let mut node = Node::new(Head::Concept(*concept));
                node.name = Some(Arc::from(form));
                push(
                    cat.clone(),
                    Term::node(node),
                    LeafKind::Name { id: *nid },
                    form,
                );
            }
        }
        for u in &m.unknown_cats {
            push(
                u.cat.clone(),
                u.sem.clone(),
                LeafKind::Unknown { infl: None },
                form,
            );
        }
        out
    }

    fn unary(&self, items: &[Item]) -> Vec<Item> {
        let m = self.model();
        let mut out = Vec::new();
        for it in items {
            if it.unary || it.conj.is_some() {
                continue;
            }
            for (rid, rule) in &m.unary {
                let (from, newcat, sem) = match rule {
                    RuleKind::FTr(x, t) => (
                        x,
                        Cat::slash(
                            t.clone(),
                            Dir::Fwd,
                            Cat::slash(t.clone(), Dir::Bwd, it.cat.clone()),
                        ),
                        Term::raise(&it.sem),
                    ),
                    RuleKind::BTr(x, t) => (
                        x,
                        Cat::slash(
                            t.clone(),
                            Dir::Bwd,
                            Cat::slash(t.clone(), Dir::Fwd, it.cat.clone()),
                        ),
                        Term::raise(&it.sem),
                    ),
                    RuleKind::Tc(x, y) => (x, y.clone(), it.sem.clone()),
                    _ => continue,
                };
                let mut b = Bindings::default();
                if !unify(from, 0, &it.cat, 1, &mut b) {
                    continue;
                }
                let add = chain_cost(&self.scorer, &newcat, *rid, 0, None, None);
                let label = sem.classify();
                let mkey = meaning_key(&sem);
                let ckey = cat_key(&newcat);
                out.push(Item {
                    mkey,
                    ckey,
                    deriv: Arc::new(Deriv::Unary {
                        rule: *rid,
                        child: it.deriv.clone(),
                        cat: newcat.clone(),
                        sem: sem.clone(),
                    }),
                    cat: newcat,
                    sem,
                    score: it.score + add,
                    world_est: it.world_est,
                    nf: NF_NONE,
                    conj: None,
                    unary: true,
                    label,
                });
            }
        }
        out
    }

    /// Every combination of an item of `lc` (left) with an item of `rc`
    /// (right): coordination items go to `done`, binary-rule results to
    /// `cands` (meanings not built yet).
    fn combine_cells<'a>(
        &'a self,
        lc: &'a Cell,
        rc: &'a Cell,
        done: &mut Vec<Item>,
        cands: &mut Vec<Cand<'a>>,
    ) {
        let m = self.model();
        // Coordination.
        if !m.coord.is_empty() {
            for l in &lc.items {
                for r in &rc.items {
                    self.coord_pair(l, r, done);
                }
            }
        }
        let try_rule =
            |l: &'a Item, r: &'a Item, rid: u32, kind: &'a RuleKind, cands: &mut Vec<Cand<'a>>| {
                if l.conj.is_some() || r.conj.is_some() {
                    return;
                }
                let Some((cat, newcat, label, nf)) = combine_cat(kind, l, r) else {
                    return;
                };
                let cat_k = cat_key(&cat);
                let newcat_k = cat_key(&newcat);
                let add =
                    self.rule_cost_keys(&cat, cat_k, rid, kind.code(), &newcat, newcat_k, label);
                cands.push(Cand {
                    score: l.score + r.score + add,
                    ckey: cat_k,
                    l,
                    r,
                    rule: rid,
                    kind,
                    cat,
                    newcat,
                    label,
                    nf,
                });
            };
        for (rid, kind) in &m.binary {
            match kind {
                RuleKind::FApp | RuleKind::FComp | RuleKind::FXComp => {
                    for (li, l) in lc.items.iter().enumerate() {
                        let Some(ls) = l.cat.as_slash() else { continue };
                        if ls.dir != Dir::Fwd {
                            continue;
                        }
                        let want = lc.arg.get(li).copied().unwrap_or(0);
                        let pool = if *kind == RuleKind::FApp {
                            rc.by_cat.get(&want)
                        } else {
                            rc.by_res.get(&want)
                        };
                        for &ri in pool.map(Vec::as_slice).unwrap_or(&[]) {
                            if let Some(r) = rc.items.get(ri) {
                                try_rule(l, r, *rid, kind, cands);
                            }
                        }
                    }
                }
                RuleKind::BApp | RuleKind::BComp | RuleKind::BXComp => {
                    for (ri, r) in rc.items.iter().enumerate() {
                        let Some(rs) = r.cat.as_slash() else { continue };
                        if rs.dir != Dir::Bwd {
                            continue;
                        }
                        let want = rc.arg.get(ri).copied().unwrap_or(0);
                        let pool = if *kind == RuleKind::BApp {
                            lc.by_cat.get(&want)
                        } else {
                            lc.by_res.get(&want)
                        };
                        for &li in pool.map(Vec::as_slice).unwrap_or(&[]) {
                            if let Some(l) = lc.items.get(li) {
                                try_rule(l, r, *rid, kind, cands);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn coord_pair(&self, l: &Item, r: &Item, out: &mut Vec<Item>) {
        let m = self.model();
        // Finish a coordination: X (CONJ X).
        if let Some((rid, conj, right)) = &r.conj {
            if l.conj.is_none() && l.ckey == r.ckey && l.cat == r.cat {
                let k = l.cat.arity();
                if let Some(sem) = Term::coordinate(conj.sem(), &l.sem, right.sem(), k, ROLE_OP) {
                    let add = chain_cost(&self.scorer, &l.cat, *rid, 0, None, None);
                    let label = sem.classify();
                    let mkey = meaning_key(&sem);
                    out.push(Item {
                        mkey,
                        ckey: l.ckey,
                        deriv: Arc::new(Deriv::Coord {
                            rule: *rid,
                            left: l.deriv.clone(),
                            conj: conj.clone(),
                            right: right.clone(),
                            cat: l.cat.clone(),
                            sem: sem.clone(),
                        }),
                        cat: l.cat.clone(),
                        sem,
                        score: l.score + r.score + add,
                        world_est: l.world_est + r.world_est,
                        nf: NF_NONE,
                        conj: None,
                        unary: false,
                        label,
                    });
                }
            }
            return;
        }
        if l.conj.is_some() {
            return;
        }
        // Start a coordination: CONJ X.
        for (rid, ccat) in &m.coord {
            if l.cat == *ccat && matches!(*l.deriv, Deriv::Leaf { .. }) {
                out.push(Item {
                    deriv: r.deriv.clone(),
                    cat: r.cat.clone(),
                    sem: r.sem.clone(),
                    score: l.score + r.score,
                    world_est: l.world_est + r.world_est,
                    nf: NF_NONE,
                    conj: Some((*rid, l.deriv.clone(), r.deriv.clone())),
                    unary: true,
                    label: r.label,
                    mkey: r.mkey,
                    ckey: r.ckey,
                });
            }
        }
    }

    /// Build the best candidates' meanings until the beam is full. Meaning
    /// construction is memoized by the inputs' meaning keys, so the many
    /// derivations that yield the same meaning are recognized unbuilt.
    fn materialize(&self, mut cands: Vec<Cand>, mut done: Vec<Item>) -> Vec<Item> {
        // Only the best few can enter the beam: select them in linear time.
        let keep = 4 * BEAM;
        if cands.len() > keep {
            cands.select_nth_unstable_by_key(keep, |a| a.score);
            cands.truncate(keep);
        }
        cands.sort_by_key(|a| a.score);
        done.sort_by_key(|a| a.score);
        let mut seen: FxHashMap<(u64, u64, bool), ()> = FxHashMap::default();
        let mut out: Vec<Item> = Vec::new();
        let best = match (cands.first(), done.first()) {
            (Some(c), Some(d)) => c.score.min(d.score),
            (Some(c), None) => c.score,
            (None, Some(d)) => d.score,
            (None, None) => 0,
        };
        let limit = best.saturating_add(cost::BEAM_DELTA);
        let mut di = done.into_iter().peekable();
        let mut ci = cands.into_iter().peekable();
        // Building meanings is the expensive step; derivations that differ
        // only spuriously build the same meaning, so stop after a bounded
        // number of attempts even if the beam is not full.
        let mut attempts = 0;
        let mut di = di.take_while(|d| d.score <= limit).peekable();
        let mut ci = ci.take_while(|c| c.score <= limit).peekable();
        while out.len() < BEAM && attempts < 3 * BEAM {
            let take_done = match (di.peek(), ci.peek()) {
                (Some(d), Some(c)) => d.score <= c.score,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            let it = if take_done {
                match di.next() {
                    Some(it) => it,
                    None => break,
                }
            } else {
                let Some(c) = ci.next() else { break };
                attempts += 1;
                let memo_key = (c.l.mkey, c.r.mkey, c.kind.code());
                let cached = self.sem_memo.borrow().get(&memo_key).cloned();
                let built = match cached {
                    Some(b) => b,
                    None => {
                        let b = combine_sem(c.kind, c.l, c.r).map(|t| {
                            let k = meaning_key(&t);
                            let l = t.classify();
                            (t, k, l)
                        });
                        let mut m = self.sem_memo.borrow_mut();
                        if m.len() > 200_000 {
                            m.clear();
                        }
                        m.insert(memo_key, b.clone());
                        b
                    }
                };
                let Some((sem, mkey, label)) = built else {
                    continue;
                };
                if seen.contains_key(&(c.ckey, mkey, false)) {
                    continue;
                }
                Item {
                    mkey,
                    ckey: c.ckey,
                    deriv: Arc::new(Deriv::Binary {
                        rule: c.rule,
                        kind: c.kind.code(),
                        left: c.l.deriv.clone(),
                        right: c.r.deriv.clone(),
                        newcat: c.newcat,
                        label: c.label,
                        cat: c.cat.clone(),
                        sem: sem.clone(),
                    }),
                    cat: c.cat,
                    sem,
                    score: c.score,
                    world_est: c.l.world_est + c.r.world_est,
                    nf: c.nf,
                    conj: None,
                    unary: false,
                    label,
                }
            };
            let key = (it.ckey, it.mkey, it.conj.is_some());
            if seen.insert(key, ()).is_none() {
                out.push(it);
            }
        }
        out
    }

    fn beam(mut items: Vec<Item>) -> Vec<Item> {
        items.sort_by_key(|a| a.score);
        let mut seen: FxHashMap<(u64, u64, bool), ()> = FxHashMap::default();
        let mut out = Vec::new();
        for it in items {
            if out.len() >= BEAM {
                break;
            }
            let key = (it.ckey, it.mkey, it.conj.is_some());
            if seen.insert(key, ()).is_none() {
                out.push(it);
            }
        }
        out
    }

    /// Analyze one line in a context, updating the context.
    pub fn parse_line(&mut self, ctx: &mut Context, raw: &str) -> LineAnalysis {
        let line = Line::parse(raw);
        let mut a = LineAnalysis {
            words: line.words.iter().map(|w| w.form.clone()).collect(),
            ..LineAnalysis::default()
        };
        let mut fmt = WorldOut::default();
        // Speaker tag.
        let prev = ctx.prev_tag as u64;
        push_ev(
            &self.scorer,
            &mut fmt,
            Table::Tag,
            [prev, 0],
            line.tag as u64,
            0,
        );
        let rank = ctx.turn(line.tag, line.speaker.as_deref());
        if matches!(line.tag, TagKind::Childes | TagKind::Swb) {
            match rank {
                Some(r) => push_ev(
                    &self.scorer,
                    &mut fmt,
                    Table::SpeakerRank,
                    [0, 0],
                    r as u64,
                    0,
                ),
                None => {
                    push_ev(&self.scorer, &mut fmt, Table::SpeakerRank, [0, 0], K_NEW, 0);
                    let name = line.speaker.clone().unwrap_or_default();
                    push_ev(
                        &self.scorer,
                        &mut fmt,
                        Table::SpeakerName,
                        [0, 0],
                        str_key(&name),
                        cost::spell(name.as_bytes()),
                    );
                    a.syms.push((str_key(&name), Sym::Str(name.clone())));
                }
            }
        }
        let n = line.words.len();
        push_ev(
            &self.scorer,
            &mut fmt,
            Table::LineEmpty,
            [line.tag as u64, 0],
            u64::from(n == 0),
            0,
        );
        if n == 0 {
            let g = line.gaps.first().cloned().unwrap_or_default();
            push_ev(
                &self.scorer,
                &mut fmt,
                Table::Gap,
                [GAP_WHOLE, 0],
                str_key(&g),
                cost::spell(g.as_bytes()),
            );
            a.syms.push((str_key(&g), Sym::Str(g.clone())));
            a.format_bits = fmt.bits;
            a.bits = fmt.bits;
            a.fixed = fmt.fixed;
            a.events = fmt.events;
            return a;
        }
        let segs = self.segment(ctx, &line);
        // Exact rescoring in decoding order.
        let mut out = WorldOut::default();
        out.bits += fmt.bits;
        out.fixed += fmt.fixed;
        out.events.extend(fmt.events);
        let mut ends = Vec::new();
        for (k, (start, end, item)) in segs.into_iter().enumerate() {
            if k > 0 {
                push_ev(&self.scorer, &mut out, Table::FragMore, [0, 0], 1, 0);
            }
            let before = out.bits;
            let rc = item.cat.clone();
            push_ev(
                &self.scorer,
                &mut out,
                Table::RootCat,
                [0, 0],
                cat_key(&rc),
                cost::cat_prior(&self.model().program, &rc),
            );
            a.syms.push((cat_key(&rc), Sym::Cat(rc.clone())));
            self.emit(&item.deriv, &rc, &mut out, &mut a.syms);
            let lang = out.bits - before;
            let meaning =
                root_meaning(&item.cat, &item.sem).unwrap_or_else(|| Term::head(Head::Missing));
            let w = world_cost(&self.scorer, ctx, &meaning).unwrap_or_default();
            out.bits += w.bits;
            out.fixed += w.fixed;
            out.events.extend(w.events.iter().copied());
            apply_mentions(ctx, &w.mentions);
            ctx.remember(meaning_key(&meaning), meaning.clone());
            a.fragments.push(Fragment {
                start,
                end,
                deriv: item.deriv.clone(),
                meaning,
                lang_bits: lang,
                world_bits: w.bits,
                modes: w.modes,
            });
            ends.push(end);
        }
        push_ev(&self.scorer, &mut out, Table::FragMore, [0, 0], 0, 0);
        // Formatting: case and gaps.
        let fmt_before = out.bits;
        for (i, w) in line.words.iter().enumerate() {
            push_ev(
                &self.scorer,
                &mut out,
                Table::Case,
                [line.case_class(i), 0],
                w.case as u64,
                0,
            );
            if w.case == Case::Mixed {
                out.bits += u64::from(w.mixed_letters) * cost::ONE_BIT;
                out.fixed += u64::from(w.mixed_letters) * cost::ONE_BIT;
            }
        }
        for (i, g) in line.gaps.iter().enumerate() {
            let class = if i == 0 {
                GAP_START
            } else if i == n {
                GAP_END
            } else if ends.contains(&i) {
                GAP_BOUNDARY
            } else {
                GAP_INTERNAL
            };
            push_ev(
                &self.scorer,
                &mut out,
                Table::Gap,
                [class, 0],
                str_key(g),
                cost::spell(g.as_bytes()),
            );
            a.syms.push((str_key(g), Sym::Str(g.clone())));
        }
        a.format_bits = fmt.bits + (out.bits - fmt_before);
        a.bits = out.bits;
        a.fixed = out.fixed;
        a.events = out.events;
        a
    }

    /// Choose fragments: chart parse within sentences, then dynamic
    /// programming over fragment boundaries.
    fn segment(&mut self, ctx: &Context, line: &Line) -> Vec<(usize, usize, Item)> {
        let n = line.words.len();
        let bounds = line.boundaries();
        let stops = line.hard_stops();
        // Sentences: maximal runs without a hard stop inside.
        let mut sentences = Vec::new();
        let mut s0 = 0;
        for i in 1..n {
            if stops.get(i).copied().unwrap_or(false) {
                sentences.push((s0, i));
                s0 = i;
            }
        }
        sentences.push((s0, n));

        let cap = MAX_FRAGMENT_WORDS;
        let mut world_memo: FxHashMap<u64, Bits> = FxHashMap::default();
        // best[j] = (cost, fragments) for words 0..j.
        let mut best: Vec<Option<(Bits, Segmentation)>> = vec![None; n + 1];
        if let Some(b) = best.get_mut(0) {
            *b = Some((0, Vec::new()));
        }
        for (a0, b0) in sentences {
            let len = b0 - a0;
            let long = len > cap;
            let allowed =
                |p: usize| p == a0 || p == b0 || long || bounds.get(p).copied().unwrap_or(false);
            // Chart: cells[i][l] for span [a0+i, a0+i+l).
            let width = len.min(cap);
            let mut cells: Vec<Vec<Cell>> = (0..len)
                .map(|_| (0..=width).map(|_| Cell::default()).collect())
                .collect();
            for i in 0..len {
                let w = line.words.get(a0 + i);
                let (form, case) = w.map_or(("", Case::Lower), |w| (w.form.as_str(), w.case));
                let mut leaves = self.leaves(a0 + i, form, case);
                let un = self.unary(&leaves);
                leaves.extend(un);
                if let Some(c) = cells.get_mut(i).and_then(|r| r.get_mut(1)) {
                    *c = Cell::new(Self::beam(leaves));
                }
            }
            for l in 2..=width {
                for i in 0..=(len - l) {
                    let mut done = Vec::new();
                    let mut cands = Vec::new();
                    for k in 1..l {
                        if let (Some(lc), Some(rc)) = (
                            cells.get(i).and_then(|r| r.get(k)),
                            cells.get(i + k).and_then(|r| r.get(l - k)),
                        ) {
                            self.combine_cells(lc, rc, &mut done, &mut cands);
                        }
                    }
                    let mut items = self.materialize(cands, done);
                    let un = self.unary(&items);
                    if !un.is_empty() {
                        items.extend(un);
                        items = Self::beam(items);
                    }
                    if let Some(c) = cells.get_mut(i).and_then(|r| r.get_mut(l)) {
                        *c = Cell::new(items);
                    }
                }
            }
            // Fragment candidates and DP within this sentence.
            for j in (a0 + 1)..=b0 {
                if !allowed(j) {
                    continue;
                }
                let mut cands: Vec<(Bits, usize, Item)> = Vec::new();
                for i in (a0..j).rev() {
                    if j - i > cap {
                        break;
                    }
                    if !allowed(i) {
                        continue;
                    }
                    let Some(Some((prev_cost, _))) = best.get(i) else {
                        continue;
                    };
                    let Some(items) = cells
                        .get(i - a0)
                        .and_then(|r| r.get(j - i))
                        .map(|c| &c.items)
                    else {
                        continue;
                    };
                    let mut k = 0;
                    for it in items {
                        if it.conj.is_some() {
                            continue;
                        }
                        if k >= self.roots_per_span {
                            break;
                        }
                        let Some(meaning) = root_meaning(&it.cat, &it.sem) else {
                            continue;
                        };
                        k += 1;
                        let mk = meaning_key(&meaning);
                        let world = match world_memo.get(&mk) {
                            Some(w) => *w,
                            None => {
                                let w = world_cost(&self.scorer, ctx, &meaning)
                                    .map_or(Bits::MAX / 4, |w| w.bits);
                                world_memo.insert(mk, w);
                                w
                            }
                        };
                        let root = self.scorer.ev(
                            Table::RootCat,
                            [0, 0],
                            cat_key(&it.cat),
                            cost::cat_prior(&self.model().program, &it.cat),
                        );
                        let mut gaps = 0;
                        for g in (i + 1)..j {
                            gaps += self.gap_bits(line, g, GAP_INTERNAL);
                        }
                        if j < n {
                            gaps += self.gap_bits(line, j, GAP_BOUNDARY);
                            gaps += self.scorer.ev(Table::FragMore, [0, 0], 1, 0);
                        }
                        let c = prev_cost + it.score - it.world_est + world + root + gaps;
                        cands.push((c, i, it.clone()));
                    }
                }
                if let Some((c, i, it)) = cands.into_iter().min_by_key(|x| x.0) {
                    let better = best
                        .get(j)
                        .is_none_or(|b| b.as_ref().is_none_or(|(bc, _)| c < *bc));
                    if better {
                        let mut frags = best
                            .get(i)
                            .and_then(|b| b.as_ref())
                            .map(|(_, f)| f.clone())
                            .unwrap_or_default();
                        frags.push((i, j, it));
                        if let Some(slot) = best.get_mut(j) {
                            *slot = Some((c, frags));
                        }
                    }
                }
            }
            // Emergency: if the sentence end is unreachable, cover it with
            // one-word unknown fragments from the last reachable point.
            if best.get(b0).is_some_and(Option::is_none) {
                let start = (a0..b0)
                    .rev()
                    .find(|p| best.get(*p).is_some_and(Option::is_some))
                    .unwrap_or(a0);
                let mut frags = best
                    .get(start)
                    .and_then(|b| b.as_ref())
                    .map(|(_, f)| f.clone())
                    .unwrap_or_default();
                for p in start..b0 {
                    if let Some(it) = cells.get(p - a0).and_then(|r| r.get(1)).and_then(|c| {
                        c.items.iter().find(|it| {
                            it.conj.is_none() && root_meaning(&it.cat, &it.sem).is_some()
                        })
                    }) {
                        frags.push((p, p + 1, it.clone()));
                    }
                }
                if let Some(slot) = best.get_mut(b0) {
                    *slot = Some((Bits::MAX / 4, frags));
                }
            }
        }
        best.pop().flatten().map(|(_, f)| f).unwrap_or_default()
    }

    fn gap_bits(&self, line: &Line, i: usize, class: u64) -> Bits {
        let g = line.gaps.get(i).map_or("", String::as_str);
        self.scorer.ev(
            Table::Gap,
            [class, 0],
            str_key(g),
            cost::spell(g.as_bytes()),
        )
    }

    /// Emit the exact derivation events top-down with instantiated
    /// categories.
    fn emit(&self, d: &Deriv, cat: &Cat, out: &mut WorldOut, syms: &mut Vec<(u64, Sym)>) {
        let s = &self.scorer;
        let ck = cat_key(cat);
        syms.push((ck, Sym::Cat(cat.clone())));
        match d {
            Deriv::Leaf {
                kind, stem, sem, ..
            } => {
                push_ev(s, out, Table::Expand, [ck, 0], K_LEX, 0);
                let feats = featset_of(sem);
                let infl_out = match kind {
                    LeafKind::Entry { infl: Some(i), .. } | LeafKind::Unknown { infl: Some(i) } => {
                        u64::from(*i)
                    }
                    _ => K_ABSENT,
                };
                let fk = featset_key(&feats);
                push_ev(s, out, Table::Infl, [ck, fk], infl_out, 0);
                syms.push((fk, Sym::Feats(feats)));
                let base = self.base_sem(sem, kind);
                syms.push((sem_key(&base), Sym::Term(base.clone())));
                let (o, spell) = match kind {
                    LeafKind::Entry { id, .. } => {
                        (TAG_ENTRY | u64::from(*id), cost::spell(stem.as_bytes()))
                    }
                    LeafKind::Name { id } => (TAG_NAME | u64::from(*id), 0),
                    LeafKind::Unknown { .. } => (str_key(stem), cost::spell(stem.as_bytes())),
                };
                push_ev(s, out, Table::Lex, [ck, sem_key(&base)], o, spell);
            }
            Deriv::Unary { rule, child, .. } => {
                push_ev(s, out, Table::Expand, [ck, 0], u64::from(*rule), 0);
                self.emit(child, child.cat(), out, syms);
            }
            Deriv::Coord {
                rule,
                left,
                conj,
                right,
                ..
            } => {
                push_ev(s, out, Table::Expand, [ck, 0], u64::from(*rule), 0);
                self.emit(left, cat, out, syms);
                self.emit(conj, conj.cat(), out, syms);
                self.emit(right, cat, out, syms);
            }
            Deriv::Binary {
                rule,
                kind,
                left,
                right,
                newcat,
                label,
                ..
            } => {
                push_ev(s, out, Table::Expand, [ck, 0], u64::from(*rule), 0);
                let p = &self.model().program;
                push_ev(
                    s,
                    out,
                    Table::NewCat,
                    [u64::from(*rule), ck],
                    cat_key(newcat),
                    cost::cat_prior(p, newcat),
                );
                syms.push((cat_key(newcat), Sym::Cat(newcat.clone())));
                push_ev(
                    s,
                    out,
                    Table::Split,
                    [u64::from(*kind), ck],
                    label_key(*label),
                    0,
                );
                let y = newcat.clone();
                let (lc, rc) = match (kind, cat.as_slash()) {
                    (0, _) => (Cat::slash(cat.clone(), Dir::Fwd, y.clone()), y),
                    (1, _) => (y.clone(), Cat::slash(cat.clone(), Dir::Bwd, y)),
                    (2, Some(s)) => (
                        Cat::slash(s.res.clone(), Dir::Fwd, y.clone()),
                        Cat::slash(y, Dir::Fwd, s.arg.clone()),
                    ),
                    (3, Some(s)) => (
                        Cat::slash(y.clone(), Dir::Bwd, s.arg.clone()),
                        Cat::slash(s.res.clone(), Dir::Bwd, y),
                    ),
                    (4, Some(s)) => (
                        Cat::slash(s.res.clone(), Dir::Fwd, y.clone()),
                        Cat::slash(y, Dir::Bwd, s.arg.clone()),
                    ),
                    (5, Some(s)) => (
                        Cat::slash(y.clone(), Dir::Fwd, s.arg.clone()),
                        Cat::slash(s.res.clone(), Dir::Bwd, y),
                    ),
                    _ => (left.cat().clone(), right.cat().clone()),
                };
                let lc = merge(&lc, left.cat());
                let rc = merge(&rc, right.cat());
                self.emit(left, &lc, out, syms);
                self.emit(right, &rc, out, syms);
            }
        }
    }
}

fn push_ev(s: &Scorer, out: &mut WorldOut, table: Table, ctx: [u64; 2], o: u64, spell: Bits) {
    let e = Event {
        table,
        ctx,
        out: o,
        spell,
    };
    out.bits += s.bits(&e);
    out.events.push(e);
}

/// The meaning of a fragment root: functor roots get `Missing` arguments.
pub fn root_meaning(cat: &Cat, sem: &Term) -> Option<Term> {
    let mut t = sem.clone();
    for _ in 0..cat.arity() {
        t = t.apply(&Term::head(Head::Missing))?;
    }
    t.is_tree().then_some(t)
}

/// Update a context with a meaning's mentions.
pub fn apply_mentions(ctx: &mut Context, mentions: &[Mention]) {
    for m in mentions {
        match m {
            Mention::New { concept, name } => {
                let id = ctx.new_entity(*concept, name.clone());
                ctx.mention(id);
            }
            Mention::Old(id) => ctx.mention(*id),
        }
    }
}
