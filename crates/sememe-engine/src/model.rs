//! A program compiled for fast use: lexicon indexed by word form, rules by
//! kind, the categories tried for unknown words, and cached event costs.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::category::{Cat, Dir};
use crate::cost::{
    self, Bits, Event, K_ANY, K_LEX, Sizes, Table, UNKNOWN_CATS, cat_key, event_bits, sem_key,
};
use crate::program::{Infl, Program, ROLE_ARG, ROLE_MOD, RuleKind};
use crate::term::{Head, Term};

/// A compiled lexical entry.
#[derive(Clone, Debug)]
pub struct LexItem {
    /// Entry ID.
    pub id: u32,
    /// Word form (or stem).
    pub form: String,
    /// Category.
    pub cat: Cat,
    /// Meaning.
    pub sem: Term,
}

/// A category tried for unknown words, with the meaning an unknown word of
/// that category gets.
#[derive(Clone, Debug)]
pub struct UnknownCat {
    /// The category.
    pub cat: Cat,
    /// The unknown meaning for it.
    pub sem: Term,
}

/// A compiled program.
#[derive(Debug)]
pub struct Model {
    /// The program.
    pub program: Arc<Program>,
    /// Sizes for base distributions.
    pub sizes: Sizes,
    /// Entries by form.
    pub by_form: FxHashMap<String, Vec<LexItem>>,
    /// Inflections.
    pub infls: Vec<(u32, Infl)>,
    /// Name entries `(id, category, concept)`.
    pub names: Vec<(u32, Cat, u32)>,
    /// Binary rules.
    pub binary: Vec<(u32, RuleKind)>,
    /// Unary rules (type raising, type change).
    pub unary: Vec<(u32, RuleKind)>,
    /// Coordination rules `(id, conjunction category)`.
    pub coord: Vec<(u32, Cat)>,
    /// Categories tried for unknown words.
    pub unknown_cats: Vec<UnknownCat>,
    /// Meaning keys of the global facts.
    pub fact_keys: rustc_hash::FxHashSet<u64>,
}

/// The meaning an unknown word of category `c` gets: an `Unknown` node for
/// atomic categories, `\x. x + mod: Unknown` for modifiers, and
/// `\y.. Unknown(arg: y, ..)` for other first-order functors. Categories
/// with functor arguments get none.
pub fn unknown_sem(c: &Cat) -> Option<Term> {
    match c {
        Cat::Atom(_) => Some(Term::head(Head::Unknown)),
        Cat::Slash(s) => {
            if c.is_modifier() {
                let body = Term::Add(
                    Arc::new(Term::Var(0)),
                    ROLE_MOD,
                    Arc::new(Term::head(Head::Unknown)),
                );
                return Some(Term::Lam(Arc::new(body)));
            }
            let args = c.args();
            if args.iter().any(|(_, a)| a.arity() > 0) {
                return None;
            }
            let _ = s;
            let k = args.len();
            let mut node = crate::term::Node::new(Head::Unknown);
            // The outermost argument is bound by the outermost lambda:
            // Var(k-1) inside k lambdas.
            for i in 0..k {
                let idx = u32::try_from(k - 1 - i).ok()?;
                node.roles.push((ROLE_ARG, Term::Var(idx)));
            }
            Some(Term::node(node).lams(k))
        }
    }
}

impl Model {
    /// Compile a program.
    pub fn new(program: Arc<Program>) -> Model {
        let sizes = Sizes::of(&program);
        let mut by_form: FxHashMap<String, Vec<LexItem>> = FxHashMap::default();
        for (id, e) in &program.entries {
            by_form.entry(e.form.clone()).or_default().push(LexItem {
                id: *id,
                form: e.form.clone(),
                cat: e.cat.clone(),
                sem: e.sem.clone(),
            });
        }
        let mut binary = Vec::new();
        let mut unary = Vec::new();
        let mut coord = Vec::new();
        for (id, r) in &program.rules {
            match &r.kind {
                RuleKind::FTr(..) | RuleKind::BTr(..) | RuleKind::Tc(..) => {
                    unary.push((*id, r.kind.clone()))
                }
                RuleKind::Coord(c) => coord.push((*id, c.clone())),
                k => binary.push((*id, k.clone())),
            }
        }
        let infls = program.infls.iter().map(|(k, v)| (*k, v.clone())).collect();
        let names = program
            .names
            .iter()
            .map(|(k, v)| (*k, v.cat.clone(), v.concept))
            .collect();
        let unknown_cats = choose_unknown_cats(&program, &sizes);
        let fact_keys = program
            .facts
            .values()
            .map(|f| cost::meaning_key(&f.term))
            .collect();
        Model {
            fact_keys,
            program,
            sizes,
            by_form,
            infls,
            names,
            binary,
            unary,
            coord,
            unknown_cats,
        }
    }
}

/// Pick the categories to try for unknown words: the leaf categories the
/// counts use most, then small categories from the atoms in prior order.
fn choose_unknown_cats(p: &Program, s: &Sizes) -> Vec<UnknownCat> {
    let names = p.cat_names();
    let mut scored: BTreeMap<(u64, String), Cat> = BTreeMap::new();
    // Leaf categories by how often they expand to a leaf.
    for ((t, ctx), cc) in &p.counts.tables {
        if *t != Table::Expand as u8 || ctx[0] == K_ANY {
            continue;
        }
        let Some(q) = cc.outs.get(&K_LEX) else {
            continue;
        };
        let Some(text) = p.counts.syms.get(&ctx[0]) else {
            continue;
        };
        let Ok(cat) = Cat::parse(text, &names) else {
            continue;
        };
        if cat.arity() > 2 || unknown_sem(&cat).is_none() {
            continue;
        }
        let n = cost::level_value(*q);
        scored.insert((u64::MAX - n, text.clone()), cat);
    }
    let mut out: Vec<UnknownCat> = Vec::new();
    for cat in scored.into_values() {
        if out.len() >= UNKNOWN_CATS {
            break;
        }
        if let Some(sem) = unknown_sem(&cat) {
            out.push(UnknownCat { cat, sem });
        }
    }
    // Fill with small categories in prior order.
    let mut defaults: Vec<(Bits, Cat)> = Vec::new();
    let atoms: Vec<Cat> = (0..p.atoms.len())
        .filter_map(|i| u16::try_from(i).ok().map(Cat::atom))
        .collect();
    for a in &atoms {
        defaults.push((cost::cat_prior(p, a), a.clone()));
        for b in &atoms {
            for d in [Dir::Fwd, Dir::Bwd] {
                let c = Cat::slash(a.clone(), d, b.clone());
                defaults.push((cost::cat_prior(p, &c), c));
            }
        }
    }
    defaults.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    // Modifiers of the sentence category come first among functors so any
    // word sequence can be read (as a chain of sentence modifiers).
    let s_mods = [
        Cat::slash(Cat::atom(0), Dir::Fwd, Cat::atom(0)),
        Cat::slash(Cat::atom(0), Dir::Bwd, Cat::atom(0)),
    ];
    let mut ordered: Vec<Cat> = atoms.clone();
    ordered.extend(s_mods.iter().cloned());
    ordered.extend(defaults.into_iter().map(|(_, c)| c));
    for cat in ordered {
        if out.len() >= UNKNOWN_CATS.max(atoms.len() + 2) {
            break;
        }
        if out.iter().any(|u| u.cat == cat) {
            continue;
        }
        if let Some(sem) = unknown_sem(&cat) {
            out.push(UnknownCat { cat, sem });
        }
    }
    let _ = s;
    out
}

/// Event costs with a per-thread cache.
pub struct Scorer<'m> {
    /// The model.
    pub model: &'m Model,
    cache: RefCell<FxHashMap<Event, Bits>>,
}

impl<'m> Scorer<'m> {
    /// A scorer for a model.
    pub fn new(model: &'m Model) -> Scorer<'m> {
        Scorer {
            model,
            cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Bits of an event.
    pub fn bits(&self, e: &Event) -> Bits {
        if let Some(b) = self.cache.borrow().get(e) {
            return *b;
        }
        let b = event_bits(&self.model.program, &self.model.sizes, e);
        let mut c = self.cache.borrow_mut();
        if c.len() > 4_000_000 {
            c.clear();
        }
        c.insert(*e, b);
        b
    }

    /// Bits of an event built from parts.
    pub fn ev(&self, table: Table, ctx: [u64; 2], out: u64, spell: Bits) -> Bits {
        self.bits(&Event {
            table,
            ctx,
            out,
            spell,
        })
    }

    /// Cached category key (no cache needed: hashing is cheap).
    pub fn cat_key(&self, c: &Cat) -> u64 {
        cat_key(c)
    }

    /// Meaning key for lexical contexts.
    pub fn sem_key(&self, t: &Term) -> u64 {
        sem_key(t)
    }
}
