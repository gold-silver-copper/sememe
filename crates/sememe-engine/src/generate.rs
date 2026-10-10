//! Forward mode: meaning to text.
//!
//! Head-driven realization. For each meaning node, pick the cheapest
//! lexical entry whose meaning is that node's head (with some children as
//! arguments and others as built-in constants), realize the argument
//! children, then attach modifier, marker and feature words wherever their
//! category fits the phrase built so far. The words come out in the order
//! the categories' slashes dictate. The result is checked by parsing it
//! back (see the round-trip test).

use crate::category::{Bindings, Cat, Dir, substitute, unify};
use crate::cost::{self, Bits, K_LEX, TAG_ENTRY, TAG_NAME, Table, cat_key, meaning_key, sem_key};
use crate::model::{LexItem, Model, Scorer};
use crate::term::{Head, Label, LabelKind, Node, Term};

/// A realized phrase.
#[derive(Clone, Debug)]
pub struct Phrase {
    /// Category.
    pub cat: Cat,
    /// Meaning.
    pub sem: Term,
    /// Words.
    pub words: Vec<String>,
    /// Lexical bits of its leaves (for choosing among alternatives).
    pub bits: Bits,
}

/// What a functor entry contributes to its argument.
#[derive(Clone, Debug)]
enum Contribution {
    /// Adds a role child equal to this meaning.
    Mod(u32, u64),
    /// Adds a role child given as its own argument (a marker word).
    Marker(u32),
    /// Sets these features.
    Feats(Vec<(u32, u16)>),
}

fn feats_of_body(t: &Term, out: &mut Vec<(u32, u16)>) {
    match t {
        Term::Feat(b, f, v) => {
            out.push((*f, *v));
            feats_of_body(b, out);
        }
        Term::Lam(b) => feats_of_body(b, out),
        _ => {}
    }
}

fn contribution(e: &LexItem) -> Option<Contribution> {
    let Label { kind, name } = e.sem.classify();
    match kind {
        LabelKind::Mod => {
            // The added child is a closed constant.
            let mut child = None;
            find_added(e.sem.body(), &mut child);
            child.map(|c| Contribution::Mod(name, meaning_key(&c)))
        }
        // A marker: \y P.. (P ..) + r: y. Its first argument fills a role
        // it adds to the phrase it then modifies.
        LabelKind::Arg if e.cat.arity() >= 2 => {
            matches!(e.sem.body(), Term::Add(..)).then_some(Contribution::Marker(name))
        }
        LabelKind::Feat => {
            let mut fs = Vec::new();
            feats_of_body(&e.sem, &mut fs);
            fs.sort();
            Some(Contribution::Feats(fs))
        }
        _ => None,
    }
}

fn find_added(t: &Term, out: &mut Option<Term>) {
    match t {
        Term::Add(b, _, u) => {
            if u.is_tree() {
                *out = Some((**u).clone());
            } else {
                find_added(b, out);
            }
        }
        Term::Lam(b) => find_added(b, out),
        _ => {}
    }
}

/// A functor attached to a phrase: the result, the child it covered, the
/// features it set.
type Attachment = (Phrase, Option<usize>, Vec<(u32, u16)>);

/// The generator.
pub struct Generator<'m> {
    scorer: Scorer<'m>,
    /// Head entries by head concept key.
    heads: rustc_hash::FxHashMap<u64, Vec<LexItem>>,
    /// Pronoun entries.
    refs: Vec<LexItem>,
    /// Functor entries and what they contribute.
    functors: Vec<(LexItem, Contribution)>,
}

impl<'m> Generator<'m> {
    /// A generator for a model.
    pub fn new(model: &'m Model) -> Generator<'m> {
        let mut heads: rustc_hash::FxHashMap<u64, Vec<LexItem>> = rustc_hash::FxHashMap::default();
        let mut refs = Vec::new();
        let mut functors = Vec::new();
        for es in model.by_form.values() {
            for e in es {
                if let Some(c) = contribution(e) {
                    functors.push((e.clone(), c));
                    continue;
                }
                if let Term::Node(n) = e.sem.body() {
                    match n.head {
                        Head::Ref(_) => refs.push(e.clone()),
                        _ => heads
                            .entry(cost::head_key(&n.head))
                            .or_default()
                            .push(e.clone()),
                    }
                }
            }
        }
        Generator {
            scorer: Scorer::new(model),
            heads,
            refs,
            functors,
        }
    }

    fn lex_bits(&self, e: &LexItem) -> Bits {
        let ck = cat_key(&e.cat);
        self.scorer.ev(Table::Expand, [ck, 0], K_LEX, 0)
            + self.scorer.ev(
                Table::Lex,
                [ck, sem_key(&e.sem)],
                TAG_ENTRY | u64::from(e.id),
                cost::spell(e.form.as_bytes()),
            )
    }

    /// Realize a closed meaning as a phrase of any category.
    pub fn realize(&self, m: &Term) -> Option<Phrase> {
        let n = m.as_node()?;
        self.node(n, 0)
    }

    fn node(&self, v: &Node, depth: usize) -> Option<Phrase> {
        if depth > 12 {
            return None;
        }
        let mut best: Option<Phrase> = None;
        for (head, args, consts) in self.head_choices(v) {
            if let Some(p) = self.build(v, &head, &args, &consts, depth)
                && best.as_ref().is_none_or(|b| p.bits < b.bits)
            {
                best = Some(p);
            }
        }
        best
    }

    /// Head entries for a node: `(phrase of the head word, argument child
    /// indices in slot order, child indices covered as constants)`.
    fn head_choices(&self, v: &Node) -> Vec<(Phrase, Vec<usize>, Vec<usize>)> {
        let mut out = Vec::new();
        if let Head::Ref(d) = v.head {
            for e in &self.refs {
                if e.sem
                    .body()
                    .as_node()
                    .is_some_and(|n| n.head == Head::Ref(d))
                {
                    out.push((self.leaf(e), Vec::new(), Vec::new()));
                }
            }
            return out;
        }
        if let Some(name) = &v.name {
            // Proper names realize as themselves.
            for (nid, cat, concept) in &self.scorer.model.names {
                if v.head == Head::Concept(*concept) {
                    let mut node = Node::new(v.head.clone());
                    node.name = Some(name.clone());
                    let bits = self.scorer.ev(
                        Table::Lex,
                        [cat_key(cat), sem_key(&Term::node(node.clone()))],
                        TAG_NAME | u64::from(*nid),
                        0,
                    );
                    out.push((
                        Phrase {
                            cat: cat.clone(),
                            sem: Term::node(node),
                            words: vec![name.to_string()],
                            bits,
                        },
                        Vec::new(),
                        Vec::new(),
                    ));
                }
            }
        }
        let Some(cands) = self.heads.get(&cost::head_key(&v.head)) else {
            return out;
        };
        for e in cands {
            let Some(n) = e.sem.body().as_node() else {
                continue;
            };
            if n.name.is_some() && n.name != v.name {
                continue;
            }
            if !n.feats.iter().all(|f| v.feats.contains(f)) {
                continue;
            }
            // Map the entry's roles onto the node's children.
            let k = e.sem.lam_depth() as u32;
            let mut slot_child: Vec<Option<usize>> = vec![None; k as usize];
            let mut consts = Vec::new();
            let mut used: Vec<bool> = vec![false; v.roles.len()];
            let mut ok = true;
            for (r, t) in &n.roles {
                let found = v.roles.iter().enumerate().position(|(i, (vr, vt))| {
                    !used.get(i).copied().unwrap_or(true)
                        && vr == r
                        && match t {
                            Term::Var(_) => true,
                            other => meaning_key(other) == meaning_key(vt),
                        }
                });
                match found {
                    Some(i) => {
                        if let Some(u) = used.get_mut(i) {
                            *u = true;
                        }
                        match t {
                            Term::Var(j) => {
                                // Var(j) is bound by lambda number k-1-j.
                                let slot = (k - 1).saturating_sub(*j) as usize;
                                if let Some(s) = slot_child.get_mut(slot) {
                                    *s = Some(i);
                                }
                            }
                            _ => consts.push(i),
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || slot_child.iter().any(Option::is_none) {
                continue;
            }
            let args: Vec<usize> = slot_child.into_iter().flatten().collect();
            out.push((self.leaf(e), args, consts));
        }
        out
    }

    fn leaf(&self, e: &LexItem) -> Phrase {
        Phrase {
            cat: e.cat.clone(),
            sem: e.sem.clone(),
            words: e.form.split(' ').map(str::to_string).collect(),
            bits: self.lex_bits(e),
        }
    }

    fn build(
        &self,
        v: &Node,
        head: &Phrase,
        args: &[usize],
        consts: &[usize],
        depth: usize,
    ) -> Option<Phrase> {
        let mut cur = head.clone();
        let mut done: Vec<bool> = vec![false; v.roles.len()];
        for &i in args.iter().chain(consts) {
            if let Some(d) = done.get_mut(i) {
                *d = true;
            }
        }
        let head_feats: Vec<(u32, u16)> = head
            .sem
            .body()
            .as_node()
            .map(|n| n.feats.clone())
            .unwrap_or_default();
        let mut feats_left: Vec<(u32, u16)> = v
            .feats
            .iter()
            .filter(|f| !head_feats.contains(f))
            .copied()
            .collect();
        let mut next_arg = 0usize;
        loop {
            // Attach whatever fits the current category: modifiers that keep
            // it first, then ones that change it.
            self.attach_functors(v, &mut cur, &mut done, &mut feats_left, depth)?;
            let Some(&ai) = args.get(next_arg) else { break };
            let (_, child) = v.roles.get(ai)?;
            let s = cur.cat.as_slash()?.clone();
            let cp = self.node(child.as_node()?, depth + 1)?;
            let mut b = Bindings::default();
            if !unify(&s.arg, 0, &cp.cat, 1, &mut b) {
                return None;
            }
            let sem = cur.sem.apply(&cp.sem)?;
            let words = match s.dir {
                Dir::Fwd => [cur.words.clone(), cp.words].concat(),
                Dir::Bwd => [cp.words, cur.words.clone()].concat(),
            };
            cur = Phrase {
                cat: substitute(&s.res, 0, &b),
                sem,
                words,
                bits: cur.bits + cp.bits,
            };
            next_arg += 1;
        }
        if done.iter().all(|d| *d) && feats_left.is_empty() {
            Some(cur)
        } else {
            None
        }
    }

    fn attach_functors(
        &self,
        v: &Node,
        cur: &mut Phrase,
        done: &mut [bool],
        feats_left: &mut Vec<(u32, u16)>,
        depth: usize,
    ) -> Option<()> {
        for pass in 0..2 {
            let mut progress = true;
            while progress {
                progress = false;
                let mut best: Option<Attachment> = None;
                for (e, c) in &self.functors {
                    // Which category does the functor apply to?
                    let (fcat, extra) = match c {
                        Contribution::Marker(_) => {
                            let s = e.cat.as_slash()?;
                            (s.res.clone(), Some(s.clone()))
                        }
                        _ => (e.cat.clone(), None),
                    };
                    let Some(fs) = fcat.as_slash() else { continue };
                    let keeps = fs.res.same_shape(&fs.arg);
                    if (pass == 0) != keeps {
                        continue;
                    }
                    let mut b = Bindings::default();
                    if !unify(&fs.arg, 0, &cur.cat, 1, &mut b) {
                        continue;
                    }
                    let (child_idx, feats) = match c {
                        Contribution::Mod(r, key) => {
                            let i = v.roles.iter().enumerate().position(|(i, (vr, vt))| {
                                !done.get(i).copied().unwrap_or(true)
                                    && vr == r
                                    && meaning_key(vt) == *key
                            });
                            match i {
                                Some(i) => (Some(i), Vec::new()),
                                None => continue,
                            }
                        }
                        Contribution::Marker(r) => {
                            let i = v.roles.iter().enumerate().position(|(i, (vr, _))| {
                                !done.get(i).copied().unwrap_or(true) && vr == r
                            });
                            match i {
                                Some(i) => (Some(i), Vec::new()),
                                None => continue,
                            }
                        }
                        Contribution::Feats(fs2) => {
                            if fs2.is_empty() || !fs2.iter().all(|f| feats_left.contains(f)) {
                                continue;
                            }
                            (None, fs2.clone())
                        }
                    };
                    // Build the functor phrase (a marker first takes its object).
                    let mut fp = self.leaf(e);
                    if let (Contribution::Marker(_), Some(ms), Some(i)) = (c, &extra, child_idx) {
                        let (_, child) = v.roles.get(i)?;
                        let Some(cp) = child.as_node().and_then(|n| self.node(n, depth + 1)) else {
                            continue;
                        };
                        let mut b2 = Bindings::default();
                        if !unify(&ms.arg, 0, &cp.cat, 1, &mut b2) {
                            continue;
                        }
                        let Some(sem) = fp.sem.apply(&cp.sem) else {
                            continue;
                        };
                        fp = Phrase {
                            words: match ms.dir {
                                Dir::Fwd => [fp.words.clone(), cp.words].concat(),
                                Dir::Bwd => [cp.words, fp.words.clone()].concat(),
                            },
                            cat: substitute(&ms.res, 0, &b2),
                            sem,
                            bits: fp.bits + cp.bits,
                        };
                    }
                    let Some(sem) = fp.sem.apply(&cur.sem) else {
                        continue;
                    };
                    let words = match fs.dir {
                        Dir::Fwd => [fp.words.clone(), cur.words.clone()].concat(),
                        Dir::Bwd => [cur.words.clone(), fp.words.clone()].concat(),
                    };
                    let cand = Phrase {
                        cat: substitute(&fs.res, 0, &b),
                        sem,
                        words,
                        bits: cur.bits + fp.bits,
                    };
                    if best.as_ref().is_none_or(|(b0, _, _)| cand.bits < b0.bits) {
                        best = Some((cand, child_idx, feats));
                    }
                }
                if let Some((p, ci, fs)) = best {
                    *cur = p;
                    if let Some(i) = ci
                        && let Some(d) = done.get_mut(i)
                    {
                        *d = true;
                    }
                    feats_left.retain(|f| !fs.contains(f));
                    progress = true;
                }
            }
        }
        Some(())
    }
}

/// Speak a meaning: the realized words, or `None` if the lexicon cannot
/// express it.
pub fn speak(model: &Model, m: &Term) -> Option<Phrase> {
    Generator::new(model).realize(m)
}

/// Render words as a sentence: capitalized, with a full stop.
pub fn sentence(words: &[String]) -> String {
    let mut s = words.join(" ");
    if let Some(f) = s.get(..1) {
        let up = f.to_uppercase();
        s = format!("{up}{}", s.get(1..).unwrap_or_default());
    }
    s.push('.');
    s
}

/// A small deterministic random number generator (xorshift).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Seeded generator.
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    /// Next value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Draw an outcome from the stored counts of the most specific context in
/// an event's back-off chain that has any.
fn draw(
    model: &Model,
    table: Table,
    ctx: [u64; 2],
    rng: &mut Rng,
    ok: &dyn Fn(u64) -> bool,
) -> Option<u64> {
    let probe = cost::Event {
        table,
        ctx,
        out: 0,
        spell: 0,
    };
    for c in cost::chain(&model.program, &probe) {
        let Some(cc) = model.program.counts.tables.get(&(table as u8, c)) else {
            continue;
        };
        let outs: Vec<(u64, u64)> = cc
            .outs
            .iter()
            .filter(|(o, _)| ok(**o))
            .map(|(o, q)| (*o, cost::level_value(*q)))
            .collect();
        let total: u64 = outs.iter().map(|x| x.1).sum();
        if total == 0 {
            continue;
        }
        let mut r = rng.next_u64() % total;
        for (o, w) in outs {
            if r < w {
                return Some(o);
            }
            r -= w;
        }
    }
    None
}

/// Sample a meaning from the world model: a root head, then features and
/// role children drawn from the stored counts, to a bounded depth.
pub fn sample_meaning(model: &Model, rng: &mut Rng) -> Option<Term> {
    sample_node(model, cost::K_ROOT, cost::K_NONE, rng, 0)
}

fn sample_node(model: &Model, role: u64, parent: u64, rng: &mut Rng, depth: usize) -> Option<Term> {
    let concept_ok =
        |o: u64| u32::try_from(o).is_ok_and(|c| model.program.concepts.contains_key(&c));
    let head = draw(model, Table::Concept, [role, parent], rng, &concept_ok)?;
    let c = u32::try_from(head).ok()?;
    let mut n = Node::new(Head::Concept(c));
    let mut prev = cost::K_ROOT;
    for _ in 0..4 {
        let f = draw(model, Table::FeatNext, [prev, head], rng, &|_| true)?;
        if f == cost::K_ABSENT {
            break;
        }
        let (fid, v) = ((f >> 16) as u32, (f & 0xffff) as u16);
        if n.feats.iter().any(|(x, _)| *x == fid) || u64::from(fid) < prev && prev != cost::K_ROOT {
            break;
        }
        n.feats.push((fid, v));
        prev = u64::from(fid);
    }
    let mut prev = cost::K_ROOT;
    for _ in 0..4 {
        if depth >= 3 {
            break;
        }
        let r = draw(model, Table::RoleNext, [prev, head], rng, &|_| true)?;
        if r == cost::K_ABSENT {
            break;
        }
        let rid = u32::try_from(r).ok()?;
        if let Some(child) = sample_node(model, r, head, rng, depth + 1) {
            n.roles.push((rid, child));
        }
        prev = r;
    }
    crate::term::sort_roles(&mut n.roles);
    n.feats.sort();
    Some(Term::node(n))
}
