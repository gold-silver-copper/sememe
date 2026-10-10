//! Learning lexical entries from meanings (GENLEX).
//!
//! The teacher's annotation gives a meaning tree with word anchors. That is
//! a dependency structure: each node has head words, feature words (a
//! determiner, an auxiliary), marked children (a preposition plus its
//! object), unmarked children and unanchored words. This module turns it
//! into a CCG derivation with application only:
//!
//! - a node's head word takes its **core** children (agent, patient, ...)
//!   as arguments, right ones first (nearest first), then left ones;
//! - other children (`attr`, `time`, ...) are modifiers `X/X` or `X\X`
//!   that add their role; with a marker word, the marker is the functor
//!   `(X|X)|NP` that adds the role with its object;
//! - feature words are functors that set their features (the outermost
//!   determiner of a noun turns `N` into `NP`);
//! - unanchored words are identity functors;
//! - unsaid children become constants in the head word's meaning.
//!
//! The derivation's leaves are the candidate entries. The composed meaning
//! is checked against the target; candidates are evidence, not truth: the
//! learner keeps only the ones that lower the total cost.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use sememe_engine::category::{Cat, Dir};
use sememe_engine::cost::meaning_key;
use sememe_engine::program::{ATOM_N, ATOM_NP, ATOM_S, Program};
use sememe_engine::term::{Deixis, Head, Node, Term, TermLookup};
use sememe_engine::text::Line;

use crate::annotate::{AnnHead, AnnNode, Unit};

/// Roles realized as arguments of their head word.
pub const CORE_ROLES: [&str; 7] = [
    "agent",
    "patient",
    "theme",
    "experiencer",
    "stimulus",
    "recipient",
    "content",
];

/// One candidate lexical entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cand {
    /// Word form (words joined by a space).
    pub form: String,
    /// Category text.
    pub cat: String,
    /// Meaning text.
    pub sem: String,
}

/// The result of converting one unit.
#[derive(Clone, Debug)]
pub struct Converted {
    /// The unit's target meaning.
    pub meaning: Term,
    /// Candidate entries, one per leaf.
    pub entries: Vec<Cand>,
}

#[derive(Clone, Debug)]
struct GNode {
    head: Head,
    name: Option<Arc<str>>,
    /// Features `(feature, value, anchor words)`.
    feats: Vec<(u32, u16, Vec<usize>)>,
    anchors: Vec<usize>,
    /// Children `(role, marker words, node)`.
    kids: Vec<(u32, Vec<usize>, GNode)>,
    /// Unanchored words attached to this node.
    loose: Vec<usize>,
}

fn to_gnode(p: &Program, a: &AnnNode) -> Result<GNode> {
    let head = match &a.head {
        AnnHead::Concept(c) => Head::Concept(
            p.concept_by_name(c)
                .ok_or_else(|| anyhow!("unknown concept {c}"))?,
        ),
        AnnHead::Query => Head::Query,
        AnnHead::Speaker => Head::Ref(Deixis::Speaker),
        AnnHead::Addressee => Head::Ref(Deixis::Addressee),
        AnnHead::Other | AnnHead::Ref(_) => Head::Ref(Deixis::Other),
    };
    let mut feats = Vec::new();
    for (f, v, an) in &a.feats {
        let fid = p
            .feat_by_name(f)
            .ok_or_else(|| anyhow!("unknown feature {f}"))?;
        let vid = p
            .val_id(fid, v)
            .ok_or_else(|| anyhow!("unknown value {f}={v}"))?;
        feats.push((fid, vid, an.clone()));
    }
    feats.sort_by_key(|x| x.0);
    feats.dedup_by_key(|x| x.0);
    let mut kids = Vec::new();
    for (r, an, c) in &a.roles {
        let rid = p
            .role_by_name(r)
            .ok_or_else(|| anyhow!("unknown role {r}"))?;
        kids.push((rid, an.clone(), to_gnode(p, c)?));
    }
    Ok(GNode {
        head,
        name: a.name.as_deref().map(Arc::from),
        feats,
        anchors: a.anchors.clone(),
        kids,
        loose: Vec::new(),
    })
}

/// Make anchors buildable: a head anchored to scattered words keeps its
/// last contiguous run (the others become unanchored); a feature anchored to
/// several words keeps the one nearest the head.
fn tidy(g: &mut GNode) {
    g.anchors.sort_unstable();
    g.anchors.dedup();
    if g.anchors.is_empty() {
        // A head with no word of its own ("where's dad?": the copula only
        // carries tense) is expressed by its first feature word, which then
        // carries those features lexically.
        if let Some(w) = g
            .feats
            .iter()
            .flat_map(|(_, _, an)| an.iter().copied())
            .min()
        {
            g.anchors = vec![w];
            for (_, _, an) in &mut g.feats {
                an.retain(|x| *x != w);
            }
        }
    }
    if let Some(&last) = g.anchors.last() {
        let mut run = vec![last];
        for &w in g.anchors.iter().rev().skip(1) {
            if run.last().is_some_and(|&r| r == w + 1) {
                run.push(w);
            } else {
                break;
            }
        }
        run.reverse();
        g.anchors = run;
    }
    let h = g.anchors.first().copied().unwrap_or(0);
    for (_, _, an) in &mut g.feats {
        if an.len() > 1 {
            an.sort_by_key(|w| w.abs_diff(h));
            an.truncate(1);
        }
    }
    for (_, m, k) in &mut g.kids {
        m.sort_unstable();
        m.dedup();
        tidy(k);
    }
}

/// The meaning tree of a node (all features, all children).
fn meaning_of(g: &GNode) -> Term {
    let mut n = Node::new(g.head.clone());
    n.name = g.name.clone();
    n.feats = g.feats.iter().map(|(f, v, _)| (*f, *v)).collect();
    n.feats.sort();
    let mut roles: Vec<(u32, Term)> = g.kids.iter().map(|(r, _, k)| (*r, meaning_of(k))).collect();
    sememe_engine::term::sort_roles(&mut roles);
    n.roles = roles;
    Term::node(n)
}

/// Every word a node's phrase covers.
fn words_of(g: &GNode, out: &mut Vec<usize>) {
    out.extend(&g.anchors);
    for (_, _, an) in &g.feats {
        out.extend(an);
    }
    out.extend(&g.loose);
    for (_, m, k) in &g.kids {
        out.extend(m);
        words_of(k, out);
    }
}

fn span(g: &GNode) -> Option<(usize, usize)> {
    let mut w = Vec::new();
    words_of(g, &mut w);
    Some((*w.iter().min()?, *w.iter().max()?))
}

fn expressed(g: &GNode) -> bool {
    span(g).is_some()
}

fn is_clause(p: &Program, g: &GNode) -> bool {
    let Head::Concept(c) = g.head else {
        return false;
    };
    ["Event", "State", "Discourse"]
        .iter()
        .filter_map(|n| p.concept_by_name(n))
        .any(|e| p.is_a(c, e))
}

/// Attach unanchored words: to the clause constituent immediately to their
/// right if there is one, otherwise to the smallest constituent around them,
/// otherwise to the root.
fn attach_loose(p: &Program, root: &mut GNode, start: usize, end: usize) {
    let mut used = Vec::new();
    words_of(root, &mut used);
    for w in start..end {
        if used.contains(&w) {
            continue;
        }
        if !attach_right_clause(p, root, w) && !attach_inside(root, w) {
            root.loose.push(w);
        }
    }
}

fn attach_right_clause(p: &Program, g: &mut GNode, w: usize) -> bool {
    for (_, m, k) in &mut g.kids {
        if m.is_empty() && is_clause(p, k) && span(k).is_some_and(|(a, _)| a == w + 1) {
            k.loose.push(w);
            return true;
        }
        if attach_right_clause(p, k, w) {
            return true;
        }
    }
    false
}

fn attach_inside(g: &mut GNode, w: usize) -> bool {
    for (_, _, k) in &mut g.kids {
        if span(k).is_some_and(|(a, b)| a < w && w < b) {
            if !attach_inside(k, w) {
                k.loose.push(w);
            }
            return true;
        }
    }
    false
}

/// A built constituent: its category, meaning and leaves.
struct Built {
    cat: Cat,
    sem: Term,
    leaves: Vec<(Vec<usize>, Cat, Term)>,
}

enum Dep {
    Arg(usize),
    Marked(usize),
    Mod(usize),
    Feat(Vec<(u32, u16)>, Vec<usize>),
    Loose(usize),
}

fn slash(res: Cat, right: bool, arg: Cat) -> Cat {
    Cat::slash(res, if right { Dir::Fwd } else { Dir::Bwd }, arg)
}

fn lam(t: Term) -> Term {
    Term::Lam(Arc::new(t))
}

/// A modifier meaning for a category with `k` arguments:
/// `\P z1..zk. f(P z1..zk)`, where `f` builds the body from the applied
/// term. With `outer` extra binders outside `P` (a marker's object).
fn lifted(k: usize, f: &dyn Fn(Term) -> Term) -> Result<Term> {
    let k32 = u32::try_from(k)?;
    let mut applied = Term::Var(k32);
    for i in (0..k32).rev() {
        applied = Term::App(Arc::new(applied), Arc::new(Term::Var(i)));
    }
    Ok(f(applied).lams(k + 1))
}

/// Build the derivation of a node's phrase.
fn build(p: &Program, g: &GNode, core: &[u32], top: bool) -> Result<Built> {
    let op = p.role_by_name("op");
    let coordination = !g.kids.is_empty() && g.kids.iter().all(|(r, _, _)| Some(*r) == op);
    let has_core = g
        .kids
        .iter()
        .any(|(r, m, k)| m.is_empty() && core.contains(r) && expressed(k));
    // Predicates are clauses: an event or state, or anything with an
    // expressed core argument ("that's not my house" = House(theme: that)).
    let clause = is_clause(p, g) || has_core;
    let _ = top;
    let mut heads = g.anchors.clone();
    heads.sort_unstable();
    heads.dedup();
    let (Some(&h0), Some(&h1)) = (heads.first(), heads.last()) else {
        bail!("a node with expressed parts has no head word");
    };
    if h1 + 1 - h0 != heads.len() {
        bail!("head words are not contiguous");
    }
    let pos = |a: usize| -> bool { a > h1 };
    // Dependents with their blocks.
    let mut deps: Vec<(usize, usize, Dep)> = Vec::new();
    let mut kid_built: BTreeMap<usize, Built> = BTreeMap::new();
    let mut consts: Vec<(u32, Term)> = Vec::new();
    for (i, (r, m, k)) in g.kids.iter().enumerate() {
        if !expressed(k) {
            consts.push((*r, meaning_of(k)));
            continue;
        }
        let (a, b) = span(k).unwrap_or((h0, h0));
        let (mut lo, mut hi) = (a, b);
        for &w in m {
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let kb = build(p, k, core, false)?;
        let simple = k.kids.iter().all(|(_, _, c)| !expressed(c))
            && k.feats.iter().all(|(_, _, an)| an.is_empty())
            && k.loose.is_empty();
        let dep = if !m.is_empty() {
            Dep::Marked(i)
        } else if coordination || core.contains(r) {
            Dep::Arg(i)
        } else if simple {
            Dep::Mod(i)
        } else {
            Dep::Arg(i)
        };
        kid_built.insert(i, kb);
        deps.push((lo, hi, dep));
    }
    // Feature words, grouped by word.
    let mut by_word: BTreeMap<usize, Vec<(u32, u16)>> = BTreeMap::new();
    for (f, v, an) in &g.feats {
        for &w in an {
            by_word.entry(w).or_default().push((*f, *v));
        }
    }
    for (w, fs) in by_word {
        deps.push((w, w, Dep::Feat(fs, vec![w])));
    }
    for &w in &g.loose {
        deps.push((w, w, Dep::Loose(w)));
    }
    // Blocks must not overlap the head or each other.
    let mut blocks: Vec<(usize, usize)> = deps.iter().map(|(a, b, _)| (*a, *b)).collect();
    blocks.push((h0, h1));
    blocks.sort_unstable();
    for w in blocks.windows(2) {
        if let [x, y] = w
            && y.0 <= x.1
        {
            bail!("crossing or overlapping phrases (non-projective)");
        }
    }
    // Processing order: right side nearest first, then left side nearest first.
    deps.sort_by_key(|(a, b, _)| {
        if pos(*a) {
            (0, *a)
        } else {
            (1, usize::MAX - *b)
        }
    });

    // Base category.
    let has_det = g.feats.iter().any(|(_, _, an)| an.iter().any(|w| *w < h0));
    let first_conjunct = if coordination {
        deps.iter()
            .filter_map(|(a, _, d)| match d {
                Dep::Arg(i) => Some((*a, *i)),
                _ => None,
            })
            .min()
            .and_then(|(_, i)| kid_built.get(&i))
            .map(|k| k.cat.clone())
    } else {
        None
    };
    let base = if let Some(c) = first_conjunct {
        c
    } else if clause {
        Cat::atom(ATOM_S)
    } else if has_det {
        Cat::atom(ATOM_N)
    } else {
        Cat::atom(ATOM_NP)
    };
    // Argument slots in processing order.
    let args: Vec<(usize, bool)> = deps
        .iter()
        .filter_map(|(a, _, d)| match d {
            Dep::Arg(i) => Some((*i, pos(*a))),
            _ => None,
        })
        .collect();
    let mut cat = base.clone();
    for (i, right) in args.iter().rev() {
        let kc = kid_built
            .get(i)
            .map(|k| k.cat.clone())
            .ok_or_else(|| anyhow!("missing child"))?;
        cat = slash(cat, *right, kc);
    }
    // Head meaning: \a0 .. a(k-1). Node(roles: args as variables, unsaid
    // children as constants).
    let k = args.len();
    let mut node = Node::new(g.head.clone());
    node.name = g.name.clone();
    node.feats = g
        .feats
        .iter()
        .filter(|(_, _, an)| an.is_empty())
        .map(|(f, v, _)| (*f, *v))
        .collect();
    node.feats.sort();
    let mut roles: Vec<(u32, Term)> = consts;
    for (j, (i, _)) in args.iter().enumerate() {
        let r = g
            .kids
            .get(*i)
            .map(|x| x.0)
            .ok_or_else(|| anyhow!("missing role"))?;
        roles.push((r, Term::Var(u32::try_from(k - 1 - j)?)));
    }
    sememe_engine::term::sort_roles(&mut roles);
    node.roles = roles;
    let head_sem = Term::node(node).lams(k);
    let mut leaves = vec![(heads.clone(), cat.clone(), head_sem.clone())];
    let mut cur_cat = cat;
    let mut cur_sem = head_sem;
    // The outermost determiner-like feature word on the left turns N into NP.
    let last_left_feat = deps
        .iter()
        .rposition(|(a, _, d)| matches!(d, Dep::Feat(..)) && !pos(*a));
    for (n, (a, _, dep)) in deps.iter().enumerate() {
        let right = pos(*a);
        match dep {
            Dep::Arg(i) => {
                let kb = kid_built
                    .remove(i)
                    .ok_or_else(|| anyhow!("missing child"))?;
                let s = cur_cat
                    .as_slash()
                    .ok_or_else(|| anyhow!("no slot left"))?
                    .clone();
                cur_cat = s.res.clone();
                cur_sem = cur_sem
                    .apply(&kb.sem)
                    .ok_or_else(|| anyhow!("argument does not apply"))?;
                leaves.extend(kb.leaves);
            }
            Dep::Mod(i) => {
                let kb = kid_built
                    .remove(i)
                    .ok_or_else(|| anyhow!("missing child"))?;
                let r = g
                    .kids
                    .get(*i)
                    .map(|x| x.0)
                    .ok_or_else(|| anyhow!("missing role"))?;
                // The modifier word: \P z... (P z...) + r: <child meaning>.
                let child = kb.sem.clone();
                let msem = lifted(cur_cat.arity(), &|a| {
                    Term::Add(Arc::new(a), r, Arc::new(child.clone()))
                })?;
                let mcat = slash(cur_cat.clone(), !right, cur_cat.clone());
                let words = kb.leaves.first().map(|l| l.0.clone()).unwrap_or_default();
                cur_sem = msem
                    .apply(&cur_sem)
                    .ok_or_else(|| anyhow!("modifier does not apply"))?;
                leaves.push((words, mcat, msem));
            }
            Dep::Marked(i) => {
                let kb = kid_built
                    .remove(i)
                    .ok_or_else(|| anyhow!("missing child"))?;
                let (r, m) = g
                    .kids
                    .get(*i)
                    .map(|x| (x.0, x.1.clone()))
                    .ok_or_else(|| anyhow!("missing role"))?;
                let mut mw = m.clone();
                mw.sort_unstable();
                // Marker before or after its object?
                let (ka, _) = span(
                    &g.kids
                        .get(*i)
                        .map(|x| x.2.clone())
                        .ok_or_else(|| anyhow!("missing"))?,
                )
                .unwrap_or((0, 0));
                let marker_first = mw.first().is_some_and(|&w| w < ka);
                // \y P z... (P z...) + r: y
                let k = cur_cat.arity();
                let y = Term::Var(u32::try_from(k + 1)?);
                let msem = lam(lifted(k, &|a| {
                    Term::Add(Arc::new(a), r, Arc::new(y.clone()))
                })?);
                let modcat = slash(cur_cat.clone(), !right, cur_cat.clone());
                let mcat = slash(modcat, marker_first, kb.cat.clone());
                let phrase = msem
                    .apply(&kb.sem)
                    .ok_or_else(|| anyhow!("marker does not apply"))?;
                cur_sem = phrase
                    .apply(&cur_sem)
                    .ok_or_else(|| anyhow!("marked phrase does not apply"))?;
                leaves.push((mw, mcat, msem));
                leaves.extend(kb.leaves);
            }
            Dep::Feat(fs, ws) => {
                let fsem = lifted(cur_cat.arity(), &|a| {
                    let mut body = a;
                    for (f, v) in fs {
                        body = Term::Feat(Arc::new(body), *f, *v);
                    }
                    body
                })?;
                let to_np = Some(n) == last_left_feat && cur_cat == Cat::atom(ATOM_N) && !clause;
                let res = if to_np {
                    Cat::atom(ATOM_NP)
                } else {
                    cur_cat.clone()
                };
                let fcat = slash(res.clone(), !right, cur_cat.clone());
                cur_sem = fsem
                    .apply(&cur_sem)
                    .ok_or_else(|| anyhow!("feature set twice"))?;
                cur_cat = res;
                leaves.push((ws.clone(), fcat, fsem));
            }
            Dep::Loose(w) => {
                let isem = lam(Term::Var(0));
                let icat = slash(cur_cat.clone(), !right, cur_cat.clone());
                leaves.push((vec![*w], icat, isem));
            }
        }
    }
    if cur_cat == Cat::atom(ATOM_N) && !clause {
        bail!("a noun phrase never got its determiner (feature words only on the right)");
    }
    Ok(Built {
        cat: cur_cat,
        sem: cur_sem,
        leaves,
    })
}

/// Convert one validated unit of a line into candidate entries.
pub fn convert(p: &Program, unit: &Unit, ann: &AnnNode, line: &Line) -> Result<Converted> {
    let mut g = to_gnode(p, ann)?;
    tidy(&mut g);
    // A word-less coordination at the root ("no, you can't") is a sequence
    // of fragments: convert each conjunct on its own.
    let op = p.role_by_name("op");
    if g.anchors.is_empty()
        && !g.kids.is_empty()
        && g.kids
            .iter()
            .all(|(r, m, _)| Some(*r) == op && m.is_empty())
    {
        let mut entries = Vec::new();
        for (_, _, k) in &g.kids {
            let Some((a, b)) = span(k) else { continue };
            let sub = Unit {
                start: a,
                end: b + 1,
                ..unit.clone()
            };
            let mut k = k.clone();
            k.loose.clear();
            entries.extend(convert_node(p, &sub, k, line)?.entries);
        }
        let meaning = meaning_of(&g)
            .normalize()
            .ok_or_else(|| anyhow!("ill-formed target meaning"))?;
        return Ok(Converted { meaning, entries });
    }
    convert_node(p, unit, g, line)
}

fn convert_node(p: &Program, unit: &Unit, mut g: GNode, line: &Line) -> Result<Converted> {
    attach_loose(p, &mut g, unit.start, unit.end);
    let core: Vec<u32> = CORE_ROLES
        .iter()
        .filter_map(|r| p.role_by_name(r))
        .collect();
    let meaning = meaning_of(&g)
        .normalize()
        .ok_or_else(|| anyhow!("ill-formed target meaning"))?;
    let b = build(p, &g, &core, true)?;
    let got = b
        .sem
        .normalize()
        .ok_or_else(|| anyhow!("ill-formed composed meaning"))?;
    if meaning_key(&got) != meaning_key(&meaning) {
        bail!("composed meaning differs from the annotation");
    }
    if b.cat.arity() != 0 {
        bail!("unit does not reduce to an atomic category");
    }
    let cn = p.cat_names();
    let mut entries = Vec::new();
    for (ws, cat, sem) in b.leaves {
        let mut forms = Vec::new();
        for w in ws {
            forms.push(
                line.words
                    .get(w)
                    .map(|x| x.form.clone())
                    .ok_or_else(|| anyhow!("word {w} out of range"))?,
            );
        }
        let sem = sem
            .normalize()
            .ok_or_else(|| anyhow!("ill-formed leaf meaning"))?;
        entries.push(Cand {
            form: forms.join(" "),
            cat: cat.show(&cn),
            sem: sem.show(p),
        });
    }
    Ok(Converted { meaning, entries })
}

/// The target meaning of an annotation, as a program term.
pub fn target_meaning(p: &Program, ann: &AnnNode) -> Result<Term> {
    meaning_of(&to_gnode(p, ann)?)
        .normalize()
        .ok_or_else(|| anyhow!("ill-formed meaning"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotate::parse_meaning;
    use crate::ops::{Op, apply};

    fn program() -> Program {
        let mut p = Program::seed();
        let c = |n: &str, parent: &str| Op::AddConcept {
            name: n.into(),
            parent: parent.into(),
            doc: String::new(),
        };
        for op in [
            c("Event", "Thing"),
            c("See", "Event"),
            c("Put", "Event"),
            c("Dog", "Thing"),
            c("Big", "Thing"),
            c("Box", "Thing"),
        ] {
            apply(&mut p, &op, 0).unwrap();
        }
        for r in [
            "experiencer",
            "stimulus",
            "attr",
            "agent",
            "patient",
            "goal",
        ] {
            apply(
                &mut p,
                &Op::AddRole {
                    name: r.into(),
                    domain: "Thing".into(),
                    range: "Thing".into(),
                    doc: String::new(),
                },
                0,
            )
            .unwrap();
        }
        for (f, vs) in [
            ("time", vec!["past", "now"]),
            ("mood", vec!["ask"]),
            ("def", vec!["yes", "no"]),
        ] {
            apply(
                &mut p,
                &Op::AddFeature {
                    name: f.into(),
                    domain: "Thing".into(),
                    values: vs.into_iter().map(String::from).collect(),
                    doc: String::new(),
                },
                0,
            )
            .unwrap();
        }
        p
    }

    fn unit(end: usize, m: &str) -> Unit {
        Unit {
            passage: "p0".into(),
            line: 0,
            start: 0,
            end,
            meaning: m.into(),
        }
    }

    #[test]
    fn a_question_becomes_lexical_entries() {
        let p = program();
        let line = Line::parse("did you see the big dog?");
        let m = "See^2#e1[time=past^0, mood=ask^0](experiencer: @addressee^1, stimulus: Dog^5#e2[def=yes^3](attr: Big^4#e3))";
        let c = convert(&p, &unit(6, m), &parse_meaning(m).unwrap(), &line).unwrap();
        let got: Vec<String> = c
            .entries
            .iter()
            .map(|e| format!("{} := {} : {}", e.form, e.cat, e.sem))
            .collect();
        assert_eq!(
            got,
            vec![
                "see := (S\\NP)/NP : \\x y. See(experiencer: y, stimulus: x)",
                "dog := N : Dog",
                "big := N/N : \\x. x + attr: Big",
                "the := NP/N : \\x. x + def=yes",
                "you := NP : @addressee",
                "did := S/S : \\x. x + time=past + mood=ask",
            ]
        );
    }

    #[test]
    fn a_marked_role_becomes_a_preposition() {
        let p = program();
        let line = Line::parse("she put it in the box");
        let m = "Put^1[time=past](agent: @other^0, patient: @other^2, goal^3: Box^5#e5[def=yes^4])";
        let c = convert(&p, &unit(6, m), &parse_meaning(m).unwrap(), &line).unwrap();
        let got: Vec<String> = c
            .entries
            .iter()
            .map(|e| format!("{} := {}", e.form, e.cat))
            .collect();
        assert_eq!(
            got,
            vec![
                "put := (S\\NP)/NP",
                "it := NP",
                "in := ((S\\NP)\\(S\\NP))/NP",
                "box := N",
                "the := NP/N",
                "she := NP"
            ]
        );
    }
}
