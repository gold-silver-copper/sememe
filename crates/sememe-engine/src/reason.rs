//! Inside mode: answer a question from the facts in the context, the global
//! knowledge base, and what one step of the inference rules adds.
//!
//! A question is a meaning with a `Query` node ("who went to the market?" =
//! `Go(agent: Query, goal: Market)`) or, for yes/no questions, without one.
//! It is matched against facts: heads must agree, every role of the
//! question must be matched by a role of the fact, and the `Query` node
//! binds the fact's subtree there: the answer.

use crate::context::Context;
use crate::cost::INFERENCE_DEPTH;
use crate::program::Program;
use crate::term::{Deixis, Head, Node, Term};

/// The answer to a question.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// The thing asked for, and the fact it came from.
    Thing(Term, Term),
    /// A yes/no question that a fact confirms.
    Yes(Term),
    /// A yes/no question that a fact denies (same fact, opposite polarity).
    No(Term),
    /// Nothing known answers it.
    Unknown,
}

fn polarity_feat(p: &Program) -> Option<u32> {
    p.feat_by_name("polarity")
}

/// Does question node `q` match fact node `f`? Binds the `Query` subtree.
/// `ignore` lists features not compared (mood, tense, ...).
fn match_node(q: &Node, f: &Node, ignore: &[u32], bind: &mut Option<Term>) -> bool {
    match &q.head {
        Head::Query => {
            *bind = Some(Term::node(f.clone()));
            return true;
        }
        Head::Ref(Deixis::Other) => {}
        h => {
            if *h != f.head {
                return false;
            }
        }
    }
    if q.name.is_some() && q.name != f.name {
        return false;
    }
    for fv in &q.feats {
        if ignore.contains(&fv.0) {
            continue;
        }
        if !f.feats.contains(fv) {
            return false;
        }
    }
    let mut used = vec![false; f.roles.len()];
    for (r, qc) in &q.roles {
        let Some(qn) = qc.as_node() else { return false };
        let mut ok = false;
        for (i, (fr, fc)) in f.roles.iter().enumerate() {
            if fr != r || used.get(i).copied().unwrap_or(true) {
                continue;
            }
            let Some(fnode) = fc.as_node() else { continue };
            let mut b = bind.clone();
            if match_node(qn, fnode, ignore, &mut b) {
                *bind = b;
                if let Some(u) = used.get_mut(i) {
                    *u = true;
                }
                ok = true;
                break;
            }
        }
        if !ok {
            return false;
        }
    }
    true
}

/// Match a rule pattern (a term under `k` lambdas, the rule's variables)
/// against a fact, binding variables to subtrees.
fn match_pattern(pat: &Term, f: &Term, binds: &mut Vec<Option<Term>>, depth: u32) -> bool {
    match pat {
        Term::Var(i) => {
            let Some(slot) = depth
                .checked_sub(1 + i)
                .and_then(|k| binds.get_mut(k as usize))
            else {
                return false;
            };
            match slot {
                Some(b) => b == f,
                None => {
                    *slot = Some(f.clone());
                    true
                }
            }
        }
        Term::Node(pn) => {
            let Some(fnode) = f.as_node() else {
                return false;
            };
            if pn.head != fnode.head || !pn.feats.iter().all(|x| fnode.feats.contains(x)) {
                return false;
            }
            pn.roles.iter().all(|(r, pc)| {
                fnode
                    .roles
                    .iter()
                    .any(|(fr, fc)| fr == r && match_pattern(pc, fc, binds, depth))
            })
        }
        _ => false,
    }
}

fn instantiate(t: &Term, binds: &[Option<Term>], depth: u32) -> Option<Term> {
    match t {
        Term::Var(i) => depth
            .checked_sub(1 + i)
            .and_then(|k| binds.get(k as usize))
            .and_then(Clone::clone),
        Term::Node(n) => {
            let mut m = (**n).clone();
            for (_, c) in &mut m.roles {
                *c = instantiate(c, binds, depth)?;
            }
            Some(Term::node(m))
        }
        _ => None,
    }
}

/// Facts that follow from `facts` by one application of each rule.
pub fn infer(p: &Program, facts: &[Term]) -> Vec<Term> {
    let mut out = Vec::new();
    let mut frontier: Vec<Term> = facts.to_vec();
    for _ in 0..INFERENCE_DEPTH {
        let mut next = Vec::new();
        for rule in p.inferences.values() {
            let k = rule.premise.lam_depth();
            let (pb, cb) = (rule.premise.body(), rule.conclusion.body());
            for f in &frontier {
                let mut binds = vec![None; k];
                if match_pattern(pb, f, &mut binds, k as u32)
                    && let Some(c) = instantiate(cb, &binds, k as u32)
                    && !out.contains(&c)
                {
                    next.push(c.clone());
                    out.push(c);
                }
            }
        }
        frontier = next;
    }
    out
}

/// Answer a question meaning from the context and the knowledge base.
pub fn answer(p: &Program, ctx: &Context, question: &Term) -> Answer {
    let Some(q) = question.as_node() else {
        return Answer::Unknown;
    };
    let ignore: Vec<u32> = ["mood", "time", "aspect", "def", "polarity"]
        .iter()
        .filter_map(|n| p.feat_by_name(n))
        .collect();
    let mut facts: Vec<Term> = ctx.facts.iter().map(|f| f.meaning.clone()).collect();
    facts.extend(p.facts.values().map(|f| f.term.clone()));
    let derived = infer(p, &facts);
    facts.extend(derived);
    let has_query = {
        let mut found = false;
        question.for_each_node(&mut |n| found |= n.head == Head::Query);
        found
    };
    let pol = polarity_feat(p);
    let negated = |n: &Node| pol.is_some_and(|pf| n.feats.iter().any(|(f, _)| *f == pf));
    for f in &facts {
        let Some(fnode) = f.as_node() else { continue };
        let mut bind = None;
        if match_node(q, fnode, &ignore, &mut bind) {
            if has_query {
                if let Some(b) = bind {
                    return Answer::Thing(b, f.clone());
                }
            } else if negated(q) == negated(fnode) {
                return Answer::Yes(f.clone());
            } else {
                return Answer::No(f.clone());
            }
        }
    }
    Answer::Unknown
}
