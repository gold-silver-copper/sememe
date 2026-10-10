//! The learner: turns annotations and analyses into candidate changes.

use sememe_engine::program::Program;

use crate::annotate::Reply;
use crate::ops::{Op, apply};

/// Add a teacher reply's proposed types to the program (concepts in an
/// order where parents come first; names already present are kept).
pub fn apply_proposals(p: &mut Program, r: &Reply, round: u32) -> Vec<Op> {
    let mut done = Vec::new();
    let mut pending: Vec<_> = r.new_concepts.iter().collect();
    for _ in 0..16 {
        let mut rest = Vec::new();
        for c in pending {
            if p.concept_by_name(&c.name).is_some() {
                continue;
            }
            if p.concept_by_name(&c.parent).is_none() {
                rest.push(c);
                continue;
            }
            let op = Op::AddConcept {
                name: c.name.clone(),
                parent: c.parent.clone(),
                doc: c.doc.clone(),
            };
            if apply(p, &op, round).is_ok() {
                done.push(op);
            }
        }
        if rest.is_empty() {
            break;
        }
        pending = rest;
    }
    for x in &r.new_roles {
        let op = Op::AddRole {
            name: x.name.clone(),
            domain: x.domain.clone(),
            range: x.range.clone(),
            doc: x.doc.clone(),
        };
        if p.role_by_name(&x.name).is_none() && apply(p, &op, round).is_ok() {
            done.push(op);
        }
    }
    for x in &r.new_features {
        if let Some(fid) = p.feat_by_name(&x.name) {
            // Extend an existing feature with new values.
            let have = p
                .feats
                .get(&fid)
                .map(|f| f.values.clone())
                .unwrap_or_default();
            for v in &x.values {
                if !have.contains(v) {
                    let op = Op::AddValue {
                        feature: x.name.clone(),
                        value: v.clone(),
                    };
                    if apply(p, &op, round).is_ok() {
                        done.push(op);
                    }
                }
            }
            continue;
        }
        let op = Op::AddFeature {
            name: x.name.clone(),
            domain: x.domain.clone(),
            values: x.values.clone(),
            doc: x.doc.clone(),
        };
        if p.feat_by_name(&x.name).is_none() && apply(p, &op, round).is_ok() {
            done.push(op);
        }
    }
    done
}

/// Inflection candidates: endings that relate pairs of entries whose
/// meanings differ only by a set of features on their head ("cat" `N :
/// Cat` and "cats" `N : Cat[num=many]`). A candidate adds the inflection
/// and removes the full forms it makes derivable; it is kept only if the
/// total cost goes down. At least `min_pairs` supporting pairs.
pub fn infl_candidates(p: &Program, min_pairs: usize) -> Vec<Vec<Op>> {
    use sememe_engine::term::TermNames;
    use std::collections::BTreeMap;
    let cn = p.cat_names();
    // Key of an entry with its head features removed.
    let mut by_base: BTreeMap<(String, String), Vec<(u32, String, Vec<(u32, u16)>)>> =
        BTreeMap::new();
    for (id, e) in &p.entries {
        if e.form.contains(' ') {
            continue;
        }
        let feats = e
            .sem
            .leaf_head()
            .map(|n| n.feats.clone())
            .unwrap_or_default();
        let base = e.sem.without_head_feats(&feats);
        by_base
            .entry((e.cat.bare().show(&cn), base.show(p)))
            .or_default()
            .push((*id, e.form.clone(), feats));
    }
    // (category, features, strip, add) -> derived entries it explains.
    let mut rules: BTreeMap<(String, Vec<(u32, u16)>, String, String), Vec<u32>> = BTreeMap::new();
    for ((cat, _), group) in &by_base {
        for (_, stem, sf) in group {
            for (did, form, df) in group {
                if form == stem || !sf.iter().all(|f| df.contains(f)) {
                    continue;
                }
                let extra: Vec<(u32, u16)> =
                    df.iter().filter(|f| !sf.contains(f)).copied().collect();
                if extra.is_empty() {
                    continue;
                }
                // Longest common prefix (in characters).
                let common = stem
                    .chars()
                    .zip(form.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                let strip: String = stem.chars().skip(common).collect();
                let add: String = form.chars().skip(common).collect();
                if strip.chars().count() > 2 || add.chars().count() > 4 || add.is_empty() {
                    continue;
                }
                rules
                    .entry((cat.clone(), extra, strip, add))
                    .or_default()
                    .push(*did);
            }
        }
    }
    let mut out = Vec::new();
    for ((cat, feats, strip, add), mut derived) in rules {
        derived.sort_unstable();
        derived.dedup();
        if derived.len() < min_pairs {
            continue;
        }
        let fs: Vec<String> = feats
            .iter()
            .map(|(f, v)| format!("{}={}", p.feat_name(*f), p.val_name(*f, *v)))
            .collect();
        let mut g = vec![Op::AddInfl {
            cat: cat.clone(),
            feats: fs.join(","),
            strip: strip.clone(),
            add: add.clone(),
            doc: format!(
                "The ending \"{add}\" (replacing \"{strip}\") marks {} on {cat}.",
                fs.join(", ")
            ),
        }];
        g.extend(derived.into_iter().map(|id| Op::RemoveEntry { id }));
        out.push(g);
    }
    out
}
