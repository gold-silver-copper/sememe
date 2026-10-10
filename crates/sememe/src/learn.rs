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
