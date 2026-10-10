//! Typed operations on the program. The owner applies every change through
//! these, and logs each one, so replaying the log rebuilds the program.

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use sememe_engine::category::Cat;
use sememe_engine::program::{
    AtomDecl, CatFeat, Concept, Entity, Entry, Fact, Feature, Infl, Meta, NameEntry, Program, Role,
    Rule, RuleKind, TOP,
};
use sememe_engine::term::{Head, Node, Term, TermLookup};
use serde::{Deserialize, Serialize};

/// One change to the program. Items are referred to by name (concepts,
/// roles, features) or stable ID (entries, rules).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// A new concept.
    AddConcept {
        /// Name.
        name: String,
        /// Supertype name.
        parent: String,
        /// Doc comment.
        doc: String,
    },
    /// A new role.
    AddRole {
        /// Name.
        name: String,
        /// Domain concept.
        domain: String,
        /// Range concept.
        range: String,
        /// Doc comment.
        doc: String,
    },
    /// A new meaning feature.
    AddFeature {
        /// Name.
        name: String,
        /// Domain concept.
        domain: String,
        /// Values.
        values: Vec<String>,
        /// Doc comment.
        doc: String,
    },
    /// A new value of an existing meaning feature.
    AddValue {
        /// Feature name.
        feature: String,
        /// Value name.
        value: String,
    },
    /// A new atomic category.
    AddAtom {
        /// Name.
        name: String,
        /// Doc comment.
        doc: String,
    },
    /// A new category feature dimension.
    AddCatFeature {
        /// Name.
        name: String,
        /// Atoms it applies to.
        atoms: Vec<String>,
        /// Values.
        values: Vec<String>,
        /// Doc comment.
        doc: String,
    },
    /// A combining rule, in the grammar file's text form (`fcomp`, `tc N NP`).
    AddRule {
        /// Rule text.
        rule: String,
        /// Doc comment.
        doc: String,
    },
    /// Remove a rule.
    RemoveRule {
        /// Rule ID.
        id: u32,
    },
    /// A lexical entry.
    AddEntry {
        /// Word form.
        form: String,
        /// Category text.
        cat: String,
        /// Meaning text.
        sem: String,
        /// Doc comment.
        doc: String,
    },
    /// Remove an entry.
    RemoveEntry {
        /// Entry ID.
        id: u32,
    },
    /// A proper-name entry.
    AddName {
        /// Category text.
        cat: String,
        /// Concept name.
        concept: String,
        /// Doc comment.
        doc: String,
    },
    /// An inflection.
    AddInfl {
        /// Category text.
        cat: String,
        /// Features `f=v,...`.
        feats: String,
        /// Removed from the stem.
        strip: String,
        /// Added.
        add: String,
        /// Doc comment.
        doc: String,
    },
    /// Remove an inflection.
    RemoveInfl {
        /// Inflection ID.
        id: u32,
    },
    /// A global named individual.
    AddEntity {
        /// Name.
        name: String,
        /// Concept name.
        concept: String,
        /// Doc comment.
        doc: String,
    },
    /// A global fact.
    AddFact {
        /// Meaning text.
        term: String,
        /// Doc comment.
        doc: String,
    },
    /// Merge one concept into another: every use of `from` becomes `into`.
    MergeConcepts {
        /// Concept that disappears.
        from: String,
        /// Concept that stays.
        into: String,
    },
    /// Rename a concept, role or feature.
    Rename {
        /// `concept`, `role` or `feature`.
        kind: String,
        /// Old name.
        from: String,
        /// New name.
        to: String,
    },
    /// Move a concept in the hierarchy.
    SetParent {
        /// Concept.
        concept: String,
        /// New supertype.
        parent: String,
    },
    /// Set the doc comment of an item (`concept`, `role`, `feature` by name;
    /// `entry`, `rule` by ID).
    SetDoc {
        /// Item kind.
        kind: String,
        /// Name or ID.
        item: String,
        /// Doc comment.
        doc: String,
    },
    /// Remove a concept that nothing uses.
    RemoveConcept {
        /// Concept name.
        name: String,
    },
}

/// A logged operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logged {
    /// Round it was applied in.
    pub round: u32,
    /// Who proposed it (`teacher`, `learner`, `context`, `agent:<name>`, ...).
    pub source: String,
    /// The operation.
    pub op: Op,
}

fn meta(round: u32, doc: &str) -> Meta {
    Meta {
        round,
        doc: doc.to_string(),
        seed: false,
    }
}

fn concept(p: &Program, name: &str) -> Result<u32> {
    p.concept_by_name(name)
        .ok_or_else(|| anyhow!("unknown concept `{name}`"))
}

/// Parse a rule's text form.
pub fn parse_rule(p: &Program, text: &str) -> Result<RuleKind> {
    let cn = p.cat_names();
    let mut w = text.split_whitespace();
    let k = w.next().unwrap_or_default();
    let mut cat = || -> Result<Cat> {
        Cat::parse(
            w.next()
                .ok_or_else(|| anyhow!("rule `{text}` needs a category"))?,
            &cn,
        )
        .map_err(|e| anyhow!(e))
    };
    Ok(match k {
        "fapp" => RuleKind::FApp,
        "bapp" => RuleKind::BApp,
        "fcomp" => RuleKind::FComp,
        "bcomp" => RuleKind::BComp,
        "fxcomp" => RuleKind::FXComp,
        "bxcomp" => RuleKind::BXComp,
        "ftr" => RuleKind::FTr(cat()?, cat()?),
        "btr" => RuleKind::BTr(cat()?, cat()?),
        "tc" => RuleKind::Tc(cat()?, cat()?),
        "coord" => RuleKind::Coord(cat()?),
        other => bail!("unknown rule kind `{other}`"),
    })
}

/// Apply one operation. Returns the ID of a created item, if any.
pub fn apply(p: &mut Program, op: &Op, round: u32) -> Result<Option<u32>> {
    let mut created = None;
    match op {
        Op::AddConcept { name, parent, doc } => {
            if p.concept_by_name(name).is_some() {
                return Ok(None);
            }
            let parent = concept(p, parent)?;
            let id = p.alloc("concept");
            p.concepts.insert(
                id,
                Concept {
                    name: name.clone(),
                    parent,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::AddRole {
            name,
            domain,
            range,
            doc,
        } => {
            if p.role_by_name(name).is_some() {
                return Ok(None);
            }
            let (d, r) = (concept(p, domain)?, concept(p, range)?);
            let id = p.alloc("role");
            p.roles.insert(
                id,
                Role {
                    name: name.clone(),
                    domain: d,
                    range: r,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::AddFeature {
            name,
            domain,
            values,
            doc,
        } => {
            if p.feat_by_name(name).is_some() {
                return Ok(None);
            }
            let d = concept(p, domain)?;
            let id = p.alloc("feature");
            p.feats.insert(
                id,
                Feature {
                    name: name.clone(),
                    domain: d,
                    values: values.clone(),
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::AddValue { feature, value } => {
            let id = p
                .feat_by_name(feature)
                .ok_or_else(|| anyhow!("unknown feature `{feature}`"))?;
            if let Some(f) = p.feats.get_mut(&id)
                && !f.values.contains(value)
            {
                f.values.push(value.clone());
            }
        }
        Op::AddAtom { name, doc } => {
            if p.atoms.iter().any(|a| a.name == *name) {
                return Ok(None);
            }
            p.atoms.push(AtomDecl {
                name: name.clone(),
                meta: meta(round, doc),
            });
            created = u32::try_from(p.atoms.len() - 1).ok();
        }
        Op::AddCatFeature {
            name,
            atoms,
            values,
            doc,
        } => {
            let ids = atoms
                .iter()
                .map(|a| {
                    p.atoms
                        .iter()
                        .position(|x| x.name == *a)
                        .and_then(|i| u16::try_from(i).ok())
                        .ok_or_else(|| anyhow!("unknown atom `{a}`"))
                })
                .collect::<Result<Vec<_>>>()?;
            p.cat_feats.push(CatFeat {
                name: name.clone(),
                values: values.clone(),
                atoms: ids,
                meta: meta(round, doc),
            });
        }
        Op::AddRule { rule, doc } => {
            let kind = parse_rule(p, rule)?;
            if p.rules.values().any(|r| r.kind == kind) {
                return Ok(None);
            }
            let id = p.alloc("rule");
            p.rules.insert(
                id,
                Rule {
                    kind,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::RemoveRule { id } => {
            p.rules.remove(id);
        }
        Op::AddEntry {
            form,
            cat,
            sem,
            doc,
        } => {
            let cat = Cat::parse(cat, &p.cat_names()).map_err(|e| anyhow!(e))?;
            let sem = Term::parse(sem, p).map_err(|e| anyhow!(e))?;
            let sem = sem
                .normalize()
                .ok_or_else(|| anyhow!("ill-formed meaning"))?;
            if sem.lam_depth() != cat.arity() {
                bail!(
                    "meaning takes {} arguments but category {} takes {}",
                    sem.lam_depth(),
                    cat.show(&p.cat_names()),
                    cat.arity()
                );
            }
            if p.entries
                .values()
                .any(|e| e.form == *form && e.cat == cat && e.sem == sem)
            {
                return Ok(None);
            }
            let id = p.alloc("entry");
            p.entries.insert(
                id,
                Entry {
                    form: form.clone(),
                    cat,
                    sem,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::RemoveEntry { id } => {
            p.entries.remove(id);
        }
        Op::AddName {
            cat,
            concept: c,
            doc,
        } => {
            let cat = Cat::parse(cat, &p.cat_names()).map_err(|e| anyhow!(e))?;
            let c = concept(p, c)?;
            if p.names.values().any(|n| n.cat == cat && n.concept == c) {
                return Ok(None);
            }
            let id = p.alloc("name");
            p.names.insert(
                id,
                NameEntry {
                    cat,
                    concept: c,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::AddInfl {
            cat,
            feats,
            strip,
            add,
            doc,
        } => {
            let cat = Cat::parse(cat, &p.cat_names()).map_err(|e| anyhow!(e))?;
            let mut fv = Vec::new();
            for pair in feats.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (f, v) = pair
                    .split_once('=')
                    .ok_or_else(|| anyhow!("bad feature `{pair}`"))?;
                let fid = p
                    .feat_by_name(f)
                    .ok_or_else(|| anyhow!("unknown feature `{f}`"))?;
                let vid = p
                    .val_id(fid, v)
                    .ok_or_else(|| anyhow!("unknown value `{v}`"))?;
                fv.push((fid, vid));
            }
            fv.sort();
            let id = p.alloc("infl");
            p.infls.insert(
                id,
                Infl {
                    cat,
                    feats: fv,
                    strip: strip.clone(),
                    add: add.clone(),
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::RemoveInfl { id } => {
            p.infls.remove(id);
        }
        Op::AddEntity {
            name,
            concept: c,
            doc,
        } => {
            if p.entity_by_name(name).is_some() {
                return Ok(None);
            }
            let c = concept(p, c)?;
            let id = p.alloc("entity");
            p.entities.insert(
                id,
                Entity {
                    name: name.clone(),
                    concept: c,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::AddFact { term, doc } => {
            let t = Term::parse(term, p).map_err(|e| anyhow!(e))?;
            if !t.is_tree() {
                bail!("a fact must be a closed meaning tree");
            }
            let id = p.alloc("fact");
            p.facts.insert(
                id,
                Fact {
                    term: t,
                    meta: meta(round, doc),
                },
            );
            created = Some(id);
        }
        Op::MergeConcepts { from, into } => {
            let (a, b) = (concept(p, from)?, concept(p, into)?);
            if a == b || a == TOP {
                bail!("cannot merge `{from}` into `{into}`");
            }
            merge_concepts(p, a, b);
        }
        Op::Rename { kind, from, to } => match kind.as_str() {
            "concept" => {
                let id = concept(p, from)?;
                if p.concept_by_name(to).is_some() {
                    bail!("`{to}` already exists");
                }
                if let Some(c) = p.concepts.get_mut(&id) {
                    c.name = to.clone();
                }
            }
            "role" => {
                let id = p
                    .role_by_name(from)
                    .ok_or_else(|| anyhow!("unknown role `{from}`"))?;
                if let Some(c) = p.roles.get_mut(&id) {
                    c.name = to.clone();
                }
            }
            "feature" => {
                let id = p
                    .feat_by_name(from)
                    .ok_or_else(|| anyhow!("unknown feature `{from}`"))?;
                if let Some(c) = p.feats.get_mut(&id) {
                    c.name = to.clone();
                }
            }
            other => bail!("cannot rename a `{other}`"),
        },
        Op::SetParent { concept: c, parent } => {
            let (a, b) = (concept(p, c)?, concept(p, parent)?);
            if p.is_a(b, a) {
                bail!("`{parent}` is below `{c}`: that would make a cycle");
            }
            if let Some(x) = p.concepts.get_mut(&a) {
                x.parent = b;
            }
        }
        Op::SetDoc { kind, item, doc } => {
            let m = match kind.as_str() {
                "concept" => {
                    let id = concept(p, item)?;
                    p.concepts.get_mut(&id).map(|x| &mut x.meta)
                }
                "role" => {
                    let id = p
                        .role_by_name(item)
                        .ok_or_else(|| anyhow!("unknown role `{item}`"))?;
                    p.roles.get_mut(&id).map(|x| &mut x.meta)
                }
                "feature" => {
                    let id = p
                        .feat_by_name(item)
                        .ok_or_else(|| anyhow!("unknown feature `{item}`"))?;
                    p.feats.get_mut(&id).map(|x| &mut x.meta)
                }
                "entry" => {
                    let id: u32 = item.trim_start_matches('e').parse()?;
                    p.entries.get_mut(&id).map(|x| &mut x.meta)
                }
                "rule" => {
                    let id: u32 = item.trim_start_matches('g').parse()?;
                    p.rules.get_mut(&id).map(|x| &mut x.meta)
                }
                other => bail!("cannot document a `{other}`"),
            };
            if let Some(m) = m {
                m.doc = doc.clone();
            }
        }
        Op::RemoveConcept { name } => {
            let id = concept(p, name)?;
            if concept_used(p, id) {
                bail!("`{name}` is still used");
            }
            p.concepts.remove(&id);
        }
    }
    p.reindex();
    Ok(created)
}

/// Rewrite every concept ID in a term.
pub fn map_concepts(t: &Term, f: &dyn Fn(u32) -> u32) -> Term {
    match t {
        Term::Var(_) => t.clone(),
        Term::Lam(b) => Term::Lam(Arc::new(map_concepts(b, f))),
        Term::App(a, b) => Term::App(Arc::new(map_concepts(a, f)), Arc::new(map_concepts(b, f))),
        Term::Node(n) => {
            let mut m: Node = (**n).clone();
            if let Head::Concept(c) = m.head {
                m.head = Head::Concept(f(c));
            }
            for (_, c) in &mut m.roles {
                *c = map_concepts(c, f);
            }
            Term::node(m)
        }
        Term::Add(a, r, b) => Term::Add(
            Arc::new(map_concepts(a, f)),
            *r,
            Arc::new(map_concepts(b, f)),
        ),
        Term::Feat(a, x, v) => Term::Feat(Arc::new(map_concepts(a, f)), *x, *v),
    }
}

fn mentions_concept(t: &Term, c: u32) -> bool {
    let mut found = false;
    t.for_each_node(&mut |n| {
        if n.head == Head::Concept(c) {
            found = true;
        }
    });
    found
}

/// Is a concept referred to by anything?
pub fn concept_used(p: &Program, c: u32) -> bool {
    p.concepts.values().any(|x| x.parent == c)
        || p.roles.values().any(|r| r.domain == c || r.range == c)
        || p.feats.values().any(|f| f.domain == c)
        || p.entries.values().any(|e| mentions_concept(&e.sem, c))
        || p.names.values().any(|n| n.concept == c)
        || p.entities.values().any(|e| e.concept == c)
        || p.facts.values().any(|f| mentions_concept(&f.term, c))
        || p.inferences
            .values()
            .any(|f| mentions_concept(&f.premise, c) || mentions_concept(&f.conclusion, c))
}

fn merge_concepts(p: &mut Program, a: u32, b: u32) {
    let f = |c: u32| if c == a { b } else { c };
    for x in p.concepts.values_mut() {
        x.parent = f(x.parent);
    }
    for r in p.roles.values_mut() {
        r.domain = f(r.domain);
        r.range = f(r.range);
    }
    for x in p.feats.values_mut() {
        x.domain = f(x.domain);
    }
    for e in p.entries.values_mut() {
        e.sem = map_concepts(&e.sem, &f);
    }
    for n in p.names.values_mut() {
        n.concept = f(n.concept);
    }
    for e in p.entities.values_mut() {
        e.concept = f(e.concept);
    }
    for x in p.facts.values_mut() {
        x.term = map_concepts(&x.term, &f);
    }
    p.concepts.remove(&a);
    // Merging can make two entries identical: keep the older one.
    let mut seen: Vec<(String, Cat, Term)> = Vec::new();
    let mut dup = Vec::new();
    for (id, e) in &p.entries {
        let k = (e.form.clone(), e.cat.clone(), e.sem.clone());
        if seen.contains(&k) {
            dup.push(*id);
        } else {
            seen.push(k);
        }
    }
    for id in dup {
        p.entries.remove(&id);
    }
}

/// Append operations to a log file.
pub fn log(path: &std::path::Path, ops: &[Logged]) -> Result<()> {
    use std::io::Write as _;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for o in ops {
        writeln!(f, "{}", serde_json::to_string(o)?)?;
    }
    Ok(())
}

/// Read an operations log.
pub fn read_log(path: &std::path::Path) -> Result<Vec<Logged>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Ok(Vec::new()),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow!("bad log line: {e}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaying_the_log_rebuilds_the_program() {
        let ops = vec![
            Op::AddConcept {
                name: "Animal".into(),
                parent: "Thing".into(),
                doc: "A living creature.".into(),
            },
            Op::AddConcept {
                name: "Cat".into(),
                parent: "Animal".into(),
                doc: "A small feline.".into(),
            },
            Op::AddConcept {
                name: "Kitty".into(),
                parent: "Thing".into(),
                doc: "A cat.".into(),
            },
            Op::AddRole {
                name: "attr".into(),
                domain: "Thing".into(),
                range: "Thing".into(),
                doc: "A property.".into(),
            },
            Op::AddEntry {
                form: "cat".into(),
                cat: "N".into(),
                sem: "Cat".into(),
                doc: String::new(),
            },
            Op::AddEntry {
                form: "kitty".into(),
                cat: "N".into(),
                sem: "Kitty".into(),
                doc: String::new(),
            },
            Op::MergeConcepts {
                from: "Kitty".into(),
                into: "Cat".into(),
            },
            Op::Rename {
                kind: "concept".into(),
                from: "Cat".into(),
                to: "Feline".into(),
            },
            Op::AddRule {
                rule: "fcomp".into(),
                doc: String::new(),
            },
        ];
        let build = || {
            let mut p = Program::seed();
            for op in &ops {
                apply(&mut p, op, 1).unwrap();
            }
            p
        };
        let (a, b) = (build(), build());
        assert_eq!(a.to_files(), b.to_files());
        // The text form reads back to the same program.
        let c = Program::from_files(&a.to_files()).unwrap();
        assert_eq!(c.to_files(), a.to_files());
        assert!(a.concept_by_name("Kitty").is_none());
        assert_eq!(a.entries.len(), 2);
        assert!(a.entries.values().all(|e| matches!(e.sem.as_node().map(|n| &n.head), Some(Head::Concept(c)) if p_name(&a, *c) == "Feline")));
    }

    fn p_name(p: &Program, c: u32) -> String {
        use sememe_engine::term::TermNames;
        p.concept_name(c)
    }
}
