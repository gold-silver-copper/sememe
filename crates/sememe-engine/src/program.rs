//! The learned program: ontology, grammar, lexicon, knowledge and counts.
//!
//! The source of truth is a canonical text form, one item per line, sorted
//! by stable ID, each with its doc comment and the round that added it.
//! Structure lives in `*.sem` files and counts in `counts.tsv`, so a counting
//! pass does not rewrite every line.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::category::{Cat, CatNames};
use crate::term::{Head, Term, TermLookup, TermNames};

/// The top of the type hierarchy, `Thing`.
pub const TOP: u32 = 0;
/// The seed concept for conversation participants introduced by speaker tags.
pub const SPEAKER: u32 = 1;
/// Seed role for the arguments of unknown functor words.
pub const ROLE_ARG: u32 = 0;
/// Seed role for what unknown modifier words add.
pub const ROLE_MOD: u32 = 1;
/// Seed role for the parts of a coordination.
pub const ROLE_OP: u32 = 2;
/// Seed atomic categories.
pub const ATOM_S: u16 = 0;
/// Seed atomic category NP.
pub const ATOM_NP: u16 = 1;
/// Seed atomic category N.
pub const ATOM_N: u16 = 2;

/// Bookkeeping shared by every item.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    /// The round that added the item.
    pub round: u32,
    /// Doc comment (free in the size measure).
    pub doc: String,
    /// Hand-written seed item.
    pub seed: bool,
}

/// A concept (type) in the ontology.
#[derive(Clone, Debug, PartialEq)]
pub struct Concept {
    /// Readable name, unique.
    pub name: String,
    /// Supertype (`TOP` for top-level concepts).
    pub parent: u32,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A role (relation between a node and a child).
#[derive(Clone, Debug, PartialEq)]
pub struct Role {
    /// Readable name, unique.
    pub name: String,
    /// The type of node that can have this role.
    pub domain: u32,
    /// The type of child that fills it.
    pub range: u32,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A feature of meaning nodes, such as `time` or `def`.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    /// Readable name, unique.
    pub name: String,
    /// The type of node that has it.
    pub domain: u32,
    /// Value names.
    pub values: Vec<String>,
    /// Bookkeeping.
    pub meta: Meta,
}

/// An atomic syntactic category.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomDecl {
    /// Name such as `S`.
    pub name: String,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A feature dimension on atomic categories, such as `num` on `NP`.
#[derive(Clone, Debug, PartialEq)]
pub struct CatFeat {
    /// Dimension name.
    pub name: String,
    /// Value names.
    pub values: Vec<String>,
    /// Atomic categories it applies to.
    pub atoms: Vec<u16>,
    /// Bookkeeping.
    pub meta: Meta,
}

/// The kinds of combining rule the engine knows.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RuleKind {
    /// `X/Y Y => X`
    FApp,
    /// `Y X\Y => X`
    BApp,
    /// `X/Y Y/Z => X/Z`
    FComp,
    /// `Y\Z X\Y => X\Z`
    BComp,
    /// `X/Y Y\Z => X\Z`
    FXComp,
    /// `Y/Z X\Y => X/Z`
    BXComp,
    /// Forward type raising `X => T/(T\X)`.
    FTr(Cat, Cat),
    /// Backward type raising `X => T\(T/X)`.
    BTr(Cat, Cat),
    /// Type change `X => Y` with the meaning unchanged.
    Tc(Cat, Cat),
    /// Coordination `X CONJ X => X`, with the given conjunction category.
    Coord(Cat),
}

impl RuleKind {
    /// A small code for the kind (used in count contexts).
    pub fn code(&self) -> u32 {
        match self {
            RuleKind::FApp => 0,
            RuleKind::BApp => 1,
            RuleKind::FComp => 2,
            RuleKind::BComp => 3,
            RuleKind::FXComp => 4,
            RuleKind::BXComp => 5,
            RuleKind::FTr(..) => 6,
            RuleKind::BTr(..) => 7,
            RuleKind::Tc(..) => 8,
            RuleKind::Coord(_) => 9,
        }
    }

    /// Number of rule kinds the engine knows.
    pub const KINDS: u32 = 10;

    /// The categories this rule instance mentions.
    pub fn cats(&self) -> Vec<&Cat> {
        match self {
            RuleKind::FTr(a, b) | RuleKind::BTr(a, b) | RuleKind::Tc(a, b) => vec![a, b],
            RuleKind::Coord(c) => vec![c],
            _ => Vec::new(),
        }
    }
}

/// A combining rule in use.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    /// What the rule does.
    pub kind: RuleKind,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A lexical entry: a word form with a category and a meaning.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Lowercased word form (or stem, when inflections apply).
    pub form: String,
    /// Syntactic category.
    pub cat: Cat,
    /// What the word contributes.
    pub sem: Term,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A generic proper-name entry: any word may name an instance of `concept`.
#[derive(Clone, Debug, PartialEq)]
pub struct NameEntry {
    /// Category of the name (usually `NP`).
    pub cat: Cat,
    /// What kind of thing it names.
    pub concept: u32,
    /// Bookkeeping.
    pub meta: Meta,
}

/// An inflection: a word ending that expresses meaning features.
#[derive(Clone, Debug, PartialEq)]
pub struct Infl {
    /// The category of the words it applies to (features ignored).
    pub cat: Cat,
    /// The meaning features the ending expresses.
    pub feats: Vec<(u32, u16)>,
    /// Text removed from the end of the stem.
    pub strip: String,
    /// Text added.
    pub add: String,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A named individual in the global knowledge base.
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    /// Readable name (also its lowercased surface name).
    pub name: String,
    /// What it is.
    pub concept: u32,
    /// Bookkeeping.
    pub meta: Meta,
}

/// A global fact.
#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
    /// The fact as a closed meaning tree.
    pub term: Term,
    /// Bookkeeping.
    pub meta: Meta,
}

/// An inference rule: if a fact matches `premise`, then `conclusion` holds.
/// Both are terms under the same lambdas; the lambdas are the rule's
/// variables, matched against subtrees of the premise fact.
#[derive(Clone, Debug, PartialEq)]
pub struct Inference {
    /// Pattern the known fact must match (a node pattern under lambdas).
    pub premise: Term,
    /// What follows (under the same lambdas).
    pub conclusion: Term,
    /// Bookkeeping.
    pub meta: Meta,
}

/// Stored counts for one context of one table: quantized levels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CtxCounts {
    /// Level of the escape count.
    pub esc: u32,
    /// Level of each stored outcome.
    pub outs: BTreeMap<u64, u32>,
}

/// All stored counts, keyed by `(table, context)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counts {
    /// The counts.
    pub tables: BTreeMap<(u8, [u64; 2]), CtxCounts>,
    /// Printable forms of hashed keys (categories, meanings, strings).
    pub syms: BTreeMap<u64, String>,
}

/// The whole learned program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Concepts by stable ID.
    pub concepts: BTreeMap<u32, Concept>,
    /// Roles by stable ID.
    pub roles: BTreeMap<u32, Role>,
    /// Meaning features by stable ID.
    pub feats: BTreeMap<u32, Feature>,
    /// Atomic categories (ID = index).
    pub atoms: Vec<AtomDecl>,
    /// Category feature dimensions (ID = index).
    pub cat_feats: Vec<CatFeat>,
    /// Combining rules in use.
    pub rules: BTreeMap<u32, Rule>,
    /// Lexical entries.
    pub entries: BTreeMap<u32, Entry>,
    /// Generic proper-name entries.
    pub names: BTreeMap<u32, NameEntry>,
    /// Inflections.
    pub infls: BTreeMap<u32, Infl>,
    /// Global named individuals.
    pub entities: BTreeMap<u32, Entity>,
    /// Global facts.
    pub facts: BTreeMap<u32, Fact>,
    /// Inference rules.
    pub inferences: BTreeMap<u32, Inference>,
    /// Stored counts.
    pub counts: Counts,
    /// Next free ID per item kind.
    pub next: BTreeMap<String, u32>,
    index: NameIndex,
}

#[derive(Clone, Debug, Default)]
struct NameIndex {
    concepts: HashMap<String, u32>,
    roles: HashMap<String, u32>,
    feats: HashMap<String, u32>,
    entities: HashMap<String, u32>,
}

/// Item kinds that have their own ID space.
pub const KINDS: [&str; 10] = [
    "concept", "role", "feature", "rule", "entry", "name", "infl", "entity", "fact", "infer",
];

fn seed_meta(doc: &str) -> Meta {
    Meta {
        round: 0,
        doc: doc.to_string(),
        seed: true,
    }
}

impl Program {
    /// The seed program: the engine's minimal starting point. An empty
    /// lexicon and an ontology holding only `Thing` and `Speaker`.
    pub fn seed() -> Program {
        let mut p = Program::default();
        p.concepts.insert(
            TOP,
            Concept {
                name: "Thing".into(),
                parent: TOP,
                meta: seed_meta("Anything at all: the top of the type hierarchy."),
            },
        );
        p.concepts.insert(
            SPEAKER,
            Concept {
                name: "Speaker".into(),
                parent: TOP,
                meta: seed_meta(
                    "A participant in a conversation, introduced by a speaker tag or as the narrator of written text.",
                ),
            },
        );
        for (id, name, doc) in [
            (
                ROLE_ARG,
                "arg",
                "An argument of a word whose meaning is not known yet.",
            ),
            (ROLE_MOD, "mod", "Something an unknown modifier word adds."),
            (ROLE_OP, "op", "One of the things joined by a coordination."),
        ] {
            p.roles.insert(
                id,
                Role {
                    name: name.into(),
                    domain: TOP,
                    range: TOP,
                    meta: seed_meta(doc),
                },
            );
        }
        for (name, doc) in [
            ("S", "A sentence or clause."),
            ("NP", "A noun phrase: something that refers."),
            (
                "N",
                "A noun: a description that still needs a determiner to refer.",
            ),
        ] {
            p.atoms.push(AtomDecl {
                name: name.into(),
                meta: seed_meta(doc),
            });
        }
        p.rules.insert(
            0,
            Rule {
                kind: RuleKind::FApp,
                meta: seed_meta("Forward application: X/Y Y => X."),
            },
        );
        p.rules.insert(
            1,
            Rule {
                kind: RuleKind::BApp,
                meta: seed_meta("Backward application: Y X\\Y => X."),
            },
        );
        p.reindex();
        p.fix_next();
        p
    }

    /// Rebuild the name indexes.
    pub fn reindex(&mut self) {
        self.index = NameIndex {
            concepts: self
                .concepts
                .iter()
                .map(|(k, v)| (v.name.clone(), *k))
                .collect(),
            roles: self
                .roles
                .iter()
                .map(|(k, v)| (v.name.clone(), *k))
                .collect(),
            feats: self
                .feats
                .iter()
                .map(|(k, v)| (v.name.clone(), *k))
                .collect(),
            entities: self
                .entities
                .iter()
                .map(|(k, v)| (v.name.clone(), *k))
                .collect(),
        };
    }

    /// Make sure every `next` counter is past every used ID.
    pub fn fix_next(&mut self) {
        let maxes: [(&str, Option<u32>); 10] = [
            ("concept", self.concepts.keys().next_back().copied()),
            ("role", self.roles.keys().next_back().copied()),
            ("feature", self.feats.keys().next_back().copied()),
            ("rule", self.rules.keys().next_back().copied()),
            ("entry", self.entries.keys().next_back().copied()),
            ("name", self.names.keys().next_back().copied()),
            ("infl", self.infls.keys().next_back().copied()),
            ("entity", self.entities.keys().next_back().copied()),
            ("fact", self.facts.keys().next_back().copied()),
            ("infer", self.inferences.keys().next_back().copied()),
        ];
        for (kind, max) in maxes {
            let want = max.map_or(0, |m| m + 1);
            let cur = self.next.entry(kind.to_string()).or_insert(0);
            if *cur < want {
                *cur = want;
            }
        }
    }

    /// Allocate a new ID of a kind.
    pub fn alloc(&mut self, kind: &str) -> u32 {
        let cur = self.next.entry(kind.to_string()).or_insert(0);
        let id = *cur;
        *cur += 1;
        id
    }

    /// Names for categories.
    pub fn cat_names(&self) -> CatNames {
        CatNames {
            atoms: self.atoms.iter().map(|a| a.name.clone()).collect(),
            dims: self
                .cat_feats
                .iter()
                .map(|d| (d.name.clone(), d.values.clone()))
                .collect(),
        }
    }

    /// Concept by name.
    pub fn concept_by_name(&self, name: &str) -> Option<u32> {
        self.index.concepts.get(name).copied()
    }

    /// Role by name.
    pub fn role_by_name(&self, name: &str) -> Option<u32> {
        self.index.roles.get(name).copied()
    }

    /// Feature by name.
    pub fn feat_by_name(&self, name: &str) -> Option<u32> {
        self.index.feats.get(name).copied()
    }

    /// Entity by name.
    pub fn entity_by_name(&self, name: &str) -> Option<u32> {
        self.index.entities.get(name).copied()
    }

    /// The supertype of a concept (`TOP`'s parent is `TOP`).
    pub fn parent(&self, c: u32) -> u32 {
        self.concepts.get(&c).map_or(TOP, |x| x.parent)
    }

    /// True if `sub` is `sup` or below it.
    pub fn is_a(&self, sub: u32, sup: u32) -> bool {
        let mut c = sub;
        for _ in 0..64 {
            if c == sup {
                return true;
            }
            if c == TOP {
                return false;
            }
            c = self.parent(c);
        }
        false
    }

    /// The chain from a concept up to `TOP`, inclusive.
    pub fn ancestors(&self, c: u32) -> Vec<u32> {
        let mut out = vec![c];
        let mut cur = c;
        for _ in 0..64 {
            if cur == TOP {
                break;
            }
            cur = self.parent(cur);
            out.push(cur);
        }
        if out.last() != Some(&TOP) {
            out.push(TOP);
        }
        out
    }

    /// The concept a meaning head belongs to, for type checks and contexts.
    pub fn head_concept(&self, h: &Head) -> u32 {
        match h {
            Head::Concept(c) => *c,
            Head::Entity(e) => self.entities.get(e).map_or(TOP, |x| x.concept),
            _ => TOP,
        }
    }
}

impl TermNames for Program {
    fn concept_name(&self, id: u32) -> String {
        self.concepts
            .get(&id)
            .map_or_else(|| format!("c{id}?"), |c| c.name.clone())
    }
    fn role_name(&self, id: u32) -> String {
        self.roles
            .get(&id)
            .map_or_else(|| format!("r{id}?"), |c| c.name.clone())
    }
    fn feat_name(&self, id: u32) -> String {
        self.feats
            .get(&id)
            .map_or_else(|| format!("f{id}?"), |c| c.name.clone())
    }
    fn val_name(&self, feat: u32, val: u16) -> String {
        self.feats
            .get(&feat)
            .and_then(|f| f.values.get(usize::from(val)))
            .map_or_else(|| format!("v{val}?"), Clone::clone)
    }
    fn entity_name(&self, id: u32) -> String {
        self.entities
            .get(&id)
            .map_or_else(|| format!("k{id}?"), |c| c.name.clone())
    }
}

impl TermLookup for Program {
    fn concept_id(&self, name: &str) -> Option<u32> {
        self.concept_by_name(name)
    }
    fn role_id(&self, name: &str) -> Option<u32> {
        self.role_by_name(name)
    }
    fn feat_id(&self, name: &str) -> Option<u32> {
        self.feat_by_name(name)
    }
    fn val_id(&self, feat: u32, name: &str) -> Option<u16> {
        self.feats
            .get(&feat)
            .and_then(|f| f.values.iter().position(|v| v == name))
            .and_then(|i| u16::try_from(i).ok())
    }
    fn entity_id(&self, name: &str) -> Option<u32> {
        self.entity_by_name(name)
    }
}

// ---------------------------------------------------------------------------
// Canonical text form.

/// The files of a program's text form, by file name.
pub type ProgramFiles = BTreeMap<String, String>;

fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn unquote(s: &str) -> Result<(String, &str), String> {
    let s = s.trim_start();
    let mut chars = s.char_indices();
    if chars.next().map(|(_, c)| c) != Some('"') {
        return Err(format!("expected a quoted string at `{s}`"));
    }
    let mut out = String::new();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Ok((out, s.get(i + 1..).unwrap_or_default())),
            '\\' => match chars.next().map(|(_, c)| c) {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(c) => out.push(c),
                None => break,
            },
            c => out.push(c),
        }
    }
    Err("unterminated string".into())
}

fn write_meta(out: &mut String, meta: &Meta) {
    for line in meta.doc.lines() {
        let _ = writeln!(out, "/// {line}");
    }
}

fn tail(meta: &Meta) -> String {
    if meta.seed {
        format!("  @{} seed", meta.round)
    } else {
        format!("  @{}", meta.round)
    }
}

impl Program {
    /// Write the canonical text form.
    pub fn to_files(&self) -> ProgramFiles {
        let mut files = ProgramFiles::new();
        let cn = self.cat_names();

        let mut o =
            String::from("# sememe program: the ontology (concepts, roles, meaning features).\n\n");
        for (id, c) in &self.concepts {
            write_meta(&mut o, &c.meta);
            let _ = writeln!(
                o,
                "c{id} concept {} < {}{}",
                c.name,
                self.concept_name(c.parent),
                tail(&c.meta)
            );
        }
        o.push('\n');
        for (id, r) in &self.roles {
            write_meta(&mut o, &r.meta);
            let _ = writeln!(
                o,
                "r{id} role {} : {} -> {}{}",
                r.name,
                self.concept_name(r.domain),
                self.concept_name(r.range),
                tail(&r.meta)
            );
        }
        o.push('\n');
        for (id, f) in &self.feats {
            write_meta(&mut o, &f.meta);
            let _ = writeln!(
                o,
                "f{id} feature {} : {} = {}{}",
                f.name,
                self.concept_name(f.domain),
                f.values.join(" | "),
                tail(&f.meta)
            );
        }
        files.insert("ontology.sem".into(), o);

        let mut g = String::from(
            "# sememe program: the grammar (categories, combining rules, inflections).\n\n",
        );
        for (id, a) in self.atoms.iter().enumerate() {
            write_meta(&mut g, &a.meta);
            let _ = writeln!(g, "a{id} atom {}{}", a.name, tail(&a.meta));
        }
        for (id, d) in self.cat_feats.iter().enumerate() {
            write_meta(&mut g, &d.meta);
            let atoms: Vec<&str> = d
                .atoms
                .iter()
                .filter_map(|a| self.atoms.get(usize::from(*a)).map(|x| x.name.as_str()))
                .collect();
            let _ = writeln!(
                g,
                "d{id} catfeature {} on {} = {}{}",
                d.name,
                atoms.join(" "),
                d.values.join(" | "),
                tail(&d.meta)
            );
        }
        g.push('\n');
        for (id, r) in &self.rules {
            write_meta(&mut g, &r.meta);
            let body = match &r.kind {
                RuleKind::FApp => "fapp".to_string(),
                RuleKind::BApp => "bapp".to_string(),
                RuleKind::FComp => "fcomp".to_string(),
                RuleKind::BComp => "bcomp".to_string(),
                RuleKind::FXComp => "fxcomp".to_string(),
                RuleKind::BXComp => "bxcomp".to_string(),
                RuleKind::FTr(x, t) => format!("ftr {} {}", x.show(&cn), t.show(&cn)),
                RuleKind::BTr(x, t) => format!("btr {} {}", x.show(&cn), t.show(&cn)),
                RuleKind::Tc(x, y) => format!("tc {} {}", x.show(&cn), y.show(&cn)),
                RuleKind::Coord(c) => format!("coord {}", c.show(&cn)),
            };
            let _ = writeln!(g, "g{id} rule {body}{}", tail(&r.meta));
        }
        g.push('\n');
        for (id, i) in &self.infls {
            write_meta(&mut g, &i.meta);
            let feats: Vec<String> = i
                .feats
                .iter()
                .map(|(f, v)| format!("{}={}", self.feat_name(*f), self.val_name(*f, *v)))
                .collect();
            let _ = writeln!(
                g,
                "i{id} infl {} [{}] {} -> {}{}",
                i.cat.show(&cn),
                feats.join(", "),
                quote(&i.strip),
                quote(&i.add),
                tail(&i.meta)
            );
        }
        files.insert("grammar.sem".into(), g);

        let mut l = String::from("# sememe program: the lexicon. `form category : meaning`.\n\n");
        for (id, n) in &self.names {
            write_meta(&mut l, &n.meta);
            let _ = writeln!(
                l,
                "n{id} name {} : {}{}",
                n.cat.show(&cn),
                self.concept_name(n.concept),
                tail(&n.meta)
            );
        }
        for (id, e) in &self.entries {
            write_meta(&mut l, &e.meta);
            let _ = writeln!(
                l,
                "e{id} entry {} {} : {}{}",
                quote(&e.form),
                e.cat.show(&cn),
                e.sem.show(self),
                tail(&e.meta)
            );
        }
        files.insert("lexicon.sem".into(), l);

        let mut k = String::from(
            "# sememe program: knowledge (named individuals, facts, inference rules).\n\n",
        );
        for (id, e) in &self.entities {
            write_meta(&mut k, &e.meta);
            let _ = writeln!(
                k,
                "k{id} entity {} : {}{}",
                e.name,
                self.concept_name(e.concept),
                tail(&e.meta)
            );
        }
        for (id, f) in &self.facts {
            write_meta(&mut k, &f.meta);
            let _ = writeln!(k, "t{id} fact {}{}", f.term.show(self), tail(&f.meta));
        }
        for (id, f) in &self.inferences {
            write_meta(&mut k, &f.meta);
            let _ = writeln!(
                k,
                "u{id} infer {} => {}{}",
                f.premise.show(self),
                f.conclusion.show(self),
                tail(&f.meta)
            );
        }
        files.insert("knowledge.sem".into(), k);
        files.insert("counts.tsv".into(), self.counts_text());
        files
    }

    fn counts_text(&self) -> String {
        let mut out = String::from(
            "# table\tcontext\tcontext\toutcome\tlevel (quantized count; see cost.rs)\n",
        );
        for ((table, ctx), cc) in &self.counts.tables {
            let tname = crate::cost::Table::from_code(*table).map_or("?", |t| t.name());
            let c0 = self.key_text(*table, 0, ctx[0]);
            let c1 = self.key_text(*table, 1, ctx[1]);
            let _ = writeln!(out, "{tname}\t{c0}\t{c1}\t<esc>\t{}", cc.esc);
            for (o, lvl) in &cc.outs {
                let _ = writeln!(
                    out,
                    "{tname}\t{c0}\t{c1}\t{}\t{lvl}",
                    self.key_text(*table, 2, *o)
                );
            }
        }
        out
    }

    /// Readable form of one key field of a count table.
    pub fn key_text(&self, table: u8, field: usize, key: u64) -> String {
        use crate::cost::KeyKind;
        let kind = crate::cost::Table::from_code(table).map_or(KeyKind::Int, |t| t.key_kind(field));
        if kind != KeyKind::None
            && let Some(s) = crate::cost::special_name(key)
        {
            return s.into();
        }
        match kind {
            KeyKind::None => "-".into(),
            KeyKind::Int => key.to_string(),
            KeyKind::Role => match crate::cost::special_name(key) {
                Some(s) => s.into(),
                None => self.role_name(key as u32),
            },
            KeyKind::Concept => match crate::cost::special_name(key) {
                Some(s) => s.into(),
                None => {
                    if key >> 32 == 1 {
                        format!("#{}", self.entity_name(key as u32))
                    } else {
                        self.concept_name(key as u32)
                    }
                }
            },
            KeyKind::Feat => self.feat_name(key as u32),
            KeyKind::FeatVal(_) => match crate::cost::special_name(key) {
                Some(s) => s.into(),
                None => {
                    let f = (key >> 16) as u32;
                    format!(
                        "{}={}",
                        self.feat_name(f),
                        self.val_name(f, (key & 0xffff) as u16)
                    )
                }
            },
            KeyKind::Rule => match crate::cost::special_name(key) {
                Some(s) => s.into(),
                None => format!("g{key}"),
            },
            KeyKind::Sym => match crate::cost::special_name(key) {
                Some(s) => s.into(),
                None => self
                    .counts
                    .syms
                    .get(&key)
                    .map_or_else(|| format!("${key:016x}"), |s| quote(s)),
            },
        }
    }

    /// Read a program from its text form. Counts are read back with their
    /// symbols; keys are re-derived from the printed forms.
    pub fn from_files(files: &ProgramFiles) -> Result<Program, String> {
        let mut p = Program::default();
        let get = |n: &str| files.get(n).map(String::as_str).unwrap_or_default();
        // Two passes over the ontology: names first (so parents can be
        // forward references), then links.
        let ont = get("ontology.sem");
        let mut pending_parents: Vec<(u32, String)> = Vec::new();
        let mut pending_roles: Vec<(u32, String, String)> = Vec::new();
        let mut pending_feats: Vec<(u32, String)> = Vec::new();
        for (doc, line) in items(ont) {
            let (id, rest) = split_id(line)?;
            let (body, meta) = split_tail(rest, doc)?;
            let mut words = body.split_whitespace();
            match words.next() {
                Some("concept") => {
                    let name = words.next().ok_or("concept without name")?.to_string();
                    let parent = match words.next() {
                        Some("<") => words.next().unwrap_or("Thing").to_string(),
                        _ => "Thing".to_string(),
                    };
                    p.concepts.insert(
                        id,
                        Concept {
                            name,
                            parent: TOP,
                            meta,
                        },
                    );
                    pending_parents.push((id, parent));
                }
                Some("role") => {
                    let name = words.next().ok_or("role without name")?.to_string();
                    let _colon = words.next();
                    let dom = words.next().unwrap_or("Thing").to_string();
                    let _arrow = words.next();
                    let rng = words.next().unwrap_or("Thing").to_string();
                    p.roles.insert(
                        id,
                        Role {
                            name,
                            domain: TOP,
                            range: TOP,
                            meta,
                        },
                    );
                    pending_roles.push((id, dom, rng));
                }
                Some("feature") => {
                    let name = words.next().ok_or("feature without name")?.to_string();
                    let _colon = words.next();
                    let dom = words.next().unwrap_or("Thing").to_string();
                    let _eq = words.next();
                    let values: Vec<String> =
                        words.filter(|w| *w != "|").map(str::to_string).collect();
                    p.feats.insert(
                        id,
                        Feature {
                            name,
                            domain: TOP,
                            values,
                            meta,
                        },
                    );
                    pending_feats.push((id, dom));
                }
                other => return Err(format!("unknown ontology item {other:?}")),
            }
        }
        p.reindex();
        for (id, parent) in pending_parents {
            let pid = p
                .concept_by_name(&parent)
                .ok_or_else(|| format!("unknown parent concept `{parent}`"))?;
            if let Some(c) = p.concepts.get_mut(&id) {
                c.parent = if id == TOP { TOP } else { pid };
            }
        }
        for (id, dom, rng) in pending_roles {
            let d = p
                .concept_by_name(&dom)
                .ok_or_else(|| format!("unknown concept `{dom}`"))?;
            let r = p
                .concept_by_name(&rng)
                .ok_or_else(|| format!("unknown concept `{rng}`"))?;
            if let Some(x) = p.roles.get_mut(&id) {
                x.domain = d;
                x.range = r;
            }
        }
        for (id, dom) in pending_feats {
            let d = p
                .concept_by_name(&dom)
                .ok_or_else(|| format!("unknown concept `{dom}`"))?;
            if let Some(x) = p.feats.get_mut(&id) {
                x.domain = d;
            }
        }

        // Grammar: atoms and category features first, then rules and
        // inflections (which need category names).
        let gram = get("grammar.sem");
        let mut later: Vec<(String, u32, String, Meta)> = Vec::new();
        for (doc, line) in items(gram) {
            let (id, rest) = split_id(line)?;
            let (body, meta) = split_tail(rest, doc)?;
            let mut words = body.splitn(2, ' ');
            let kw = words.next().unwrap_or_default();
            let rest = words.next().unwrap_or_default().trim();
            match kw {
                "atom" => p.atoms.push(AtomDecl {
                    name: rest.to_string(),
                    meta,
                }),
                "catfeature" => {
                    let (name, rest) = rest.split_once(" on ").ok_or("catfeature without `on`")?;
                    let (atoms, vals) = rest.split_once(" = ").ok_or("catfeature without `=`")?;
                    let atoms = atoms
                        .split_whitespace()
                        .map(|a| {
                            p.atoms
                                .iter()
                                .position(|x| x.name == a)
                                .and_then(|i| u16::try_from(i).ok())
                                .ok_or_else(|| format!("unknown atom `{a}`"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    p.cat_feats.push(CatFeat {
                        name: name.trim().to_string(),
                        values: vals.split('|').map(|v| v.trim().to_string()).collect(),
                        atoms,
                        meta,
                    });
                }
                other => later.push((other.to_string(), id, rest.to_string(), meta)),
            }
        }
        let cn = p.cat_names();
        for (kw, id, rest, meta) in later {
            match kw.as_str() {
                "rule" => {
                    let mut w = rest.split_whitespace();
                    let k = w.next().unwrap_or_default();
                    let mut cat = || -> Result<Cat, String> {
                        Cat::parse(w.next().ok_or("rule missing category")?, &cn)
                    };
                    let kind = match k {
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
                        other => return Err(format!("unknown rule kind `{other}`")),
                    };
                    p.rules.insert(id, Rule { kind, meta });
                }
                "infl" => {
                    let (cat, rest) = rest.split_once(" [").ok_or("infl without features")?;
                    let (feats, rest) = rest.split_once(']').ok_or("infl without `]`")?;
                    let (strip, rest) = unquote(rest)?;
                    let rest = rest
                        .trim_start()
                        .strip_prefix("->")
                        .ok_or("infl without `->`")?;
                    let (add, _) = unquote(rest)?;
                    let mut fv = Vec::new();
                    for pair in feats.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        let (f, v) = pair.split_once('=').ok_or("bad infl feature")?;
                        let fid = p
                            .feat_by_name(f)
                            .ok_or_else(|| format!("unknown feature `{f}`"))?;
                        let vid = p
                            .val_id(fid, v)
                            .ok_or_else(|| format!("unknown value `{v}`"))?;
                        fv.push((fid, vid));
                    }
                    fv.sort();
                    p.infls.insert(
                        id,
                        Infl {
                            cat: Cat::parse(cat, &cn)?,
                            feats: fv,
                            strip,
                            add,
                            meta,
                        },
                    );
                }
                other => return Err(format!("unknown grammar item `{other}`")),
            }
        }

        // Knowledge entities come before the lexicon (meanings may name them).
        let know = get("knowledge.sem");
        let mut facts: Vec<(String, u32, String, Meta)> = Vec::new();
        for (doc, line) in items(know) {
            let (id, rest) = split_id(line)?;
            let (body, meta) = split_tail(rest, doc)?;
            let (kw, rest) = body.split_once(' ').unwrap_or((body, ""));
            if kw == "entity" {
                let (name, concept) = rest.split_once(" : ").ok_or("entity without `:`")?;
                let c = p
                    .concept_by_name(concept.trim())
                    .ok_or_else(|| format!("unknown concept `{concept}`"))?;
                p.entities.insert(
                    id,
                    Entity {
                        name: name.trim().to_string(),
                        concept: c,
                        meta,
                    },
                );
            } else {
                facts.push((kw.to_string(), id, rest.to_string(), meta));
            }
        }
        p.reindex();
        for (kw, id, rest, meta) in facts {
            match kw.as_str() {
                "fact" => {
                    let term = Term::parse(&rest, &p)?;
                    p.facts.insert(id, Fact { term, meta });
                }
                "infer" => {
                    let (a, b) = rest.split_once(" => ").ok_or("infer without `=>`")?;
                    let premise = Term::parse(a, &p)?;
                    let conclusion = Term::parse(b, &p)?;
                    p.inferences.insert(
                        id,
                        Inference {
                            premise,
                            conclusion,
                            meta,
                        },
                    );
                }
                other => return Err(format!("unknown knowledge item `{other}`")),
            }
        }

        let lex = get("lexicon.sem");
        for (doc, line) in items(lex) {
            let (id, rest) = split_id(line)?;
            let (body, meta) = split_tail(rest, doc)?;
            let (kw, rest) = body.split_once(' ').unwrap_or((body, ""));
            match kw {
                "entry" => {
                    let (form, rest) = unquote(rest)?;
                    let (cat, sem) = rest.split_once(" : ").ok_or("entry without `:`")?;
                    let cat = Cat::parse(cat.trim(), &cn)?;
                    let sem = Term::parse(sem, &p)?;
                    p.entries.insert(
                        id,
                        Entry {
                            form,
                            cat,
                            sem,
                            meta,
                        },
                    );
                }
                "name" => {
                    let (cat, concept) = rest.split_once(" : ").ok_or("name without `:`")?;
                    let c = p
                        .concept_by_name(concept.trim())
                        .ok_or_else(|| format!("unknown concept `{concept}`"))?;
                    p.names.insert(
                        id,
                        NameEntry {
                            cat: Cat::parse(cat.trim(), &cn)?,
                            concept: c,
                            meta,
                        },
                    );
                }
                other => return Err(format!("unknown lexicon item `{other}`")),
            }
        }
        p.counts = p.parse_counts(get("counts.tsv"))?;
        p.reindex();
        p.fix_next();
        Ok(p)
    }

    fn parse_counts(&self, text: &str) -> Result<Counts, String> {
        let mut counts = Counts::default();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            let [t, c0, c1, o, lvl] = cols.as_slice() else {
                return Err(format!("bad counts line `{line}`"));
            };
            let table =
                crate::cost::Table::from_name(t).ok_or_else(|| format!("unknown table `{t}`"))?;
            let tc = table as u8;
            let k0 = self.parse_key(tc, 0, c0, &mut counts.syms)?;
            let k1 = self.parse_key(tc, 1, c1, &mut counts.syms)?;
            let lvl: u32 = lvl.parse().map_err(|_| format!("bad level `{lvl}`"))?;
            let cc = counts.tables.entry((tc, [k0, k1])).or_default();
            if *o == "<esc>" {
                cc.esc = lvl;
            } else {
                let ko = self.parse_key(tc, 2, o, &mut counts.syms)?;
                cc.outs.insert(ko, lvl);
            }
        }
        Ok(counts)
    }

    fn parse_key(
        &self,
        table: u8,
        field: usize,
        text: &str,
        syms: &mut BTreeMap<u64, String>,
    ) -> Result<u64, String> {
        use crate::cost::KeyKind;
        if let Some(k) = crate::cost::special_key(text) {
            return Ok(k);
        }
        let kind = crate::cost::Table::from_code(table).map_or(KeyKind::Int, |t| t.key_kind(field));
        let id = |o: Option<u32>| {
            o.map(u64::from)
                .ok_or_else(|| format!("unknown name `{text}`"))
        };
        match kind {
            KeyKind::None => Ok(0),
            KeyKind::Int => text.parse().map_err(|_| format!("bad number `{text}`")),
            KeyKind::FeatVal(_) => {
                let (f, v) = text
                    .split_once('=')
                    .ok_or_else(|| format!("bad feature value `{text}`"))?;
                let fid = self
                    .feat_by_name(f)
                    .ok_or_else(|| format!("unknown feature `{f}`"))?;
                let vid = self
                    .val_id(fid, v)
                    .ok_or_else(|| format!("unknown value `{v}`"))?;
                Ok(crate::cost::featval_key(fid, vid))
            }
            KeyKind::Role => id(self.role_by_name(text)),
            KeyKind::Concept => {
                if let Some(e) = text.strip_prefix('#') {
                    Ok((1u64 << 32) | u64::from(self.entity_by_name(e).ok_or("unknown entity")?))
                } else {
                    id(self.concept_by_name(text))
                }
            }
            KeyKind::Feat => id(self.feat_by_name(text)),
            KeyKind::Rule => text
                .strip_prefix('g')
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| format!("bad rule `{text}`")),
            KeyKind::Sym => {
                if let Some(hex) = text.strip_prefix('$') {
                    return u64::from_str_radix(hex, 16).map_err(|_| format!("bad key `{text}`"));
                }
                let (s, _) = unquote(text)?;
                let key = crate::cost::sym_key_from_text(self, table, field, &s)?;
                syms.insert(key, s);
                Ok(key)
            }
        }
    }
}

/// Lines of a `.sem` file as `(doc comment, item line)`.
fn items(text: &str) -> Vec<(String, &str)> {
    let mut out = Vec::new();
    let mut doc = String::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(d) = t.strip_prefix("///") {
            if !doc.is_empty() {
                doc.push('\n');
            }
            doc.push_str(d.strip_prefix(' ').unwrap_or(d));
        } else if t.is_empty() || t.starts_with('#') {
            continue;
        } else {
            out.push((std::mem::take(&mut doc), t));
        }
    }
    out
}

fn split_id(line: &str) -> Result<(u32, &str), String> {
    let (id, rest) = line
        .split_once(' ')
        .ok_or_else(|| format!("bad item line `{line}`"))?;
    let num: String = id.chars().skip(1).collect();
    let id = num.parse().map_err(|_| format!("bad item id `{id}`"))?;
    Ok((id, rest))
}

fn split_tail(rest: &str, doc: String) -> Result<(&str, Meta), String> {
    let (body, tail) = rest
        .rsplit_once("  @")
        .ok_or_else(|| format!("item without round: `{rest}`"))?;
    let mut w = tail.split_whitespace();
    let round = w
        .next()
        .and_then(|r| r.parse().ok())
        .ok_or_else(|| format!("bad round in `{rest}`"))?;
    let seed = w.next() == Some("seed");
    Ok((body.trim(), Meta { round, doc, seed }))
}

/// Shared pointer to an immutable program version.
pub type ProgramRef = Arc<Program>;

impl Program {
    /// Type-check a meaning term against the ontology: every role child's
    /// parent must be a kind of the role's domain and the child a kind of
    /// its range; every feature's node a kind of the feature's domain.
    /// Unknown, missing, reference and query nodes fit any type. Returns
    /// the errors found.
    pub fn type_errors(&self, t: &Term) -> Vec<String> {
        let mut errs = Vec::new();
        self.check_term(t, &mut errs);
        errs
    }

    fn concept_of(&self, h: &Head) -> Option<u32> {
        match h {
            Head::Concept(c) => Some(*c),
            Head::Entity(e) => self.entities.get(e).map(|x| x.concept),
            _ => None,
        }
    }

    fn check_term(&self, t: &Term, errs: &mut Vec<String>) {
        match t {
            Term::Node(n) => {
                let here = self.concept_of(&n.head);
                for (f, v) in &n.feats {
                    match self.feats.get(f) {
                        None => errs.push(format!("unknown feature {f}")),
                        Some(fd) => {
                            if usize::from(*v) >= fd.values.len() {
                                errs.push(format!("feature {} has no value {v}", fd.name));
                            }
                            if let Some(c) = here
                                && !self.is_a(c, fd.domain)
                            {
                                errs.push(format!(
                                    "{} cannot have feature {}: it is not a kind of {}",
                                    self.concept_name(c),
                                    fd.name,
                                    self.concept_name(fd.domain)
                                ));
                            }
                        }
                    }
                }
                for (r, child) in &n.roles {
                    match self.roles.get(r) {
                        None => errs.push(format!("unknown role {r}")),
                        Some(rd) => {
                            if let Some(c) = here
                                && !self.is_a(c, rd.domain)
                            {
                                errs.push(format!(
                                    "{} cannot have role {}: it is not a kind of {}",
                                    self.concept_name(c),
                                    rd.name,
                                    self.concept_name(rd.domain)
                                ));
                            }
                            if let Some(cn) = child.as_node()
                                && let Some(c) = self.concept_of(&cn.head)
                                && !self.is_a(c, rd.range)
                            {
                                errs.push(format!(
                                    "{} cannot fill role {}: it is not a kind of {}",
                                    self.concept_name(c),
                                    rd.name,
                                    self.concept_name(rd.range)
                                ));
                            }
                        }
                    }
                    self.check_term(child, errs);
                }
            }
            Term::Lam(b) => self.check_term(b, errs),
            Term::App(a, b) | Term::Add(a, _, b) => {
                self.check_term(a, errs);
                self.check_term(b, errs);
            }
            Term::Feat(a, _, _) => self.check_term(a, errs),
            Term::Var(_) => {}
        }
    }
}
