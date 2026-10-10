//! THE FROZEN COST CODE.
//!
//! Everything that defines how many bits a program and a text cost lives in
//! this one module (plus the character model `charmodel.tsv` it embeds):
//!
//! - the search bounds (fragment length cap, referent lookback, candidate
//!   facts, inference depth, beam),
//! - the fixed character model for spelling,
//! - the count tables, what each count is conditioned on, the back-off
//!   chains and the base distributions,
//! - the code for program bits.
//!
//! Its version hash (`COST_HASH`) is recorded with every metric. Nothing
//! changes it during a run; a change means a new run from scratch.
//!
//! ## How a probability is computed from counts
//!
//! Each coding event has a table, a context and an outcome. The context has
//! a back-off chain from specific to general (for concepts, up the type
//! hierarchy). At each level that has stored counts, with quantized values
//! `v(o)` for stored outcomes and `v(esc)` for the escape count:
//!
//! ```text
//! N = sum v(o) + v(esc) + 1
//! P(o) = v(o) / N              if o is stored here
//! P(esc) = (v(esc) + 1) / N    otherwise: escape and try the next level
//! ```
//!
//! A level with no stored counts escapes with probability 1. After the last
//! level, the outcome is coded by the table's fixed base distribution
//! (uniform over the outcome space, the category prior, or the character
//! model). Every distribution sums to at most 1 (escape mass is not
//! redistributed), so the code is a valid, slightly deficient prefix code.
//!
//! ## What the bits mean
//!
//! A line costs the bits of its single cheapest analysis found by the
//! bounded search (meaning, derivation, formatting, unknowns): a Viterbi
//! path. That is a valid code length and an upper bound on the model's true
//! cost of the line.

use std::sync::OnceLock;

use crate::category::{Cat, Dir, FVal};
use crate::program::{CtxCounts, Program, TOP};
use crate::term::{Head, Label, Term};

/// The version hash of this module and the character model.
pub const COST_HASH: &str = env!("SEMEME_COST_HASH");

// ---------------------------------------------------------------------------
// Fixed-point bits.

/// Bits as fixed-point integers with 16 fractional bits, summed in line
/// order so results never depend on thread scheduling.
pub type Bits = u64;

/// One bit in fixed point.
pub const ONE_BIT: Bits = 1 << 16;

/// Convert a non-negative bit count to fixed point.
pub fn fixed(x: f64) -> Bits {
    if x.is_finite() && x > 0.0 {
        (x * ONE_BIT as f64).round() as Bits
    } else {
        0
    }
}

/// Convert fixed point to a float number of bits.
pub fn float(b: Bits) -> f64 {
    b as f64 / ONE_BIT as f64
}

/// `log2(n)` bits in fixed point (0 for n <= 1).
pub fn lg(n: u64) -> Bits {
    if n <= 1 { 0 } else { fixed((n as f64).log2()) }
}

/// Bits of `-log2(num / den)`.
pub fn neg_log(num: u64, den: u64) -> Bits {
    if num == 0 || den == 0 || num >= den {
        0
    } else {
        fixed((den as f64 / num as f64).log2())
    }
}

/// Elias gamma code length for `n >= 1`.
pub fn gamma(n: u64) -> Bits {
    let n = n.max(1);
    let floor_log = 63 - u64::from(n.leading_zeros());
    (2 * floor_log + 1) * ONE_BIT
}

// ---------------------------------------------------------------------------
// Search bounds (part of the code: they decide which analyses exist).

/// Longest fragment, in words. Longer punctuation-free runs get a forced
/// fragment boundary at the cap.
pub const MAX_FRAGMENT_WORDS: usize = 12;
/// How many earlier entities a reference can point at (plus speaker and
/// addressee).
pub const REF_LOOKBACK: usize = 32;
/// How many earlier fragment meanings can be restated.
pub const FACT_LOOKBACK: usize = 32;
/// Most candidate facts considered for restating one slot.
pub const MAX_FACT_CANDIDATES: usize = 256;
/// Maximum inference depth.
pub const INFERENCE_DEPTH: usize = 1;
/// How many recent speakers can be referred to by rank.
pub const SPEAKER_LOOKBACK: usize = 8;
/// Chart beam: analyses kept per span.
pub const BEAM: usize = 12;
/// Relative beam: chart items more than this many bits worse than the best
/// item of their span are dropped.
pub const BEAM_DELTA: Bits = 24 * ONE_BIT;
/// Categories tried for an unknown word.
pub const UNKNOWN_CATS: usize = 10;
/// Categories tried as an unknown reading of a word that has entries.
pub const UNKNOWN_CATS_KNOWN: usize = 3;
/// Largest category arity considered in derivations.
pub const MAX_ARITY: usize = 3;
/// Longest document in a training set, in lines.
pub const DOC_MAX_LINES: usize = 64;

// ---------------------------------------------------------------------------
// The character model.

const CHARMODEL: &str = include_str!("charmodel.tsv");

/// Number of spelling symbols: 256 bytes and the end marker.
pub const CHAR_SYMBOLS: usize = 257;

fn char_bits_table() -> &'static [Bits; CHAR_SYMBOLS] {
    static TABLE: OnceLock<[Bits; CHAR_SYMBOLS]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut counts = [1u64; CHAR_SYMBOLS];
        for line in CHARMODEL.lines() {
            if line.starts_with('#') {
                continue;
            }
            let Some((sym, n)) = line.split_once('\t') else {
                continue;
            };
            let Ok(n) = n.trim().parse::<u64>() else {
                continue;
            };
            let idx = if sym == "end" {
                256
            } else {
                sym.parse::<usize>().unwrap_or(usize::MAX)
            };
            if let Some(c) = counts.get_mut(idx) {
                *c += n;
            }
        }
        let total: u64 = counts.iter().sum();
        let mut out = [0; CHAR_SYMBOLS];
        for (o, c) in out.iter_mut().zip(counts.iter()) {
            *o = neg_log(*c, total);
        }
        out
    })
}

/// Bits to spell a byte string with the fixed character model, including
/// the end marker.
pub fn spell(s: &[u8]) -> Bits {
    let t = char_bits_table();
    let mut b: Bits = t.get(256).copied().unwrap_or(0);
    for &c in s {
        b += t.get(usize::from(c)).copied().unwrap_or(0);
    }
    b
}

// ---------------------------------------------------------------------------
// Quantized counts.

/// The count a stored level stands for. Levels 0..=16 are exact; above,
/// eight levels per doubling.
pub fn level_value(q: u32) -> u64 {
    if q <= 16 {
        u64::from(q)
    } else {
        let v = 16.0 * 2f64.powf(f64::from(q - 16) / 8.0);
        v.floor() as u64
    }
}

/// The nearest level for a count.
pub fn level_of(count: u64) -> u32 {
    if count <= 16 {
        count as u32
    } else {
        let q = 16.0 + 8.0 * (count as f64 / 16.0).log2();
        q.round() as u32
    }
}

// ---------------------------------------------------------------------------
// Keys.

/// Reserved key values (never real IDs or hashes in practice).
pub const K_ANY: u64 = u64::MAX;
/// The root slot (no parent role).
pub const K_ROOT: u64 = u64::MAX - 1;
/// No parent concept.
pub const K_NONE: u64 = u64::MAX - 2;
/// Head: unknown meaning.
pub const K_UNKNOWN: u64 = u64::MAX - 3;
/// Head: missing (unsaid) meaning.
pub const K_MISSING: u64 = u64::MAX - 4;
/// Head: question placeholder.
pub const K_QUERY: u64 = u64::MAX - 5;
/// Absent feature value / no inflection.
pub const K_ABSENT: u64 = u64::MAX - 6;
/// Lexical choice: leaf by a combining-rule expansion code `lex`.
pub const K_LEX: u64 = u64::MAX - 7;
/// Speaker rank: a new speaker.
pub const K_NEW: u64 = u64::MAX - 8;

const SPECIALS: [(u64, &str); 9] = [
    (K_ANY, "*"),
    (K_ROOT, "<root>"),
    (K_NONE, "<none>"),
    (K_UNKNOWN, "<unknown>"),
    (K_MISSING, "<missing>"),
    (K_QUERY, "<query>"),
    (K_ABSENT, "<absent>"),
    (K_LEX, "<lex>"),
    (K_NEW, "<new>"),
];

/// Printable name of a reserved key.
pub fn special_name(k: u64) -> Option<&'static str> {
    SPECIALS.iter().find(|(v, _)| *v == k).map(|(_, n)| *n)
}

/// Reserved key from its printed name.
pub fn special_key(s: &str) -> Option<u64> {
    SPECIALS.iter().find(|(_, n)| *n == s).map(|(v, _)| *v)
}

/// Tag for a lexical-entry outcome.
pub const TAG_ENTRY: u64 = 2 << 32;
/// Tag for a name-entry outcome.
pub const TAG_NAME: u64 = 3 << 32;
/// Tag for a global-entity head outcome.
pub const TAG_ENTITY: u64 = 1 << 32;

/// The key of a meaning head as a concept-table outcome or context.
pub fn head_key(h: &Head) -> u64 {
    match h {
        Head::Concept(c) => u64::from(*c),
        Head::Entity(e) => TAG_ENTITY | u64::from(*e),
        Head::Unknown => K_UNKNOWN,
        Head::Missing => K_MISSING,
        Head::Query => K_QUERY,
        Head::Ref(_) => K_NONE,
    }
}

struct Fnv(u64);

impl Fnv {
    fn new(tag: u8) -> Fnv {
        let mut f = Fnv(0xcbf2_9ce4_8422_2325);
        f.byte(tag);
        f
    }
    fn byte(&mut self, b: u8) {
        self.0 ^= u64::from(b);
        self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
    }
    fn u32(&mut self, x: u32) {
        for b in x.to_le_bytes() {
            self.byte(b);
        }
    }
    fn bytes(&mut self, s: &[u8]) {
        self.u32(s.len() as u32);
        for &b in s {
            self.byte(b);
        }
    }
}

/// Stable key of a functor category from its parts' keys.
pub fn slash_key(dir: Dir, res: u64, arg: u64) -> u64 {
    let mut f = Fnv::new(b'C');
    f.byte(2);
    f.byte(match dir {
        Dir::Fwd => 0,
        Dir::Bwd => 1,
    });
    for b in res.to_le_bytes().into_iter().chain(arg.to_le_bytes()) {
        f.byte(b);
    }
    f.0
}

fn hash_cat(c: &Cat, f: &mut Fnv) {
    match c {
        Cat::Atom(a) => {
            f.byte(1);
            f.u32(u32::from(a.id));
            f.u32(a.feats.len() as u32);
            for (d, v) in &a.feats {
                f.u32(u32::from(*d));
                match v {
                    FVal::Val(x) => {
                        f.byte(0);
                        f.u32(u32::from(*x));
                    }
                    FVal::Var(x) => {
                        f.byte(1);
                        f.u32(u32::from(*x));
                    }
                }
            }
        }
        Cat::Slash(s) => {
            for b in s.key.to_le_bytes() {
                f.byte(b);
            }
        }
    }
}

/// Stable key of a category: atoms hash their id and features; functors
/// combine their direction and their parts' keys.
pub fn cat_key(c: &Cat) -> u64 {
    match c {
        Cat::Slash(s) => s.key,
        Cat::Atom(_) => {
            let mut f = Fnv::new(b'C');
            hash_cat(c, &mut f);
            f.0
        }
    }
}

fn hash_head(h: &Head, f: &mut Fnv) {
    match h {
        Head::Concept(c) => {
            f.byte(1);
            f.u32(*c);
        }
        Head::Unknown => f.byte(2),
        Head::Missing => f.byte(3),
        Head::Ref(d) => {
            f.byte(4);
            f.byte(*d as u8);
        }
        Head::Query => f.byte(5),
        Head::Entity(e) => {
            f.byte(6);
            f.u32(*e);
        }
    }
}

fn hash_term(t: &Term, f: &mut Fnv, with_names: bool) {
    match t {
        Term::Var(i) => {
            f.byte(1);
            f.u32(*i);
        }
        Term::Lam(b) => {
            f.byte(2);
            hash_term(b, f, with_names);
        }
        Term::App(a, b) => {
            f.byte(3);
            hash_term(a, f, with_names);
            hash_term(b, f, with_names);
        }
        Term::Node(n) => {
            f.byte(4);
            hash_head(&n.head, f);
            match (&n.name, with_names) {
                (Some(s), true) => f.bytes(s.as_bytes()),
                (Some(_), false) => f.byte(0xfe),
                (None, _) => f.byte(0xff),
            }
            f.u32(n.feats.len() as u32);
            for (a, b) in &n.feats {
                f.u32(*a);
                f.u32(u32::from(*b));
            }
            f.u32(n.roles.len() as u32);
            for (r, c) in &n.roles {
                f.u32(*r);
                hash_term(c, f, with_names);
            }
        }
        Term::Add(a, r, b) => {
            f.byte(5);
            f.u32(*r);
            hash_term(a, f, with_names);
            hash_term(b, f, with_names);
        }
        Term::Feat(a, x, v) => {
            f.byte(6);
            f.u32(*x);
            f.u32(u32::from(*v));
            hash_term(a, f, with_names);
        }
    }
}

/// Stable key of a lexical meaning (names abstracted: a name entry's
/// meaning key does not depend on the name it carries).
pub fn sem_key(t: &Term) -> u64 {
    let mut f = Fnv::new(b'T');
    hash_term(t, &mut f, false);
    f.0
}

/// Stable key of a meaning including names (for comparing facts).
pub fn meaning_key(t: &Term) -> u64 {
    let mut f = Fnv::new(b'M');
    hash_term(t, &mut f, true);
    f.0
}

/// Stable key of a string (words, gaps, speaker names).
pub fn str_key(s: &str) -> u64 {
    let mut f = Fnv::new(b'S');
    f.bytes(s.as_bytes());
    f.0
}

/// Stable key of a set of meaning features.
pub fn featset_key(fs: &[(u32, u16)]) -> u64 {
    let mut f = Fnv::new(b'F');
    f.u32(fs.len() as u32);
    for (a, b) in fs {
        f.u32(*a);
        f.u32(u32::from(*b));
    }
    f.0
}

/// Key of a split label.
pub fn label_key(l: Label) -> u64 {
    (u64::from(l.kind as u8) << 32) | u64::from(l.name)
}

// ---------------------------------------------------------------------------
// Tables.

/// The count tables. Each comment says what the count is conditioned on
/// (the context fields) and what it predicts (the outcome).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Table {
    /// Speaker-tag kind given the previous line's tag kind.
    Tag = 0,
    /// Which recent speaker talks (rank in the speaker list), or `<new>`.
    SpeakerRank = 1,
    /// A new speaker's name (base: spelled).
    SpeakerName = 2,
    /// Whether a line has no words, given its tag kind.
    LineEmpty = 3,
    /// Whether another fragment follows on the line.
    FragMore = 4,
    /// How a meaning slot is filled (fresh, refer back, restate, infer),
    /// given the slot's role. Back-off: any role.
    Mode = 5,
    /// The head of a fresh node, given its role and the parent's concept.
    /// Back-off: the parent's supertypes, then any context.
    Concept = 6,
    /// The node's next feature and its value (features are coded in
    /// feature-ID order), or `<absent>` for no more, given the previous
    /// feature (`<root>` before the first) and the node's concept.
    /// Back-off: the concept's supertypes, then any previous feature.
    FeatNext = 7,
    /// The node's next role child (in role-ID order, repeats allowed), or
    /// `<absent>` for no more, given the previous role (`<root>` before the
    /// first) and the node's concept. Back-off: supertypes, then any.
    RoleNext = 8,
    /// Unused (kept so table codes stay stable).
    Unused9 = 9,
    /// Which context entity a reference points at (recency rank; 0 is the
    /// speaker, 1 the addressee), given the slot's role. Back-off: any.
    RefRank = 10,
    /// Whether a fresh node carries a proper name, given its concept.
    /// Back-off: supertypes. (The name itself is spelled.)
    HasName = 11,
    /// The category at the root of a fragment's derivation.
    RootCat = 12,
    /// How a derivation node expands (`<lex>` leaf or a rule), given its
    /// category. Back-off: the category without features, then any.
    Expand = 13,
    /// The new category a binary rule introduces, given rule and parent
    /// category. Back-off: the rule alone.
    NewCat = 14,
    /// How a binary step splits the meaning (which role or feature goes to
    /// which child), given the rule kind and parent category. Back-off: the
    /// rule kind alone.
    Split = 15,
    /// Which inflection a leaf uses, given its category and the meaning
    /// features of its head. Back-off: any.
    Infl = 16,
    /// Which lexical entry (or spelled word) realizes a leaf, given its
    /// category and meaning.
    Lex = 17,
    /// Letter case of a word, given its position class.
    Case = 18,
    /// The text between words, given the gap's position class.
    Gap = 19,
}

/// What a key field holds, for printing and coding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyKind {
    /// Unused field.
    None,
    /// A small integer.
    Int,
    /// A role ID (or `<root>`/`*`).
    Role,
    /// A concept, entity or special head.
    Concept,
    /// A feature ID.
    Feat,
    /// A value of the feature in context field 0.
    FeatVal(u8),
    /// A rule ID, `<lex>` or a rule-kind code.
    Rule,
    /// A hashed symbol: category, meaning or string.
    Sym,
}

const TABLES: [Table; 20] = [
    Table::Tag,
    Table::SpeakerRank,
    Table::SpeakerName,
    Table::LineEmpty,
    Table::FragMore,
    Table::Mode,
    Table::Concept,
    Table::FeatNext,
    Table::RoleNext,
    Table::Unused9,
    Table::RefRank,
    Table::HasName,
    Table::RootCat,
    Table::Expand,
    Table::NewCat,
    Table::Split,
    Table::Infl,
    Table::Lex,
    Table::Case,
    Table::Gap,
];

impl Table {
    /// All tables.
    pub fn all() -> &'static [Table] {
        &TABLES
    }

    /// Table from its code.
    pub fn from_code(c: u8) -> Option<Table> {
        TABLES.get(usize::from(c)).copied()
    }

    /// Printable name.
    pub fn name(self) -> &'static str {
        match self {
            Table::Tag => "tag",
            Table::SpeakerRank => "speaker_rank",
            Table::SpeakerName => "speaker_name",
            Table::LineEmpty => "line_empty",
            Table::FragMore => "frag_more",
            Table::Mode => "mode",
            Table::Concept => "concept",
            Table::FeatNext => "feat_next",
            Table::RoleNext => "role_next",
            Table::Unused9 => "unused",
            Table::RefRank => "ref_rank",
            Table::HasName => "has_name",
            Table::RootCat => "root_cat",
            Table::Expand => "expand",
            Table::NewCat => "new_cat",
            Table::Split => "split",
            Table::Infl => "infl",
            Table::Lex => "lex",
            Table::Case => "case",
            Table::Gap => "gap",
        }
    }

    /// Table from its printable name.
    pub fn from_name(s: &str) -> Option<Table> {
        TABLES.iter().copied().find(|t| t.name() == s)
    }

    /// What field `i` (0, 1 = context, 2 = outcome) holds.
    pub fn key_kind(self, i: usize) -> KeyKind {
        use KeyKind::*;
        let k: [KeyKind; 3] = match self {
            Table::Tag => [Int, None, Int],
            Table::SpeakerRank => [None, None, Int],
            Table::SpeakerName => [None, None, Sym],
            Table::LineEmpty => [Int, None, Int],
            Table::FragMore => [None, None, Int],
            Table::Mode => [Role, None, Int],
            Table::Concept => [Role, Concept, Concept],
            Table::FeatNext => [Feat, Concept, FeatVal(0)],
            Table::RoleNext => [Role, Concept, Role],
            Table::Unused9 => [None, None, None],
            Table::RefRank => [Role, None, Int],
            Table::HasName => [Concept, None, Int],
            Table::RootCat => [None, None, Sym],
            Table::Expand => [Sym, None, Rule],
            Table::NewCat => [Rule, Sym, Sym],
            Table::Split => [Int, Sym, Int],
            Table::Infl => [Sym, Sym, Rule],
            Table::Lex => [Sym, Sym, Sym],
            Table::Case => [Int, None, Int],
            Table::Gap => [Int, None, Sym],
        };
        k.get(i).copied().unwrap_or(None)
    }
}

/// One coding event of an analysis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Event {
    /// Which table codes it.
    pub table: Table,
    /// The most specific context.
    pub ctx: [u64; 2],
    /// The outcome.
    pub out: u64,
    /// Base-distribution bits for outcomes that are spelled (words, gaps,
    /// speaker names); 0 otherwise.
    pub spell: Bits,
}

/// Fixed numbers about the program that the base distributions use.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sizes {
    /// Concepts.
    pub concepts: u64,
    /// Global entities.
    pub entities: u64,
    /// Roles.
    pub roles: u64,
    /// Meaning features.
    pub feats: u64,
    /// Rules.
    pub rules: u64,
    /// Inflections.
    pub infls: u64,
    /// Lexical entries.
    pub entries: u64,
    /// Name entries.
    pub names: u64,
    /// Atomic categories.
    pub atoms: u64,
    /// Total number of meaning-feature values over all features.
    pub feat_values: u64,
}

impl Sizes {
    /// Sizes of a program.
    pub fn of(p: &Program) -> Sizes {
        Sizes {
            concepts: p.concepts.len() as u64,
            entities: p.entities.len() as u64,
            roles: p.roles.len() as u64,
            feats: p.feats.len() as u64,
            rules: p.rules.len() as u64,
            infls: p.infls.len() as u64,
            entries: p.entries.len() as u64,
            names: p.names.len() as u64,
            atoms: p.atoms.len() as u64,
            feat_values: p.feats.values().map(|f| f.values.len() as u64).sum(),
        }
    }

    /// Number of possible fresh heads: concepts, entities, unknown,
    /// missing, query.
    pub fn heads(&self) -> u64 {
        self.concepts + self.entities + 3
    }

    /// Number of possible split labels.
    pub fn labels(&self) -> u64 {
        2 * self.roles + self.feats + 3
    }
}

/// The back-off chain of an event's context, most specific first. The
/// chain of any later context in a chain is the rest of the chain.
pub fn chain(p: &Program, e: &Event) -> Vec<[u64; 2]> {
    let mut c = chain_raw(p, e);
    c.dedup();
    c
}

fn chain_raw(p: &Program, e: &Event) -> Vec<[u64; 2]> {
    let [a, b] = e.ctx;
    match e.table {
        Table::Mode | Table::RefRank => {
            if a == K_ANY {
                vec![[K_ANY, 0]]
            } else {
                vec![[a, 0], [K_ANY, 0]]
            }
        }
        Table::Concept => {
            let mut out = Vec::new();
            if b == K_NONE || b == K_ANY {
                out.push([a, b]);
            } else {
                for c in concept_chain(p, b) {
                    out.push([a, c]);
                }
            }
            if a != K_ANY {
                out.push([K_ANY, K_ANY]);
            }
            out
        }
        Table::FeatNext | Table::RoleNext => {
            let mut out: Vec<[u64; 2]> = concept_chain(p, b).into_iter().map(|c| [a, c]).collect();
            if a != K_ANY {
                out.push([K_ANY, u64::from(TOP)]);
            }
            out
        }
        Table::HasName => concept_chain(p, a).into_iter().map(|c| [c, 0]).collect(),
        Table::Expand => vec![[a, 0], [K_ANY, 0]],
        Table::NewCat | Table::Split => vec![[a, b], [a, K_ANY]],
        Table::Infl => vec![[a, b], [K_ANY, K_ANY]],
        _ => vec![[a, b]],
    }
}

/// A concept (or special head key) and its supertypes up to `TOP`.
fn concept_chain(p: &Program, c: u64) -> Vec<u64> {
    if c == K_UNKNOWN || c == K_MISSING || c == K_QUERY || c == K_NONE {
        return vec![c, u64::from(TOP)];
    }
    if c & TAG_ENTITY != 0 && c >> 32 == 1 {
        let concept = p.entities.get(&(c as u32)).map_or(TOP, |e| e.concept);
        let mut v = vec![c];
        v.extend(p.ancestors(concept).into_iter().map(u64::from));
        return v;
    }
    match u32::try_from(c) {
        Ok(c) => p.ancestors(c).into_iter().map(u64::from).collect(),
        Err(_) => vec![u64::from(TOP)],
    }
}

/// Bits of an event's outcome under the table's base distribution.
pub fn base_bits(_p: &Program, s: &Sizes, e: &Event) -> Bits {
    match e.table {
        Table::Tag => lg(4),
        Table::SpeakerRank => lg(SPEAKER_LOOKBACK as u64 + 1),
        Table::SpeakerName | Table::Gap => e.spell,
        Table::LineEmpty | Table::FragMore | Table::HasName => ONE_BIT,
        Table::Unused9 => 0,
        Table::RoleNext => lg(s.roles + 1),
        Table::Mode => lg(4),
        Table::Concept => lg(s.heads()),
        Table::FeatNext => lg(s.feat_values + 1),
        Table::RefRank => lg(REF_LOOKBACK as u64 + 3),
        Table::RootCat | Table::NewCat => e.spell,
        Table::Expand => lg(s.rules + 1),
        Table::Split => lg(s.labels()),
        Table::Infl => lg(s.infls + 1),
        Table::Lex => {
            if e.out & 0xffff_ffff_0000_0000 == TAG_NAME {
                lg(s.names + 1)
            } else {
                e.spell
            }
        }
        Table::Case => lg(5),
    }
}

/// Bits of one event under the program's stored counts.
pub fn event_bits(p: &Program, s: &Sizes, e: &Event) -> Bits {
    let mut bits: Bits = 0;
    for ctx in chain(p, e) {
        let Some(cc) = p.counts.tables.get(&(e.table as u8, ctx)) else {
            continue;
        };
        let (hit, b) = level_bits(cc, e.out);
        bits += b;
        if hit {
            return bits;
        }
    }
    bits + base_bits(p, s, e)
}

/// Bits at one stored context: `(true, bits)` if the outcome is stored,
/// else `(false, escape bits)`.
pub fn level_bits(cc: &CtxCounts, out: u64) -> (bool, Bits) {
    let total: u64 =
        cc.outs.values().map(|q| level_value(*q)).sum::<u64>() + level_value(cc.esc) + 1;
    match cc.outs.get(&out) {
        Some(q) => (true, neg_log(level_value(*q).max(1), total)),
        None => (false, neg_log(level_value(cc.esc) + 1, total)),
    }
}

// ---------------------------------------------------------------------------
// The category prior.

/// Bits of a category under the fixed prior: at each node, one bit for
/// atomic vs functor; an atom then names one of the atomic categories and
/// codes each applicable feature dimension (unspecified, a value, or for
/// lexical categories a variable); a functor codes its direction (1 bit),
/// then its result and argument.
pub fn cat_prior(p: &Program, c: &Cat) -> Bits {
    match c {
        Cat::Atom(a) => {
            let mut b = ONE_BIT + lg(p.atoms.len() as u64);
            for d in &p.cat_feats {
                if d.atoms.contains(&a.id) {
                    b += lg(d.values.len() as u64 + 2);
                }
            }
            b
        }
        Cat::Slash(s) => 2 * ONE_BIT + cat_prior(p, &s.res) + cat_prior(p, &s.arg),
    }
}

// ---------------------------------------------------------------------------
// Program bits.

/// Bits of a meaning term in the program code: each constructor costs
/// log2(7), then its payload.
pub fn term_bits(p: &Program, s: &Sizes, t: &Term) -> Bits {
    let k = lg(7);
    match t {
        Term::Var(i) => k + gamma(u64::from(*i) + 1),
        Term::Lam(b) => k + term_bits(p, s, b),
        Term::App(a, b) => k + term_bits(p, s, a) + term_bits(p, s, b),
        Term::Node(n) => {
            let mut b = k + lg(s.heads() + 3);
            if let Some(name) = &n.name {
                b += ONE_BIT + spell(name.as_bytes());
            } else {
                b += ONE_BIT;
            }
            b += gamma(n.feats.len() as u64 + 1);
            for (f, _) in &n.feats {
                let nv = p.feats.get(f).map_or(1, |x| x.values.len() as u64);
                b += lg(s.feats) + lg(nv);
            }
            b += gamma(n.roles.len() as u64 + 1);
            for (_, c) in &n.roles {
                b += lg(s.roles) + term_bits(p, s, c);
            }
            b
        }
        Term::Add(a, _, u) => k + lg(s.roles) + term_bits(p, s, a) + term_bits(p, s, u),
        Term::Feat(a, f, _) => {
            let nv = p.feats.get(f).map_or(1, |x| x.values.len() as u64);
            k + lg(s.feats) + lg(nv) + term_bits(p, s, a)
        }
    }
}

/// Program size, split into parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramBits {
    /// Ontology and knowledge base.
    pub ontology: Bits,
    /// Categories, rules and inflections.
    pub grammar: Bits,
    /// Lexical entries.
    pub lexicon: Bits,
    /// Stored counts.
    pub counts: Bits,
}

impl ProgramBits {
    /// Total.
    pub fn total(&self) -> Bits {
        self.ontology + self.grammar + self.lexicon + self.counts
    }
}

/// Bits of one lexical entry.
pub fn entry_bits(p: &Program, s: &Sizes, e: &crate::program::Entry) -> Bits {
    spell(e.form.as_bytes()) + cat_prior(p, &e.cat) + term_bits(p, s, &e.sem)
}

/// Bits of a rule instance.
pub fn rule_bits(p: &Program, r: &crate::program::RuleKind) -> Bits {
    lg(u64::from(crate::program::RuleKind::KINDS))
        + r.cats().iter().map(|c| cat_prior(p, c)).sum::<Bits>()
}

/// Bits of one stored context's counts: its context definition, the number
/// of outcomes, the escape level, and each outcome with its level.
pub fn ctx_bits(p: &Program, s: &Sizes, table: Table, ctx: [u64; 2], cc: &CtxCounts) -> Bits {
    let mut b = key_bits(p, s, table, 0, ctx[0]) + key_bits(p, s, table, 1, ctx[1]);
    b += gamma(cc.outs.len() as u64 + 1) + gamma(u64::from(cc.esc) + 1);
    for (o, q) in &cc.outs {
        b += key_bits(p, s, table, 2, *o) + gamma(u64::from(*q) + 1);
    }
    b
}

/// Bits to name one key field in the counts code.
pub fn key_bits(p: &Program, s: &Sizes, table: Table, field: usize, k: u64) -> Bits {
    if special_name(k).is_some() {
        return lg(16);
    }
    match table.key_kind(field) {
        KeyKind::None => 0,
        KeyKind::Int => gamma((k & 0xffff_ffff) + 1),
        KeyKind::Role => lg(s.roles + 2),
        KeyKind::Concept => lg(s.heads() + 2),
        KeyKind::Feat => lg(s.feats),
        KeyKind::FeatVal(_) => lg(s.feat_values + 1),
        KeyKind::Rule => lg(s.rules + s.infls + 2),
        KeyKind::Sym => match (table, field) {
            (Table::Lex, 2) => {
                let tag = k & 0xffff_ffff_0000_0000;
                if tag == TAG_ENTRY || tag == TAG_NAME {
                    lg(s.entries + s.names + 1)
                } else {
                    p.counts
                        .syms
                        .get(&k)
                        .map_or(lg(1 << 20), |w| spell(w.as_bytes()))
                }
            }
            (Table::Split, _) => lg(s.labels()),
            _ => p
                .counts
                .syms
                .get(&k)
                .map_or(lg(1 << 20), |text| sym_text_bits(p, s, table, field, text)),
        },
    }
}

fn sym_text_bits(p: &Program, s: &Sizes, table: Table, field: usize, text: &str) -> Bits {
    match sym_kind(table, field) {
        SymKind::Cat => Cat::parse(text, &p.cat_names()).map_or(lg(1 << 20), |c| cat_prior(p, &c)),
        SymKind::Term => Term::parse(text, p).map_or(lg(1 << 20), |t| term_bits(p, s, &t)),
        SymKind::Str => spell(text.as_bytes()),
        SymKind::FeatSet => lg(1 << 12),
    }
}

/// What a hashed symbol field holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymKind {
    /// A category.
    Cat,
    /// A meaning term.
    Term,
    /// A string.
    Str,
    /// A set of meaning features.
    FeatSet,
}

/// The kind of symbol in a `Sym` field.
pub fn sym_kind(table: Table, field: usize) -> SymKind {
    match (table, field) {
        (Table::Lex, 1) => SymKind::Term,
        (Table::Lex, 2) | (Table::Gap, 2) | (Table::SpeakerName, 2) => SymKind::Str,
        (Table::Infl, 1) => SymKind::FeatSet,
        _ => SymKind::Cat,
    }
}

/// The key of a printed symbol (used when reading `counts.tsv`).
pub fn sym_key_from_text(p: &Program, table: u8, field: usize, text: &str) -> Result<u64, String> {
    let t = Table::from_code(table).ok_or("bad table")?;
    Ok(match sym_kind(t, field) {
        SymKind::Cat => cat_key(&Cat::parse(text, &p.cat_names())?),
        SymKind::Term => sem_key(&Term::parse(text, p)?),
        SymKind::Str => str_key(text),
        SymKind::FeatSet => {
            let mut fs = Vec::new();
            for pair in text.split(',').map(str::trim).filter(|x| !x.is_empty()) {
                let (f, v) = pair.split_once('=').ok_or("bad feature set")?;
                let fid = p.feat_by_name(f).ok_or("unknown feature")?;
                let vid = crate::term::TermLookup::val_id(p, fid, v).ok_or("unknown value")?;
                fs.push((fid, vid));
            }
            featset_key(&fs)
        }
    })
}

/// The program's size in bits.
pub fn program_bits(p: &Program) -> ProgramBits {
    let s = Sizes::of(p);
    let mut out = ProgramBits::default();
    let lc = lg(s.concepts);
    for _ in p.concepts.values() {
        out.ontology += lc;
    }
    for _ in p.roles.values() {
        out.ontology += 2 * lc;
    }
    for f in p.feats.values() {
        out.ontology += lc + gamma(f.values.len() as u64);
    }
    for e in p.entities.values() {
        out.ontology += lc + spell(e.name.as_bytes());
    }
    for f in p.facts.values() {
        out.ontology += term_bits(p, &s, &f.term);
    }
    for f in p.inferences.values() {
        out.ontology += term_bits(p, &s, &f.premise) + term_bits(p, &s, &f.conclusion);
    }
    for _ in &p.atoms {
        out.grammar += 8 * ONE_BIT;
    }
    for d in &p.cat_feats {
        out.grammar += gamma(d.values.len() as u64)
            + gamma(d.atoms.len() as u64)
            + d.atoms.len() as u64 * lg(s.atoms);
    }
    for r in p.rules.values() {
        out.grammar += rule_bits(p, &r.kind);
    }
    for i in p.infls.values() {
        out.grammar += cat_prior(p, &i.cat) + gamma(i.feats.len() as u64 + 1);
        for (f, _) in &i.feats {
            let nv = p.feats.get(f).map_or(1, |x| x.values.len() as u64);
            out.grammar += lg(s.feats) + lg(nv);
        }
        out.grammar += spell(i.strip.as_bytes()) + spell(i.add.as_bytes());
    }
    for e in p.entries.values() {
        out.lexicon += entry_bits(p, &s, e);
    }
    for n in p.names.values() {
        out.lexicon += cat_prior(p, &n.cat) + lc;
    }
    for ((t, ctx), cc) in &p.counts.tables {
        if let Some(table) = Table::from_code(*t) {
            out.counts += ctx_bits(p, &s, table, *ctx, cc);
        }
    }
    // The number of items of each kind.
    for n in [
        s.concepts, s.roles, s.feats, s.entities, s.atoms, s.rules, s.infls, s.entries, s.names,
    ] {
        out.ontology += gamma(n + 1);
    }
    out
}

/// The outcome key of a feature value in `FeatNext`.
pub fn featval_key(f: u32, v: u16) -> u64 {
    (u64::from(f) << 16) | u64::from(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_round_trip_and_codes_are_sane() {
        for c in [1u64, 2, 16, 17, 100, 1000, 123_456] {
            let v = level_value(level_of(c));
            let err = (v as f64 - c as f64).abs() / c as f64;
            assert!(err < 0.05, "count {c} -> {v}");
        }
        assert_eq!(gamma(1), ONE_BIT);
        assert_eq!(gamma(4), 5 * ONE_BIT);
        // Spelling a letter costs a few bits; a space-heavy model makes `e`
        // cheaper than `q`.
        assert!(spell(b"e") < spell(b"q"));
        assert!(spell(b"") > 0);
    }
}
