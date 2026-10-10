//! The teacher: Opus reads passages and writes down what they mean, in the
//! compact notation of `docs/meaning-guide.md`. Annotations are validated
//! against the ontology, retried with the errors up to three times, and
//! stored raw with the ontology version, model and prompt they were written
//! with.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use sememe_engine::program::Program;
use sememe_engine::term::TermLookup;
use sememe_engine::text::Line;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::opus::{Opus, Tier};

/// Version of the teacher prompt (bumped when it changes).
pub const PROMPT_VERSION: u32 = 1;

/// A passage sent to the teacher: consecutive lines of one document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    /// Passage ID within a batch (`p0`, `p1`, ...).
    pub id: String,
    /// Document index in the training set.
    pub doc: usize,
    /// Index of the first line in the document.
    pub first_line: usize,
    /// The lines.
    pub lines: Vec<String>,
}

/// A head in the annotation notation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnnHead {
    /// A concept by name.
    Concept(String),
    /// The questioned thing.
    Query,
    /// The speaker.
    Speaker,
    /// The addressee.
    Addressee,
    /// Some other referent outside the passage.
    Other,
    /// A reference to an entity of the passage (a pronoun).
    Ref(String),
}

/// A node in the annotation notation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnNode {
    /// What it is.
    pub head: AnnHead,
    /// Words that express it.
    pub anchors: Vec<usize>,
    /// Entity ID.
    pub entity: Option<String>,
    /// Proper name.
    pub name: Option<String>,
    /// Features `(name, value, anchors)`.
    pub feats: Vec<(String, String, Vec<usize>)>,
    /// Roles `(name, anchors, child)`.
    pub roles: Vec<(String, Vec<usize>, AnnNode)>,
}

/// One annotated unit (fragment) of a line.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Unit {
    /// Passage ID.
    pub passage: String,
    /// Line within the passage.
    pub line: usize,
    /// First word (line-relative).
    pub start: usize,
    /// One past the last word.
    pub end: usize,
    /// The meaning in the notation.
    pub meaning: String,
}

/// A new concept proposed by the teacher.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewConcept {
    /// Name.
    pub name: String,
    /// Supertype.
    pub parent: String,
    /// Doc comment.
    pub doc: String,
}

/// A new role proposed by the teacher.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewRole {
    /// Name.
    pub name: String,
    /// Domain type.
    pub domain: String,
    /// Range type.
    pub range: String,
    /// Doc comment.
    pub doc: String,
}

/// A new meaning feature proposed by the teacher.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewFeature {
    /// Name.
    pub name: String,
    /// Domain type.
    pub domain: String,
    /// Values.
    pub values: Vec<String>,
    /// Doc comment.
    pub doc: String,
}

/// What one teacher call returns.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reply {
    /// New concepts.
    #[serde(default)]
    pub new_concepts: Vec<NewConcept>,
    /// New roles.
    #[serde(default)]
    pub new_roles: Vec<NewRole>,
    /// New features.
    #[serde(default)]
    pub new_features: Vec<NewFeature>,
    /// Units.
    #[serde(default)]
    pub units: Vec<Unit>,
}

/// A stored batch: request, raw reply, and how validation went.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    /// Round.
    pub round: u32,
    /// Batch number within the round.
    pub index: u32,
    /// Model.
    pub model: String,
    /// Prompt version.
    pub prompt_version: u32,
    /// Hash of the ontology text the teacher saw.
    pub ontology_hash: String,
    /// Passages sent.
    pub passages: Vec<Passage>,
    /// Raw replies (first answer, then retries).
    pub replies: Vec<Reply>,
    /// Units that validated.
    pub valid: Vec<Unit>,
    /// Units dropped after retries, with their last error.
    pub dropped: Vec<(Unit, String)>,
    /// Whether this batch is a second annotation of passages (agreement check).
    #[serde(default)]
    pub double: bool,
}

// ---------------------------------------------------------------------------
// The notation.

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(usize),
    Str(String),
    Sym(char),
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let mut it = s.chars().peekable();
    while let Some(&c) = it.peek() {
        if c.is_whitespace() {
            it.next();
        } else if c == '"' {
            it.next();
            let mut t = String::new();
            loop {
                match it.next() {
                    Some('"') => break,
                    Some(ch) => t.push(ch),
                    None => return Err("unterminated name string".into()),
                }
            }
            out.push(Tok::Str(t));
        } else if c.is_ascii_digit() {
            let mut n = 0usize;
            while let Some(&d) = it.peek() {
                match d.to_digit(10) {
                    Some(v) => {
                        n = n.saturating_mul(10).saturating_add(v as usize);
                        it.next();
                    }
                    None => break,
                }
            }
            out.push(Tok::Num(n));
        } else if c.is_alphabetic() || c == '_' || c == '@' {
            let mut t = String::new();
            while let Some(&d) = it.peek() {
                if d.is_alphanumeric() || d == '_' || d == '@' {
                    t.push(d);
                    it.next();
                } else {
                    break;
                }
            }
            out.push(Tok::Ident(t));
        } else {
            out.push(Tok::Sym(c));
            it.next();
        }
    }
    Ok(out)
}

struct P {
    toks: Vec<Tok>,
    pos: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn ident(&mut self) -> Result<String, String> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                Ok(s)
            }
            other => Err(format!(
                "expected a name at token {}, found {other:?}",
                self.pos
            )),
        }
    }
    fn anchors(&mut self) -> Result<Vec<usize>, String> {
        let mut v = Vec::new();
        if self.eat('^') {
            loop {
                match self.peek().cloned() {
                    Some(Tok::Num(n)) => {
                        self.pos += 1;
                        v.push(n);
                    }
                    other => return Err(format!("expected a word index after ^, found {other:?}")),
                }
                // `^1,2` continues only if a number follows the comma.
                if self.peek() == Some(&Tok::Sym(','))
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Num(_)))
                {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        Ok(v)
    }
    fn node(&mut self) -> Result<AnnNode, String> {
        let h = self.ident()?;
        let head = match h.as_str() {
            "@speaker" => AnnHead::Speaker,
            "@addressee" => AnnHead::Addressee,
            "@other" => AnnHead::Other,
            "Query" => AnnHead::Query,
            s if s.starts_with('e')
                && s.len() > 1
                && s.chars().skip(1).all(|c| c.is_ascii_digit()) =>
            {
                AnnHead::Ref(s.to_string())
            }
            s => AnnHead::Concept(s.to_string()),
        };
        let anchors = self.anchors()?;
        let mut entity = None;
        if self.eat('#') {
            entity = Some(self.ident()?);
        }
        let mut name = None;
        if let Some(Tok::Str(s)) = self.peek().cloned() {
            self.pos += 1;
            name = Some(s.to_lowercase());
        }
        let mut feats = Vec::new();
        if self.eat('[') {
            loop {
                let f = self.ident()?;
                if !self.eat('=') {
                    return Err(format!("expected `=` after feature {f}"));
                }
                let v = self.ident()?;
                let a = self.anchors()?;
                feats.push((f, v, a));
                if self.eat(']') {
                    break;
                }
                if !self.eat(',') {
                    return Err("expected `,` or `]` in features".into());
                }
            }
        }
        let mut roles = Vec::new();
        if self.eat('(') {
            loop {
                let r = self.ident()?;
                let a = self.anchors()?;
                if !self.eat(':') {
                    return Err(format!("expected `:` after role {r}"));
                }
                let child = self.node()?;
                roles.push((r, a, child));
                if self.eat(')') {
                    break;
                }
                if !self.eat(',') {
                    return Err("expected `,` or `)` in roles".into());
                }
            }
        }
        Ok(AnnNode {
            head,
            anchors,
            entity,
            name,
            feats,
            roles,
        })
    }
}

/// Parse a meaning in the notation.
pub fn parse_meaning(s: &str) -> Result<AnnNode, String> {
    let mut p = P {
        toks: lex(s)?,
        pos: 0,
    };
    let n = p.node()?;
    if p.pos != p.toks.len() {
        return Err(format!(
            "trailing text after the meaning at token {}",
            p.pos
        ));
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Validation.

/// The ontology as the teacher sees it: the program's plus this batch's
/// proposals.
pub struct Onto<'a> {
    /// The program.
    pub p: &'a Program,
    /// Proposed concepts (name -> parent).
    pub concepts: BTreeMap<String, String>,
    /// Proposed roles.
    pub roles: BTreeMap<String, (String, String)>,
    /// Proposed features (name -> values).
    pub feats: BTreeMap<String, Vec<String>>,
}

impl<'a> Onto<'a> {
    /// The program's ontology plus a reply's proposals.
    pub fn new(p: &'a Program, r: &Reply) -> Onto<'a> {
        Onto {
            p,
            concepts: r
                .new_concepts
                .iter()
                .map(|c| (c.name.clone(), c.parent.clone()))
                .collect(),
            roles: r
                .new_roles
                .iter()
                .map(|c| (c.name.clone(), (c.domain.clone(), c.range.clone())))
                .collect(),
            feats: r
                .new_features
                .iter()
                .map(|c| (c.name.clone(), c.values.clone()))
                .collect(),
        }
    }

    fn has_concept(&self, n: &str) -> bool {
        self.p.concept_by_name(n).is_some() || self.concepts.contains_key(n)
    }

    fn has_role(&self, n: &str) -> bool {
        self.p.role_by_name(n).is_some() || self.roles.contains_key(n)
    }

    fn has_value(&self, f: &str, v: &str) -> bool {
        if let Some(fid) = self.p.feat_by_name(f) {
            return self.p.val_id(fid, v).is_some();
        }
        self.feats
            .get(f)
            .is_some_and(|vs| vs.iter().any(|x| x == v))
    }

    /// Check proposals: parents, domains and ranges must exist.
    pub fn check_proposals(&self) -> Vec<String> {
        let mut errs = Vec::new();
        for (c, parent) in &self.concepts {
            if !self.has_concept(parent) {
                errs.push(format!("new concept {c}: unknown parent {parent}"));
            }
            if !is_name(c) {
                errs.push(format!(
                    "new concept name `{c}` must be UpperCamelCase letters and digits"
                ));
            }
        }
        for (r, (d, g)) in &self.roles {
            for t in [d, g] {
                if !self.has_concept(t) {
                    errs.push(format!("new role {r}: unknown type {t}"));
                }
            }
        }
        errs
    }
}

fn is_name(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && s.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Validate one unit against the ontology and its line. `entities` holds
/// the entity IDs defined so far in the passage (and is extended).
pub fn validate(
    onto: &Onto,
    u: &Unit,
    line: &Line,
    entities: &mut Vec<String>,
) -> Result<AnnNode, String> {
    let n = line.words.len();
    if u.start >= u.end || u.end > n {
        return Err(format!(
            "unit range {}..{} is outside the line's {} words",
            u.start, u.end, n
        ));
    }
    let bounds = line.boundaries();
    let long = u.end - u.start > sememe_engine::cost::MAX_FRAGMENT_WORDS;
    for b in [u.start, u.end] {
        if !long && !bounds.get(b).copied().unwrap_or(false) {
            return Err(format!(
                "unit boundary {b} is not at punctuation or the line's ends"
            ));
        }
    }
    let node = parse_meaning(&u.meaning)?;
    let mut errs = Vec::new();
    check_node(onto, &node, u, entities, &mut errs);
    match errs.first() {
        Some(_) => Err(errs.join("; ")),
        None => Ok(node),
    }
}

fn check_anchor(a: &[usize], u: &Unit, errs: &mut Vec<String>) {
    for &i in a {
        if i < u.start || i >= u.end {
            errs.push(format!(
                "anchor ^{i} is outside the unit's words {}..{}",
                u.start, u.end
            ));
        }
    }
}

fn check_node(
    onto: &Onto,
    n: &AnnNode,
    u: &Unit,
    entities: &mut Vec<String>,
    errs: &mut Vec<String>,
) {
    match &n.head {
        AnnHead::Concept(c) if !onto.has_concept(c) => {
            errs.push(format!("unknown concept {c} (propose it in new_concepts)"));
        }
        AnnHead::Ref(e) if !entities.contains(e) && n.entity.as_deref() != Some(e) => {
            errs.push(format!(
                "reference {e} to an entity not introduced earlier in the passage"
            ));
        }
        _ => {}
    }
    if let Some(e) = &n.entity
        && !entities.contains(e)
    {
        entities.push(e.clone());
    }
    check_anchor(&n.anchors, u, errs);
    for (f, v, a) in &n.feats {
        if !onto.has_value(f, v) {
            errs.push(format!("unknown feature value {f}={v}"));
        }
        check_anchor(a, u, errs);
    }
    for (r, a, c) in &n.roles {
        if !onto.has_role(r) {
            errs.push(format!("unknown role {r} (propose it in new_roles)"));
        }
        check_anchor(a, u, errs);
        check_node(onto, c, u, entities, errs);
    }
}

// ---------------------------------------------------------------------------
// The call.

/// The teacher's system prompt: its job plus the meaning guide.
pub fn system_prompt(guide: &str) -> String {
    format!(
        "You are the teacher for sememe, a system that learns a readable language model from text. \
         You read passages of English and write down what each sentence means, as a typed meaning \
         tree in the notation below, using the current ontology. Be consistent: the same idea gets \
         the same concept and the same structure every time. Follow the meaning guide exactly.\n\n{guide}\n\n\
         Output JSON matching the given schema: any new concepts, roles and features you need (each \
         with a one-line doc), and one unit per sentence-like fragment of each line. A unit covers \
         words start..end (end exclusive, line-relative indices); units of a line must not overlap \
         and must cover all its words; unit boundaries may only fall where the line lists a break. \
         Lines with no words get no units."
    )
}

/// The ontology listing for the prompt.
pub fn ontology_text(p: &Program) -> String {
    use sememe_engine::term::TermNames;
    let mut s = String::new();
    s.push_str("Concepts (Name < Parent: doc):\n");
    for c in p.concepts.values() {
        s.push_str(&format!(
            "  {} < {}: {}\n",
            c.name,
            p.concept_name(c.parent),
            c.meta.doc.lines().next().unwrap_or("")
        ));
    }
    s.push_str("Roles (name: Domain -> Range: doc):\n");
    for r in p.roles.values() {
        if r.meta.seed {
            continue;
        }
        s.push_str(&format!(
            "  {}: {} -> {}: {}\n",
            r.name,
            p.concept_name(r.domain),
            p.concept_name(r.range),
            r.meta.doc.lines().next().unwrap_or("")
        ));
    }
    s.push_str("Features (name on Domain = values: doc):\n");
    for f in p.feats.values() {
        s.push_str(&format!(
            "  {} on {} = {}: {}\n",
            f.name,
            p.concept_name(f.domain),
            f.values.join(" | "),
            f.meta.doc.lines().next().unwrap_or("")
        ));
    }
    s
}

/// How passages are shown to the teacher.
pub fn passages_text(passages: &[Passage]) -> String {
    let mut s = String::new();
    for p in passages {
        s.push_str(&format!("Passage {}:\n", p.id));
        for (li, raw) in p.lines.iter().enumerate() {
            let line = Line::parse(raw);
            let speaker = line
                .speaker
                .as_deref()
                .map(|x| format!(" (speaker {x})"))
                .unwrap_or_default();
            s.push_str(&format!(
                "  line {li}{speaker}: {}\n",
                serde_json::to_string(raw).unwrap_or_default()
            ));
            if line.words.is_empty() {
                s.push_str("    (no words)\n");
                continue;
            }
            let words: Vec<String> = line
                .words
                .iter()
                .enumerate()
                .map(|(i, w)| format!("{i} {}", w.written))
                .collect();
            s.push_str(&format!("    words: {}\n", words.join(" | ")));
            let breaks: Vec<String> = line
                .boundaries()
                .iter()
                .enumerate()
                .filter(|(i, b)| **b && *i > 0 && *i < line.words.len())
                .map(|(i, _)| i.to_string())
                .collect();
            if !breaks.is_empty() {
                s.push_str(&format!(
                    "    breaks allowed before word: {}\n",
                    breaks.join(", ")
                ));
            }
        }
    }
    s
}

/// The reply schema.
pub fn schema() -> Value {
    let concept = json!({"type": "object", "properties": {"name": {"type": "string"}, "parent": {"type": "string"}, "doc": {"type": "string"}}, "required": ["name", "parent", "doc"]});
    let role = json!({"type": "object", "properties": {"name": {"type": "string"}, "domain": {"type": "string"}, "range": {"type": "string"}, "doc": {"type": "string"}}, "required": ["name", "domain", "range", "doc"]});
    let feat = json!({"type": "object", "properties": {"name": {"type": "string"}, "domain": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}, "doc": {"type": "string"}}, "required": ["name", "domain", "values", "doc"]});
    let unit = json!({"type": "object", "properties": {"passage": {"type": "string"}, "line": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "meaning": {"type": "string"}}, "required": ["passage", "line", "start", "end", "meaning"]});
    json!({"type": "object", "properties": {
        "new_concepts": {"type": "array", "items": concept},
        "new_roles": {"type": "array", "items": role},
        "new_features": {"type": "array", "items": feat},
        "units": {"type": "array", "items": unit}},
        "required": ["new_concepts", "new_roles", "new_features", "units"]})
}

fn hash_text(s: &str) -> String {
    format!("{:016x}", sememe_engine::cost::str_key(s))
}

/// Annotate passages: one call, then up to three retries for the units that
/// fail validation (with their errors). Returns the stored batch.
#[allow(clippy::too_many_arguments)]
pub fn annotate(
    opus: &Opus,
    p: &Program,
    guide: &str,
    round: u32,
    index: u32,
    passages: Vec<Passage>,
    double: bool,
    store_dir: &Path,
) -> Result<Batch> {
    let onto_text = ontology_text(p);
    let system = system_prompt(guide);
    let prompt = format!(
        "## Current ontology\n{onto_text}\n## Passages\n{}",
        passages_text(&passages)
    );
    let mut batch = Batch {
        round,
        index,
        model: crate::opus::MODEL.into(),
        prompt_version: PROMPT_VERSION,
        ontology_hash: hash_text(&onto_text),
        passages: passages.clone(),
        replies: Vec::new(),
        valid: Vec::new(),
        dropped: Vec::new(),
        double,
    };
    let reply = opus.structured(
        Tier::Teacher,
        round,
        "annotate",
        &system,
        &prompt,
        &schema(),
        Duration::from_secs(600),
    )?;
    let mut r: Reply = serde_json::from_value(reply.value)?;
    let mut pending = check_reply(p, &passages, &r, &mut batch);
    batch.replies.push(r.clone());
    for attempt in 0..3 {
        if pending.is_empty() {
            break;
        }
        let errors: Vec<String> = pending
            .iter()
            .map(|(u, e)| {
                format!(
                    "passage {} line {} words {}..{}: `{}` -> {}",
                    u.passage, u.line, u.start, u.end, u.meaning, e
                )
            })
            .collect();
        let retry_prompt = format!(
            "{prompt}\n## Fix these units\nYour previous answer had these errors. Return corrected units for exactly these \
             lines (all units of each listed line), plus any new concepts, roles or features they need. Previously proposed \
             new types: {}.\n{}",
            serde_json::to_string(&(&r.new_concepts, &r.new_roles, &r.new_features))
                .unwrap_or_default(),
            errors.join("\n")
        );
        let reply = match opus.structured(
            Tier::Teacher,
            round,
            "annotate-retry",
            &system,
            &retry_prompt,
            &schema(),
            Duration::from_secs(600),
        ) {
            Ok(x) => x,
            Err(_) => break,
        };
        let mut fix: Reply = serde_json::from_value(reply.value)?;
        // Keep earlier proposals too.
        let mut merged = r.clone();
        merged.new_concepts.append(&mut fix.new_concepts);
        merged.new_roles.append(&mut fix.new_roles);
        merged.new_features.append(&mut fix.new_features);
        merged.units = fix.units;
        let retried: Vec<(String, usize)> = pending
            .iter()
            .map(|(u, _)| (u.passage.clone(), u.line))
            .collect();
        merged
            .units
            .retain(|u| retried.contains(&(u.passage.clone(), u.line)));
        let before = pending.len();
        pending = check_reply(p, &passages, &merged, &mut batch);
        batch.replies.push(merged.clone());
        r.new_concepts = merged.new_concepts;
        r.new_roles = merged.new_roles;
        r.new_features = merged.new_features;
        let _ = (attempt, before);
    }
    batch.dropped = pending;
    save_batch(store_dir, &batch)?;
    Ok(batch)
}

/// Validate a reply's units; valid ones go into the batch, invalid ones are
/// returned with their errors. A line counts as valid only if all its units
/// are.
fn check_reply(
    p: &Program,
    passages: &[Passage],
    r: &Reply,
    batch: &mut Batch,
) -> Vec<(Unit, String)> {
    let onto = Onto::new(p, r);
    let mut bad = Vec::new();
    let prop_errs = onto.check_proposals();
    for pass in passages {
        let mut entities: Vec<String> = Vec::new();
        // Entities from already-valid units of this passage come first.
        for (li, raw) in pass.lines.iter().enumerate() {
            let line = Line::parse(raw);
            let units: Vec<&Unit> = r
                .units
                .iter()
                .filter(|u| u.passage == pass.id && u.line == li)
                .collect();
            if units.is_empty() {
                continue;
            }
            let mut line_errs = Vec::new();
            let mut ok_units = Vec::new();
            for u in &units {
                match validate(&onto, u, &line, &mut entities) {
                    Ok(ann) => {
                        let errs = crate::genlex::target_meaning(&typed, &ann)
                            .map(|t| typed.type_errors(&t))
                            .unwrap_or_default();
                        if errs.is_empty() {
                            ok_units.push((*u).clone());
                        } else {
                            line_errs
                                .push(((*u).clone(), format!("type error: {}", errs.join("; "))));
                        }
                    }
                    Err(e) => line_errs.push(((*u).clone(), e)),
                }
            }
            if line_errs.is_empty() && prop_errs.is_empty() {
                batch
                    .valid
                    .retain(|v| !(v.passage == pass.id && v.line == li));
                batch.valid.extend(ok_units);
            } else {
                for (u, e) in line_errs {
                    bad.push((u, e));
                }
                if !prop_errs.is_empty() {
                    for u in ok_units {
                        bad.push((u, prop_errs.join("; ")));
                    }
                }
            }
        }
    }
    bad
}

fn save_batch(dir: &Path, b: &Batch) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!(
        "r{:05}-b{:03}{}.json",
        b.round,
        b.index,
        if b.double { "-double" } else { "" }
    ));
    std::fs::write(&path, serde_json::to_string(b)?)?;
    Ok(path)
}

/// Read every stored batch in a directory, in name order.
pub fn load_batches(dir: &Path) -> Result<Vec<Batch>> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect(),
        Err(_) => return Ok(Vec::new()),
    };
    paths.sort();
    let mut out = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(&p)?;
        out.push(serde_json::from_str(&text)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notation_parses() {
        let n = parse_meaning(
            "See^2#e1[time=past^0, mood=ask^0](experiencer: @addressee^1, stimulus: Dog^5#e2[def=yes^3](attr: Big^4#e3))",
        )
        .unwrap();
        assert_eq!(n.head, AnnHead::Concept("See".into()));
        assert_eq!(n.anchors, vec![2]);
        assert_eq!(n.feats.len(), 2);
        assert_eq!(n.roles[1].2.roles[0].2.anchors, vec![4]);
        let m = parse_meaning(
            "Put^1[time=past](agent: e1^0, patient: e2^2, goal^3: Box^5#e5[def=yes^4])",
        )
        .unwrap();
        assert_eq!(m.roles[0].2.head, AnnHead::Ref("e1".into()));
        assert_eq!(m.roles[2].1, vec![3]);
        let k = parse_meaning("City^0,1#e1\"new york\"").unwrap();
        assert_eq!(k.anchors, vec![0, 1]);
        assert_eq!(k.name.as_deref(), Some("new york"));
    }
}
