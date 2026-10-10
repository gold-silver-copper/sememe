//! Meanings and the small typed lambda calculus that builds them.
//!
//! A meaning is a tree of [`Node`]s: a head (usually a concept), optional
//! name, features (`time=past`) and role children (`agent: ...`). Lexical
//! entries carry lambda terms over these trees: `\y x. Eat(agent: x,
//! patient: y)` for a transitive verb, `\x. x + attr: Big` for an adjective,
//! `\x. x + def=yes` for a determiner. Variables use de Bruijn indices so
//! that equal meanings are structurally equal.

use std::sync::Arc;

/// Who a pronoun-like reference points at, relative to the conversation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Deixis {
    /// The current speaker ("I").
    Speaker,
    /// The one spoken to ("you").
    Addressee,
    /// Anyone or anything else already in the context ("she", "it").
    Other,
}

/// What a meaning node is.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Head {
    /// An instance of a concept from the ontology.
    Concept(u32),
    /// A meaning the program cannot explain yet (an unknown word).
    Unknown,
    /// Something left unsaid (an argument a fragment never supplies).
    Missing,
    /// A reference back to something in the context.
    Ref(Deixis),
    /// The thing a question asks for.
    Query,
    /// A named individual from the global knowledge base.
    Entity(u32),
}

/// One node of a meaning tree.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Node {
    /// What the node is.
    pub head: Head,
    /// A proper name, lowercased, if the node is named in the text.
    pub name: Option<Arc<str>>,
    /// Feature values `(feature, value)`, sorted by feature.
    pub feats: Vec<(u32, u16)>,
    /// Role children `(role, filler)`, sorted by role (stable).
    pub roles: Vec<(u32, Term)>,
}

impl Node {
    /// A bare node with this head.
    pub fn new(head: Head) -> Node {
        Node {
            head,
            name: None,
            feats: Vec::new(),
            roles: Vec::new(),
        }
    }
}

/// A lambda term over meaning trees.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    /// A bound variable (de Bruijn index).
    Var(u32),
    /// A function.
    Lam(Arc<Term>),
    /// Application.
    App(Arc<Term>, Arc<Term>),
    /// A meaning node.
    Node(Arc<Node>),
    /// Add a role child to the head node of a term.
    Add(Arc<Term>, u32, Arc<Term>),
    /// Set a feature on the head node of a term.
    Feat(Arc<Term>, u32, u16),
}

/// Names for printing terms.
pub trait TermNames {
    /// Concept name.
    fn concept_name(&self, id: u32) -> String;
    /// Role name.
    fn role_name(&self, id: u32) -> String;
    /// Feature name.
    fn feat_name(&self, id: u32) -> String;
    /// Feature value name.
    fn val_name(&self, feat: u32, val: u16) -> String;
    /// Global entity name.
    fn entity_name(&self, id: u32) -> String;
}

/// Name lookups for parsing terms.
pub trait TermLookup {
    /// Concept by name.
    fn concept_id(&self, name: &str) -> Option<u32>;
    /// Role by name.
    fn role_id(&self, name: &str) -> Option<u32>;
    /// Feature by name.
    fn feat_id(&self, name: &str) -> Option<u32>;
    /// Feature value by name.
    fn val_id(&self, feat: u32, name: &str) -> Option<u16>;
    /// Global entity by name.
    fn entity_id(&self, name: &str) -> Option<u32>;
}

/// The kind of semantic contribution a functor makes, used to code how a
/// derivation splits its meaning between two children.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum LabelKind {
    /// The argument fills role `name` of the functor's head.
    Arg,
    /// The functor adds a role-`name` child to its argument.
    Mod,
    /// The functor sets feature `name` on its argument.
    Feat,
    /// The functor passes its argument through unchanged.
    Id,
    /// The functor applies its argument (type raising, question words).
    Ho,
    /// The functor ignores its argument.
    Vac,
}

/// How a binary derivation step splits the meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Label {
    /// The kind of split.
    pub kind: LabelKind,
    /// The role or feature involved (0 when not applicable).
    pub name: u32,
}

const FUEL: u32 = 256;
const MAX_SIZE: usize = 400;

impl Term {
    /// A term for a node.
    pub fn node(n: Node) -> Term {
        Term::Node(Arc::new(n))
    }

    /// A bare node term with this head.
    pub fn head(h: Head) -> Term {
        Term::node(Node::new(h))
    }

    /// Wrap in `k` lambdas.
    pub fn lams(mut self, k: usize) -> Term {
        for _ in 0..k {
            self = Term::Lam(Arc::new(self));
        }
        self
    }

    /// Number of leading lambdas.
    pub fn lam_depth(&self) -> usize {
        match self {
            Term::Lam(b) => 1 + b.lam_depth(),
            _ => 0,
        }
    }

    /// The body under all leading lambdas.
    pub fn body(&self) -> &Term {
        match self {
            Term::Lam(b) => b.body(),
            t => t,
        }
    }

    /// The node if this term is a node.
    pub fn as_node(&self) -> Option<&Node> {
        match self {
            Term::Node(n) => Some(n),
            _ => None,
        }
    }

    /// Size in constructors.
    pub fn size(&self) -> usize {
        match self {
            Term::Var(_) => 1,
            Term::Lam(b) => 1 + b.size(),
            Term::App(f, a) => 1 + f.size() + a.size(),
            Term::Node(n) => 1 + n.roles.iter().map(|(_, t)| t.size()).sum::<usize>(),
            Term::Add(t, _, u) => 1 + t.size() + u.size(),
            Term::Feat(t, _, _) => 1 + t.size(),
        }
    }

    /// True if the term is a closed meaning tree (no variables, lambdas,
    /// applications or pending operations).
    pub fn is_tree(&self) -> bool {
        match self {
            Term::Node(n) => n.roles.iter().all(|(_, t)| t.is_tree()),
            _ => false,
        }
    }

    fn shift(&self, d: i64, cutoff: u32) -> Term {
        match self {
            Term::Var(i) => {
                if *i >= cutoff {
                    let v = i64::from(*i) + d;
                    Term::Var(u32::try_from(v.max(0)).unwrap_or(0))
                } else {
                    Term::Var(*i)
                }
            }
            Term::Lam(b) => Term::Lam(Arc::new(b.shift(d, cutoff + 1))),
            Term::App(f, a) => {
                Term::App(Arc::new(f.shift(d, cutoff)), Arc::new(a.shift(d, cutoff)))
            }
            Term::Node(n) => {
                if n.roles.is_empty() {
                    return self.clone();
                }
                let mut m = (**n).clone();
                for (_, t) in &mut m.roles {
                    *t = t.shift(d, cutoff);
                }
                Term::node(m)
            }
            Term::Add(t, r, u) => Term::Add(
                Arc::new(t.shift(d, cutoff)),
                *r,
                Arc::new(u.shift(d, cutoff)),
            ),
            Term::Feat(t, f, v) => Term::Feat(Arc::new(t.shift(d, cutoff)), *f, *v),
        }
    }

    /// Substitute `s` for variable `j`.
    fn subst(&self, j: u32, s: &Term) -> Term {
        match self {
            Term::Var(i) => {
                if *i == j {
                    s.clone()
                } else {
                    Term::Var(*i)
                }
            }
            Term::Lam(b) => Term::Lam(Arc::new(b.subst(j + 1, &s.shift(1, 0)))),
            Term::App(f, a) => Term::App(Arc::new(f.subst(j, s)), Arc::new(a.subst(j, s))),
            Term::Node(n) => {
                if n.roles.is_empty() {
                    return self.clone();
                }
                let mut m = (**n).clone();
                for (_, t) in &mut m.roles {
                    *t = t.subst(j, s);
                }
                Term::node(m)
            }
            Term::Add(t, r, u) => Term::Add(Arc::new(t.subst(j, s)), *r, Arc::new(u.subst(j, s))),
            Term::Feat(t, f, v) => Term::Feat(Arc::new(t.subst(j, s)), *f, *v),
        }
    }

    fn beta(body: &Term, arg: &Term) -> Term {
        body.subst(0, &arg.shift(1, 0)).shift(-1, 0)
    }

    /// Function composition `\z. f (g z)`, normalized.
    pub fn compose(f: &Term, g: &Term) -> Option<Term> {
        let inner = Term::App(Arc::new(g.shift(1, 0)), Arc::new(Term::Var(0)));
        Term::Lam(Arc::new(Term::App(
            Arc::new(f.shift(1, 0)),
            Arc::new(inner),
        )))
        .normalize()
    }

    /// Type raising `\f. f a`.
    pub fn raise(a: &Term) -> Term {
        Term::Lam(Arc::new(Term::App(
            Arc::new(Term::Var(0)),
            Arc::new(a.shift(1, 0)),
        )))
    }

    /// Coordination of two terms of the same type with `k` arguments:
    /// `\x1..xk. conj + op: (l x1..xk) + op: (r x1..xk)`.
    pub fn coordinate(conj: &Term, l: &Term, r: &Term, k: usize, op: u32) -> Option<Term> {
        let k32 = u32::try_from(k).ok()?;
        let applied = |t: &Term| {
            let mut a = t.shift(i64::from(k32), 0);
            for i in (0..k32).rev() {
                a = Term::App(Arc::new(a), Arc::new(Term::Var(i)));
            }
            a
        };
        let body = Term::Add(
            Arc::new(Term::Add(
                Arc::new(conj.shift(i64::from(k32), 0)),
                op,
                Arc::new(applied(l)),
            )),
            op,
            Arc::new(applied(r)),
        );
        body.lams(k).normalize()
    }

    /// The node a leaf meaning contributes: the body node of a head entry,
    /// or the child a modifier adds.
    pub fn leaf_head(&self) -> Option<&Node> {
        match self.body() {
            Term::Node(n) => Some(n),
            Term::Add(_, _, u) => u.as_node(),
            _ => None,
        }
    }

    /// The same leaf meaning with features added to its contributed node.
    pub fn with_head_feats(&self, feats: &[(u32, u16)]) -> Option<Term> {
        fn go(t: &Term, feats: &[(u32, u16)]) -> Option<Term> {
            match t {
                Term::Lam(b) => Some(Term::Lam(Arc::new(go(b, feats)?))),
                Term::Node(n) => Some(Term::node(add_feats(n, feats)?)),
                Term::Add(b, r, u) => {
                    let n = u.as_node()?;
                    Some(Term::Add(
                        b.clone(),
                        *r,
                        Arc::new(Term::node(add_feats(n, feats)?)),
                    ))
                }
                _ => None,
            }
        }
        go(self, feats)
    }

    /// The same leaf meaning with these features removed from its
    /// contributed node.
    pub fn without_head_feats(&self, feats: &[(u32, u16)]) -> Term {
        fn go(t: &Term, feats: &[(u32, u16)]) -> Term {
            match t {
                Term::Lam(b) => Term::Lam(Arc::new(go(b, feats))),
                Term::Node(n) => {
                    let mut m = (**n).clone();
                    m.feats.retain(|f| !feats.contains(f));
                    Term::node(m)
                }
                Term::Add(b, r, u) => match u.as_node() {
                    Some(n) => {
                        let mut m = n.clone();
                        m.feats.retain(|f| !feats.contains(f));
                        Term::Add(b.clone(), *r, Arc::new(Term::node(m)))
                    }
                    None => t.clone(),
                },
                _ => t.clone(),
            }
        }
        go(self, feats)
    }

    /// Visit every node in the term (pre-order).
    pub fn for_each_node(&self, f: &mut dyn FnMut(&Node)) {
        match self {
            Term::Var(_) => {}
            Term::Lam(b) => b.for_each_node(f),
            Term::App(a, b) | Term::Add(a, _, b) => {
                a.for_each_node(f);
                b.for_each_node(f);
            }
            Term::Node(n) => {
                f(n);
                for (_, c) in &n.roles {
                    c.for_each_node(f);
                }
            }
            Term::Feat(a, _, _) => a.for_each_node(f),
        }
    }

    /// Apply this term to an argument and normalize.
    pub fn apply(&self, arg: &Term) -> Option<Term> {
        Term::App(Arc::new(self.clone()), Arc::new(arg.clone())).normalize()
    }

    /// Normal form: beta-reduce and perform `Add`/`Feat` on nodes. `None`
    /// if the term is ill-formed (adding to a function, a feature set twice)
    /// or too large.
    pub fn normalize(&self) -> Option<Term> {
        let mut fuel = FUEL;
        let t = norm(self, &mut fuel)?;
        (t.size() <= MAX_SIZE).then_some(t)
    }

    /// Classify a normal-form functor term by what it does with its first
    /// (outermost) argument.
    pub fn classify(&self) -> Label {
        let Term::Lam(body) = self else {
            return Label {
                kind: LabelKind::Vac,
                name: 0,
            };
        };
        // The outermost binder is Var(d) at lambda depth d inside `body`.
        let mut depth = 0u32;
        let mut cur: &Term = body;
        while let Term::Lam(b) = cur {
            depth += 1;
            cur = b;
        }
        if *cur == Term::Var(depth) {
            return Label {
                kind: LabelKind::Id,
                name: 0,
            };
        }
        find_use(cur, depth).unwrap_or(Label {
            kind: LabelKind::Vac,
            name: 0,
        })
    }

    /// Print with names.
    pub fn show(&self, names: &dyn TermNames) -> String {
        let mut out = String::new();
        write_term(self, names, 0, &mut out, Prec::Top);
        out
    }

    /// Parse a term written in the program's text form.
    pub fn parse(text: &str, look: &dyn TermLookup) -> Result<Term, String> {
        let toks = lex(text)?;
        let mut p = Parser {
            toks,
            pos: 0,
            look,
            vars: Vec::new(),
        };
        let t = p.term()?;
        if p.pos != p.toks.len() {
            return Err(format!("trailing tokens in term `{text}`"));
        }
        Ok(t)
    }
}

/// The canonical order of a node's children: by role, then by the child's
/// meaning key (so equal meanings are equal however they were built).
pub fn sort_roles(roles: &mut [(u32, Term)]) {
    roles.sort_by_cached_key(|(r, c)| (*r, crate::cost::meaning_key(c)));
}

fn add_feats(n: &Node, feats: &[(u32, u16)]) -> Option<Node> {
    let mut m = n.clone();
    for (f, v) in feats {
        if m.feats.iter().any(|(x, _)| x == f) {
            return None;
        }
        let at = m.feats.partition_point(|(x, _)| x < f);
        m.feats.insert(at, (*f, *v));
    }
    Some(m)
}

/// Find how variable `v` is used in normal-form `t`.
fn find_use(t: &Term, v: u32) -> Option<Label> {
    match t {
        Term::Var(_) => None,
        Term::Lam(b) => find_use(b, v + 1),
        Term::App(f, a) => {
            if **f == Term::Var(v) || head_var(f) == Some(v) {
                return Some(Label {
                    kind: LabelKind::Ho,
                    name: 0,
                });
            }
            find_use(f, v).or_else(|| find_use(a, v))
        }
        Term::Node(n) => {
            for (r, c) in &n.roles {
                if *c == Term::Var(v) {
                    return Some(Label {
                        kind: LabelKind::Arg,
                        name: *r,
                    });
                }
            }
            n.roles.iter().find_map(|(_, c)| find_use(c, v))
        }
        Term::Add(base, r, u) => {
            if base_var(base) == Some(v) {
                return Some(Label {
                    kind: LabelKind::Mod,
                    name: *r,
                });
            }
            if **u == Term::Var(v) {
                return Some(Label {
                    kind: LabelKind::Arg,
                    name: *r,
                });
            }
            find_use(base, v).or_else(|| find_use(u, v))
        }
        Term::Feat(base, f, _) => {
            if base_var(base) == Some(v) {
                return Some(Label {
                    kind: LabelKind::Feat,
                    name: *f,
                });
            }
            find_use(base, v)
        }
    }
}

fn head_var(t: &Term) -> Option<u32> {
    match t {
        Term::Var(i) => Some(*i),
        Term::App(f, _) => head_var(f),
        _ => None,
    }
}

/// The variable at the base of a chain of `Add`/`Feat` (looking through
/// applications of it, as in a lifted modifier `\P z. (P z) + r: u`).
fn base_var(t: &Term) -> Option<u32> {
    match t {
        Term::Var(i) => Some(*i),
        Term::Add(b, _, _) | Term::Feat(b, _, _) => base_var(b),
        Term::App(f, _) => head_var(f),
        _ => None,
    }
}

fn norm(t: &Term, fuel: &mut u32) -> Option<Term> {
    match t {
        Term::Var(_) => Some(t.clone()),
        Term::Lam(b) => Some(Term::Lam(Arc::new(norm(b, fuel)?))),
        Term::App(f, a) => {
            let f = norm(f, fuel)?;
            if let Term::Lam(body) = &f {
                *fuel = fuel.checked_sub(1)?;
                let reduced = Term::beta(body, a);
                norm(&reduced, fuel)
            } else {
                Some(Term::App(Arc::new(f), Arc::new(norm(a, fuel)?)))
            }
        }
        Term::Node(n) => {
            if n.roles.is_empty() {
                return Some(t.clone());
            }
            let mut m = (**n).clone();
            for (_, c) in &mut m.roles {
                *c = norm(c, fuel)?;
            }
            // Children filled in by substitution must return to the
            // canonical order when a role repeats.
            if m.roles
                .windows(2)
                .any(|w| matches!(w, [a, b] if a.0 == b.0))
            {
                sort_roles(&mut m.roles);
            }
            Some(Term::node(m))
        }
        Term::Add(base, r, u) => {
            let base = norm(base, fuel)?;
            let u = norm(u, fuel)?;
            match base {
                Term::Node(n) => {
                    let mut m = (*n).clone();
                    let k = crate::cost::meaning_key(&u);
                    let at = m
                        .roles
                        .partition_point(|(x, c)| (*x, crate::cost::meaning_key(c)) <= (*r, k));
                    m.roles.insert(at, (*r, u));
                    Some(Term::node(m))
                }
                Term::Lam(_) => None,
                other => Some(Term::Add(Arc::new(other), *r, Arc::new(u))),
            }
        }
        Term::Feat(base, f, v) => {
            let base = norm(base, fuel)?;
            match base {
                Term::Node(n) => {
                    if n.feats.iter().any(|(x, _)| x == f) {
                        return None;
                    }
                    let mut m = (*n).clone();
                    let at = m.feats.partition_point(|(x, _)| *x < *f);
                    m.feats.insert(at, (*f, *v));
                    Some(Term::node(m))
                }
                Term::Lam(_) => None,
                other => Some(Term::Feat(Arc::new(other), *f, *v)),
            }
        }
    }
}

const VAR_NAMES: [&str; 8] = ["x", "y", "z", "w", "v", "u", "s", "r"];

fn var_name(k: usize) -> String {
    VAR_NAMES
        .get(k)
        .map_or_else(|| format!("x{k}"), |s| (*s).to_string())
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    Top,
    Add,
    App,
    Atom,
}

fn write_term(t: &Term, names: &dyn TermNames, depth: usize, out: &mut String, prec: Prec) {
    match t {
        Term::Var(i) => {
            let k = depth.checked_sub(1 + *i as usize);
            out.push_str(&k.map_or_else(|| format!("free{i}"), var_name));
        }
        Term::Lam(_) => {
            if prec > Prec::Top {
                out.push('(');
            }
            out.push('\\');
            let mut cur = t;
            let mut d = depth;
            let mut first = true;
            while let Term::Lam(b) = cur {
                if !first {
                    out.push(' ');
                }
                first = false;
                out.push_str(&var_name(d));
                d += 1;
                cur = b;
            }
            out.push_str(". ");
            write_term(cur, names, d, out, Prec::Top);
            if prec > Prec::Top {
                out.push(')');
            }
        }
        Term::App(f, a) => {
            if prec > Prec::App {
                out.push('(');
            }
            write_term(f, names, depth, out, Prec::App);
            out.push(' ');
            write_term(a, names, depth, out, Prec::Atom);
            if prec > Prec::App {
                out.push(')');
            }
        }
        Term::Node(n) => write_node(n, names, depth, out),
        Term::Add(b, r, u) => {
            if prec > Prec::Add {
                out.push('(');
            }
            write_term(b, names, depth, out, Prec::Add);
            out.push_str(" + ");
            out.push_str(&names.role_name(*r));
            out.push_str(": ");
            write_term(u, names, depth, out, Prec::App);
            if prec > Prec::Add {
                out.push(')');
            }
        }
        Term::Feat(b, f, v) => {
            if prec > Prec::Add {
                out.push('(');
            }
            write_term(b, names, depth, out, Prec::Add);
            out.push_str(" + ");
            out.push_str(&names.feat_name(*f));
            out.push('=');
            out.push_str(&names.val_name(*f, *v));
            if prec > Prec::Add {
                out.push(')');
            }
        }
    }
}

fn write_node(n: &Node, names: &dyn TermNames, depth: usize, out: &mut String) {
    match &n.head {
        Head::Concept(c) => out.push_str(&names.concept_name(*c)),
        Head::Unknown => out.push_str("Unknown"),
        Head::Missing => out.push_str("Missing"),
        Head::Ref(Deixis::Speaker) => out.push_str("@speaker"),
        Head::Ref(Deixis::Addressee) => out.push_str("@addressee"),
        Head::Ref(Deixis::Other) => out.push_str("@other"),
        Head::Query => out.push_str("Query"),
        Head::Entity(e) => {
            out.push('#');
            out.push_str(&names.entity_name(*e));
        }
    }
    if let Some(name) = &n.name {
        out.push_str(" \"");
        out.push_str(name);
        out.push('"');
    }
    if !n.feats.is_empty() {
        out.push('[');
        for (i, (f, v)) in n.feats.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&names.feat_name(*f));
            out.push('=');
            out.push_str(&names.val_name(*f, *v));
        }
        out.push(']');
    }
    if !n.roles.is_empty() {
        out.push('(');
        for (i, (r, c)) in n.roles.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&names.role_name(*r));
            out.push_str(": ");
            write_term(c, names, depth, out, Prec::Top);
        }
        out.push(')');
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Sym(char),
}

fn lex(text: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut it = text.chars().peekable();
    while let Some(&c) = it.peek() {
        if c.is_whitespace() {
            it.next();
        } else if c == '"' {
            it.next();
            let mut s = String::new();
            loop {
                match it.next() {
                    Some('"') => break,
                    Some('\\') => {
                        if let Some(n) = it.next() {
                            s.push(n);
                        }
                    }
                    Some(ch) => s.push(ch),
                    None => return Err("unterminated string".into()),
                }
            }
            toks.push(Tok::Str(s));
        } else if c.is_alphanumeric() || c == '_' || c == '@' || c == '#' {
            let mut s = String::new();
            while let Some(&ch) = it.peek() {
                if ch.is_alphanumeric()
                    || ch == '_'
                    || ch == '@'
                    || ch == '#'
                    || ch == '-'
                    || ch == '\''
                {
                    s.push(ch);
                    it.next();
                } else {
                    break;
                }
            }
            toks.push(Tok::Ident(s));
        } else {
            toks.push(Tok::Sym(c));
            it.next();
        }
    }
    Ok(toks)
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    look: &'a dyn TermLookup,
    vars: Vec<String>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn peek_at(&self, k: usize) -> Option<&Tok> {
        self.toks.get(self.pos + k)
    }

    fn eat_sym(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, c: char) -> Result<(), String> {
        if self.eat_sym(c) {
            Ok(())
        } else {
            Err(format!("expected `{c}` at token {}", self.pos))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                Ok(s)
            }
            other => Err(format!("expected a name, found {other:?}")),
        }
    }

    fn term(&mut self) -> Result<Term, String> {
        if self.eat_sym('\\') {
            let mut bound = 0;
            while let Some(Tok::Ident(_)) = self.peek() {
                let v = self.ident()?;
                self.vars.push(v);
                bound += 1;
            }
            self.expect_sym('.')?;
            let body = self.term()?;
            for _ in 0..bound {
                self.vars.pop();
            }
            return Ok(body.lams(bound));
        }
        self.add()
    }

    fn add(&mut self) -> Result<Term, String> {
        let mut t = self.app()?;
        while self.eat_sym('+') {
            let name = self.ident()?;
            if self.eat_sym(':') {
                let r = self
                    .look
                    .role_id(&name)
                    .ok_or_else(|| format!("unknown role `{name}`"))?;
                let u = self.app()?;
                t = Term::Add(Arc::new(t), r, Arc::new(u));
            } else {
                self.expect_sym('=')?;
                let f = self
                    .look
                    .feat_id(&name)
                    .ok_or_else(|| format!("unknown feature `{name}`"))?;
                let vname = self.ident()?;
                let v = self
                    .look
                    .val_id(f, &vname)
                    .ok_or_else(|| format!("unknown value `{vname}` of `{name}`"))?;
                t = Term::Feat(Arc::new(t), f, v);
            }
        }
        Ok(t)
    }

    fn app(&mut self) -> Result<Term, String> {
        let mut t = self.atom()?;
        while let Some(Tok::Ident(_)) | Some(Tok::Sym('(')) = self.peek() {
            let a = self.atom()?;
            t = Term::App(Arc::new(t), Arc::new(a));
        }
        Ok(t)
    }

    fn atom(&mut self) -> Result<Term, String> {
        if self.eat_sym('(') {
            let t = self.term()?;
            self.expect_sym(')')?;
            return Ok(t);
        }
        let name = self.ident()?;
        if let Some(i) = self.vars.iter().rev().position(|v| *v == name) {
            return Ok(Term::Var(u32::try_from(i).map_err(|_| "too deep")?));
        }
        let head = match name.as_str() {
            "Unknown" => Head::Unknown,
            "Missing" => Head::Missing,
            "Query" => Head::Query,
            "@speaker" => Head::Ref(Deixis::Speaker),
            "@addressee" => Head::Ref(Deixis::Addressee),
            "@other" => Head::Ref(Deixis::Other),
            n if n.starts_with('#') => Head::Entity(
                self.look
                    .entity_id(n.get(1..).unwrap_or_default())
                    .ok_or_else(|| format!("unknown entity `{n}`"))?,
            ),
            n => Head::Concept(
                self.look
                    .concept_id(n)
                    .ok_or_else(|| format!("unknown concept or variable `{n}`"))?,
            ),
        };
        let mut node = Node::new(head);
        if let Some(Tok::Str(s)) = self.peek().cloned() {
            self.pos += 1;
            node.name = Some(Arc::from(s.as_str()));
        }
        if self.eat_sym('[') {
            loop {
                let f = self.ident()?;
                self.expect_sym('=')?;
                let v = self.ident()?;
                let fid = self
                    .look
                    .feat_id(&f)
                    .ok_or_else(|| format!("unknown feature `{f}`"))?;
                let vid = self
                    .look
                    .val_id(fid, &v)
                    .ok_or_else(|| format!("unknown value `{v}` of `{f}`"))?;
                node.feats.push((fid, vid));
                if self.eat_sym(']') {
                    break;
                }
                self.expect_sym(',')?;
            }
            node.feats.sort();
        }
        // Role list: `(` followed by `name :`.
        if self.peek() == Some(&Tok::Sym('('))
            && matches!(self.peek_at(1), Some(Tok::Ident(_)))
            && self.peek_at(2) == Some(&Tok::Sym(':'))
        {
            self.pos += 1;
            loop {
                let r = self.ident()?;
                self.expect_sym(':')?;
                let rid = self
                    .look
                    .role_id(&r)
                    .ok_or_else(|| format!("unknown role `{r}`"))?;
                let c = self.term()?;
                node.roles.push((rid, c));
                if self.eat_sym(')') {
                    break;
                }
                self.expect_sym(',')?;
            }
            sort_roles(&mut node.roles);
        }
        Ok(Term::node(node))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny fixed vocabulary for tests.
    pub struct Vocab;

    const CONCEPTS: [&str; 4] = ["Eat", "Girl", "Cake", "Big"];
    const ROLES: [&str; 3] = ["agent", "patient", "attr"];

    impl TermNames for Vocab {
        fn concept_name(&self, id: u32) -> String {
            CONCEPTS[id as usize].into()
        }
        fn role_name(&self, id: u32) -> String {
            ROLES[id as usize].into()
        }
        fn feat_name(&self, _: u32) -> String {
            "def".into()
        }
        fn val_name(&self, _: u32, v: u16) -> String {
            ["yes", "no"][v as usize].into()
        }
        fn entity_name(&self, _: u32) -> String {
            "E".into()
        }
    }

    impl TermLookup for Vocab {
        fn concept_id(&self, name: &str) -> Option<u32> {
            CONCEPTS.iter().position(|c| *c == name).map(|i| i as u32)
        }
        fn role_id(&self, name: &str) -> Option<u32> {
            ROLES.iter().position(|c| *c == name).map(|i| i as u32)
        }
        fn feat_id(&self, name: &str) -> Option<u32> {
            (name == "def").then_some(0)
        }
        fn val_id(&self, _: u32, name: &str) -> Option<u16> {
            ["yes", "no"]
                .iter()
                .position(|c| *c == name)
                .map(|i| i as u16)
        }
        fn entity_id(&self, _: &str) -> Option<u32> {
            None
        }
    }

    #[test]
    fn compose_a_sentence_meaning() {
        let v = Vocab;
        let eat = Term::parse("\\y x. Eat(agent: x, patient: y)", &v).unwrap();
        assert_eq!(eat.show(&v), "\\x y. Eat(agent: y, patient: x)");
        let the = Term::parse("\\x. x + def=yes", &v).unwrap();
        let big = Term::parse("\\x. x + attr: Big", &v).unwrap();
        let cake = Term::parse("Cake", &v).unwrap();
        let girl = Term::parse("Girl", &v).unwrap();
        let np = the.apply(&big.apply(&cake).unwrap()).unwrap();
        assert_eq!(np.show(&v), "Cake[def=yes](attr: Big)");
        let vp = eat.apply(&np).unwrap();
        let s = vp.apply(&girl).unwrap();
        assert_eq!(
            s.show(&v),
            "Eat(agent: Girl, patient: Cake[def=yes](attr: Big))"
        );
        assert!(s.is_tree());
        // Setting a feature twice is ill-formed.
        assert!(the.apply(&np).is_none());
        // Labels: what each functor does with its first argument.
        assert_eq!(
            eat.classify(),
            Label {
                kind: LabelKind::Arg,
                name: 1
            }
        );
        assert_eq!(
            big.classify(),
            Label {
                kind: LabelKind::Mod,
                name: 2
            }
        );
        assert_eq!(
            the.classify(),
            Label {
                kind: LabelKind::Feat,
                name: 0
            }
        );
        let raise = Term::parse("\\f. f Girl", &v).unwrap();
        assert_eq!(raise.classify().kind, LabelKind::Ho);
        // Composition: (\x. x + def=yes) . (\y. Eat(patient: y)) style.
        let comp = Term::parse(
            "\\z. (\\x. x + attr: Big) ((\\y x. Eat(agent: x, patient: y)) z Girl)",
            &v,
        )
        .unwrap()
        .normalize()
        .unwrap();
        assert_eq!(
            comp.show(&v),
            "\\x. Eat(agent: Girl, patient: x, attr: Big)"
        );
    }
}
