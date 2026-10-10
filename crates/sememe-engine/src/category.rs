//! Combinatory categorial grammar categories.
//!
//! A category is either atomic (`S`, `NP[num=pl]`) or a functor `X/Y` (looks
//! for a `Y` to its right and becomes `X`) or `X\Y` (looks to its left).
//! Atomic categories carry features, either fixed values or variables shared
//! within one lexical category (`NP[num=?a]/N[num=?a]`), so agreement has a
//! place to live. Derived categories never keep variables: after a rule
//! fires, unbound variables become unspecified.

use std::fmt::Write as _;
use std::sync::Arc;

/// Which side a functor looks for its argument on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Dir {
    /// `X/Y`: the argument is to the right.
    Fwd,
    /// `X\Y`: the argument is to the left.
    Bwd,
}

/// The value slot of one feature on an atomic category.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum FVal {
    /// A fixed value (index into the feature's values).
    Val(u16),
    /// A variable shared within one lexical category.
    Var(u8),
}

/// An atomic category with its features, sorted by feature dimension.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Atom {
    /// Index into the program's atomic categories.
    pub id: u16,
    /// `(dimension, value)` pairs, sorted by dimension; absent = unspecified.
    pub feats: Vec<(u16, FVal)>,
}

/// A category.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Cat {
    /// An atomic category.
    Atom(Atom),
    /// A functor category.
    Slash(Arc<Slash>),
}

/// The parts of a functor category `res/arg` or `res\arg`.
#[derive(Clone, Debug)]
pub struct Slash {
    /// What the functor becomes once it has its argument.
    pub res: Cat,
    /// Which side the argument is on.
    pub dir: Dir,
    /// The argument it looks for.
    pub arg: Cat,
    /// Cached `cost::cat_key` of the whole category.
    pub key: u64,
    /// Cached: does any atom in it carry features?
    pub feats: bool,
    /// Cached: does any atom in it carry a variable?
    pub vars: bool,
}

impl PartialEq for Slash {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.dir == o.dir && self.res == o.res && self.arg == o.arg
    }
}

impl Eq for Slash {}

impl std::hash::Hash for Slash {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.key);
    }
}

impl PartialOrd for Slash {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Slash {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (self.dir, &self.res, &self.arg).cmp(&(o.dir, &o.res, &o.arg))
    }
}

/// Names used to print and parse categories.
#[derive(Clone, Debug, Default)]
pub struct CatNames {
    /// Atomic category names, by id.
    pub atoms: Vec<String>,
    /// Feature dimensions: name and value names, by id.
    pub dims: Vec<(String, Vec<String>)>,
}

impl Cat {
    /// An atomic category without features.
    pub fn atom(id: u16) -> Cat {
        Cat::Atom(Atom {
            id,
            feats: Vec::new(),
        })
    }

    /// A functor category.
    pub fn slash(res: Cat, dir: Dir, arg: Cat) -> Cat {
        let key =
            crate::cost::slash_key(dir, crate::cost::cat_key(&res), crate::cost::cat_key(&arg));
        let feats = res.has_feats() || arg.has_feats();
        let vars = res.has_vars() || arg.has_vars();
        Cat::Slash(Arc::new(Slash {
            res,
            dir,
            arg,
            key,
            feats,
            vars,
        }))
    }

    /// True if any atom carries features.
    pub fn has_feats(&self) -> bool {
        match self {
            Cat::Atom(a) => !a.feats.is_empty(),
            Cat::Slash(s) => s.feats,
        }
    }

    /// The functor parts, if this is a functor.
    pub fn as_slash(&self) -> Option<&Slash> {
        match self {
            Cat::Slash(s) => Some(s),
            Cat::Atom(_) => None,
        }
    }

    /// Number of arguments before the category becomes atomic.
    pub fn arity(&self) -> usize {
        match self {
            Cat::Atom(_) => 0,
            Cat::Slash(s) => 1 + s.res.arity(),
        }
    }

    /// The atomic category at the end of the result chain.
    pub fn target(&self) -> &Atom {
        match self {
            Cat::Atom(a) => a,
            Cat::Slash(s) => s.res.target(),
        }
    }

    /// The arguments from outermost (taken first) to innermost.
    pub fn args(&self) -> Vec<(Dir, &Cat)> {
        let mut out = Vec::new();
        let mut cur = self;
        while let Cat::Slash(s) = cur {
            out.push((s.dir, &s.arg));
            cur = &s.res;
        }
        out
    }

    /// True if this is `X/X` or `X\X` (ignoring features): a modifier shape.
    pub fn is_modifier(&self) -> bool {
        match self {
            Cat::Slash(s) => s.res.same_shape(&s.arg),
            Cat::Atom(_) => false,
        }
    }

    /// Structural equality ignoring features.
    pub fn same_shape(&self, other: &Cat) -> bool {
        match (self, other) {
            (Cat::Atom(a), Cat::Atom(b)) => a.id == b.id,
            (Cat::Slash(a), Cat::Slash(b)) => {
                a.dir == b.dir && a.res.same_shape(&b.res) && a.arg.same_shape(&b.arg)
            }
            _ => false,
        }
    }

    /// The same category with all features removed.
    pub fn bare(&self) -> Cat {
        match self {
            Cat::Atom(a) => Cat::atom(a.id),
            Cat::Slash(s) => Cat::slash(s.res.bare(), s.dir, s.arg.bare()),
        }
    }

    /// True if the category mentions any feature variable.
    pub fn has_vars(&self) -> bool {
        match self {
            Cat::Atom(a) => a.feats.iter().any(|(_, v)| matches!(v, FVal::Var(_))),
            Cat::Slash(s) => s.vars,
        }
    }

    /// Number of nodes (atoms plus slashes): a size measure.
    pub fn size(&self) -> usize {
        match self {
            Cat::Atom(_) => 1,
            Cat::Slash(s) => 1 + s.res.size() + s.arg.size(),
        }
    }

    /// Print with names.
    pub fn show(&self, names: &CatNames) -> String {
        let mut out = String::new();
        self.write(names, &mut out, false);
        out
    }

    fn write(&self, names: &CatNames, out: &mut String, paren: bool) {
        match self {
            Cat::Atom(a) => {
                out.push_str(
                    names
                        .atoms
                        .get(usize::from(a.id))
                        .map_or("?", |s| s.as_str()),
                );
                if !a.feats.is_empty() {
                    out.push('[');
                    for (i, (dim, val)) in a.feats.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        let (dname, vals) = names
                            .dims
                            .get(usize::from(*dim))
                            .map_or(("?", None), |(n, v)| (n.as_str(), Some(v)));
                        out.push_str(dname);
                        out.push('=');
                        match val {
                            FVal::Val(v) => out.push_str(
                                vals.and_then(|vs| vs.get(usize::from(*v)))
                                    .map_or("?", |s| s.as_str()),
                            ),
                            FVal::Var(v) => {
                                let _ = write!(out, "?{}", char::from(b'a' + (v % 26)));
                            }
                        }
                    }
                    out.push(']');
                }
            }
            Cat::Slash(s) => {
                if paren {
                    out.push('(');
                }
                s.res.write(names, out, true);
                out.push(match s.dir {
                    Dir::Fwd => '/',
                    Dir::Bwd => '\\',
                });
                s.arg.write(names, out, true);
                if paren {
                    out.push(')');
                }
            }
        }
    }

    /// Parse a category such as `(S\NP[num=?a])/NP`. Slashes associate to
    /// the left, as usual in CCG.
    pub fn parse(text: &str, names: &CatNames) -> Result<Cat, String> {
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        let mut pos = 0;
        let cat = parse_slashes(&chars, &mut pos, names)?;
        if pos != chars.len() {
            return Err(format!("trailing text in category `{text}`"));
        }
        Ok(cat)
    }
}

fn parse_slashes(chars: &[char], pos: &mut usize, names: &CatNames) -> Result<Cat, String> {
    let mut left = parse_primary(chars, pos, names)?;
    while let Some(&c) = chars.get(*pos) {
        let dir = match c {
            '/' => Dir::Fwd,
            '\\' => Dir::Bwd,
            _ => break,
        };
        *pos += 1;
        let right = parse_primary(chars, pos, names)?;
        left = Cat::slash(left, dir, right);
    }
    Ok(left)
}

fn parse_primary(chars: &[char], pos: &mut usize, names: &CatNames) -> Result<Cat, String> {
    if chars.get(*pos) == Some(&'(') {
        *pos += 1;
        let inner = parse_slashes(chars, pos, names)?;
        if chars.get(*pos) != Some(&')') {
            return Err("missing `)` in category".into());
        }
        *pos += 1;
        return Ok(inner);
    }
    let start = *pos;
    while chars
        .get(*pos)
        .is_some_and(|c| c.is_alphanumeric() || *c == '_')
    {
        *pos += 1;
    }
    let name: String = chars.get(start..*pos).unwrap_or_default().iter().collect();
    let id = names
        .atoms
        .iter()
        .position(|a| *a == name)
        .ok_or_else(|| format!("unknown atomic category `{name}`"))?;
    let id = u16::try_from(id).map_err(|_| "too many atomic categories".to_string())?;
    let mut feats = Vec::new();
    if chars.get(*pos) == Some(&'[') {
        *pos += 1;
        loop {
            let s = *pos;
            while chars
                .get(*pos)
                .is_some_and(|c| *c != '=' && *c != ',' && *c != ']')
            {
                *pos += 1;
            }
            let dname: String = chars.get(s..*pos).unwrap_or_default().iter().collect();
            if chars.get(*pos) != Some(&'=') {
                return Err(format!("expected `=` after feature `{dname}`"));
            }
            *pos += 1;
            let s = *pos;
            while chars.get(*pos).is_some_and(|c| *c != ',' && *c != ']') {
                *pos += 1;
            }
            let vname: String = chars.get(s..*pos).unwrap_or_default().iter().collect();
            let dim = names
                .dims
                .iter()
                .position(|(n, _)| *n == dname)
                .ok_or_else(|| format!("unknown category feature `{dname}`"))?;
            let val = if let Some(var) = vname.strip_prefix('?') {
                let b = var.bytes().next().unwrap_or(b'a');
                FVal::Var(b.wrapping_sub(b'a') % 26)
            } else {
                let vals = names.dims.get(dim).map(|(_, v)| v);
                let v = vals
                    .and_then(|vs| vs.iter().position(|x| *x == vname))
                    .ok_or_else(|| format!("unknown value `{vname}` of `{dname}`"))?;
                FVal::Val(u16::try_from(v).map_err(|_| "too many values".to_string())?)
            };
            let dim = u16::try_from(dim).map_err(|_| "too many dims".to_string())?;
            feats.push((dim, val));
            match chars.get(*pos) {
                Some(',') => *pos += 1,
                Some(']') => {
                    *pos += 1;
                    break;
                }
                _ => return Err("unterminated feature list".into()),
            }
        }
        feats.sort();
    }
    Ok(Cat::Atom(Atom { id, feats }))
}

/// Variable bindings while unifying a functor (side 0) with an argument
/// (side 1). Each side has its own variables.
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    vars: [[Option<u16>; 8]; 2],
}

impl Bindings {
    fn get(&self, side: usize, var: u8) -> Option<u16> {
        self.vars
            .get(side)
            .and_then(|vs| vs.get(usize::from(var % 8)))
            .copied()
            .flatten()
    }

    fn set(&mut self, side: usize, var: u8, val: u16) {
        if let Some(slot) = self
            .vars
            .get_mut(side)
            .and_then(|vs| vs.get_mut(usize::from(var % 8)))
        {
            *slot = Some(val);
        }
    }

    fn resolve(&self, side: usize, v: FVal) -> Option<FVal> {
        match v {
            FVal::Val(_) => Some(v),
            FVal::Var(x) => self.get(side, x).map(FVal::Val),
        }
    }
}

fn unify_atom(a: &Atom, sa: usize, b: &Atom, sb: usize, bind: &mut Bindings) -> bool {
    if a.id != b.id {
        return false;
    }
    for (dim, va) in &a.feats {
        let Some((_, vb)) = b.feats.iter().find(|(d, _)| d == dim) else {
            continue;
        };
        match (bind.resolve(sa, *va), bind.resolve(sb, *vb)) {
            (Some(FVal::Val(x)), Some(FVal::Val(y))) => {
                if x != y {
                    return false;
                }
            }
            (Some(FVal::Val(x)), None) => {
                if let FVal::Var(v) = vb {
                    bind.set(sb, *v, x);
                }
            }
            (None, Some(FVal::Val(y))) => {
                if let FVal::Var(v) = va {
                    bind.set(sa, *v, y);
                }
            }
            _ => {}
        }
    }
    true
}

/// Unify category `a` (on side `sa`) with `b` (on side `sb`), extending
/// `bind`. Unspecified features match anything.
pub fn unify(a: &Cat, sa: usize, b: &Cat, sb: usize, bind: &mut Bindings) -> bool {
    match (a, b) {
        (Cat::Atom(x), Cat::Atom(y)) => unify_atom(x, sa, y, sb, bind),
        (Cat::Slash(x), Cat::Slash(y)) => {
            if !x.feats && !y.feats {
                return x.key == y.key;
            }
            x.dir == y.dir
                && unify(&x.res, sa, &y.res, sb, bind)
                && unify(&x.arg, sa, &y.arg, sb, bind)
        }
        _ => false,
    }
}

/// Apply bindings to a category from `side`; unbound variables are dropped
/// (become unspecified).
pub fn substitute(c: &Cat, side: usize, bind: &Bindings) -> Cat {
    if !c.has_vars() {
        return c.clone();
    }
    match c {
        Cat::Atom(a) => Cat::Atom(Atom {
            id: a.id,
            feats: a
                .feats
                .iter()
                .filter_map(|(d, v)| bind.resolve(side, *v).map(|v| (*d, v)))
                .collect(),
        }),
        Cat::Slash(s) => Cat::slash(
            substitute(&s.res, side, bind),
            s.dir,
            substitute(&s.arg, side, bind),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> CatNames {
        CatNames {
            atoms: vec!["S".into(), "NP".into(), "N".into()],
            dims: vec![("num".into(), vec!["sg".into(), "pl".into()])],
        }
    }

    #[test]
    fn parse_print_round_trip_and_agreement() {
        let n = names();
        for s in [
            "(S\\NP)/NP",
            "NP[num=?a]/N[num=?a]",
            "S",
            "(S/(S\\NP))\\NP[num=pl]",
        ] {
            let c = Cat::parse(s, &n).unwrap();
            assert_eq!(c.show(&n), s);
        }
        // "the" NP[?a]/N[?a] applied to "cats" N[pl] gives NP[pl].
        let the = Cat::parse("NP[num=?a]/N[num=?a]", &n).unwrap();
        let cats = Cat::parse("N[num=pl]", &n).unwrap();
        let s = the.as_slash().unwrap();
        let mut b = Bindings::default();
        assert!(unify(&s.arg, 0, &cats, 1, &mut b));
        assert_eq!(substitute(&s.res, 0, &b).show(&n), "NP[num=pl]");
        // "are" (S\NP[pl]) rejects a singular subject.
        let are = Cat::parse("S\\NP[num=pl]", &n).unwrap();
        let sg = Cat::parse("NP[num=sg]", &n).unwrap();
        let mut b = Bindings::default();
        assert!(!unify(&are.as_slash().unwrap().arg, 0, &sg, 1, &mut b));
    }
}
