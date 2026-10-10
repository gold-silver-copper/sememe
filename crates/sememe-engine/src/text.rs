//! The fixed formatting layer: how a line's exact bytes split into a
//! speaker tag, words (with their letter case) and the gaps between words.
//!
//! A line is `tag? gap0 word1 gap1 word2 ... wordN gapN`. Words are runs of
//! letters and digits, with `'`, `’` or `-` allowed between two of them.
//! Everything else (spaces, punctuation) belongs to gaps. Words are
//! lowercased for the lexicon; their case is coded separately. Gaps are
//! coded as whole strings. Together these cover every byte of the line.

/// The kinds of speaker tag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TagKind {
    /// No tag.
    None = 0,
    /// CHILDES style: `*MOT:\t`.
    Childes = 1,
    /// Switchboard style: `A:\t`.
    Swb = 2,
    /// Subtitle turn marker: a leading `- `.
    Dash = 3,
}

/// Letter case of a word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Case {
    /// all lowercase
    Lower = 0,
    /// First letter uppercase.
    Cap = 1,
    /// ALL UPPERCASE.
    Upper = 2,
    /// Mixed: one bit per letter says which are uppercase.
    Mixed = 3,
    /// Case cannot be recovered from the lowercase form: the word is kept
    /// as written.
    Raw = 4,
}

/// A word of a line.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// The lexical form (lowercased unless `case` is `Raw`).
    pub form: String,
    /// How to restore the written case.
    pub case: Case,
    /// For `Mixed`: number of letters that have case (one bit each).
    pub mixed_letters: u32,
    /// The word as written.
    pub written: String,
}

/// A tokenized line.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// Speaker tag kind.
    pub tag: TagKind,
    /// Speaker name for `Childes` and `Swb` tags.
    pub speaker: Option<String>,
    /// Words in order.
    pub words: Vec<Word>,
    /// Gaps: `words.len() + 1` strings (before, between and after words).
    pub gaps: Vec<String>,
}

/// Case position classes.
pub const CASE_LINE_START: u64 = 0;
/// After a gap holding `.`, `!` or `?`.
pub const CASE_SENTENCE_START: u64 = 1;
/// Anywhere else.
pub const CASE_OTHER: u64 = 2;

/// Gap position classes.
pub const GAP_START: u64 = 0;
/// Between two words of one fragment.
pub const GAP_INTERNAL: u64 = 1;
/// Between two fragments.
pub const GAP_BOUNDARY: u64 = 2;
/// After the last word.
pub const GAP_END: u64 = 3;
/// A line with no words: its whole text.
pub const GAP_WHOLE: u64 = 4;

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '’'
}

fn split_tag(raw: &str) -> (TagKind, Option<String>, &str) {
    if let Some(rest) = raw.strip_prefix('*')
        && let Some((name, after)) = rest.split_once(":\t")
        && !name.is_empty()
        && name.len() <= 8
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return (TagKind::Childes, Some(name.to_string()), after);
    }
    if let Some((name, after)) = raw.split_once(":\t")
        && name.len() == 1
        && name.chars().all(|c| c.is_ascii_uppercase())
    {
        return (TagKind::Swb, Some(name.to_string()), after);
    }
    if let Some(after) = raw.strip_prefix("- ") {
        return (TagKind::Dash, None, after);
    }
    (TagKind::None, None, raw)
}

fn case_of(w: &str) -> (String, Case, u32) {
    let lower = w.to_lowercase();
    if lower == w {
        return (lower, Case::Lower, 0);
    }
    let mut cap = String::new();
    let mut chars = lower.chars();
    if let Some(first) = chars.next() {
        cap.extend(first.to_uppercase());
        cap.push_str(chars.as_str());
    }
    if cap == w {
        return (lower, Case::Cap, 0);
    }
    if lower.to_uppercase() == w {
        return (lower, Case::Upper, 0);
    }
    // Mixed: rebuild char by char.
    let mut rebuilt = String::new();
    let mut letters = 0;
    for (lc, wc) in lower.chars().zip(w.chars()) {
        let up: String = lc.to_uppercase().collect();
        if up != lc.to_string() {
            letters += 1;
        }
        if wc.to_string() == up {
            rebuilt.push_str(&up);
        } else {
            rebuilt.push(lc);
        }
    }
    if rebuilt == w && lower.chars().count() == w.chars().count() {
        return (lower, Case::Mixed, letters);
    }
    (w.to_string(), Case::Raw, 0)
}

impl Line {
    /// Tokenize one line (without its newline).
    pub fn parse(raw: &str) -> Line {
        let (tag, speaker, rest) = split_tag(raw);
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        let mut words = Vec::new();
        let mut gaps = Vec::new();
        let mut gap_start = 0usize;
        let mut i = 0usize;
        while let Some(&(pos, c)) = chars.get(i) {
            // An apostrophe between letters starts a new word, a clitic:
            // "it's" is "it" + "'s", "don't" is "don" + "'t".
            let clitic = is_apostrophe(c)
                && i > 0
                && chars.get(i - 1).is_some_and(|&(_, p)| is_word_char(p))
                && chars.get(i + 1).is_some_and(|&(_, n)| is_word_char(n));
            if !is_word_char(c) && !clitic {
                i += 1;
                continue;
            }
            // A word starts here; a hyphen between letters joins.
            let mut j = i + 1;
            loop {
                match chars.get(j) {
                    Some(&(_, c)) if is_word_char(c) => j += 1,
                    Some(&(_, c))
                        if c == '-' && chars.get(j + 1).is_some_and(|&(_, n)| is_word_char(n)) =>
                    {
                        j += 2
                    }
                    _ => break,
                }
            }
            let end = chars.get(j).map_or(rest.len(), |&(p, _)| p);
            gaps.push(rest.get(gap_start..pos).unwrap_or_default().to_string());
            let text = rest.get(pos..end).unwrap_or_default();
            let (form, case, mixed_letters) = case_of(text);
            words.push(Word {
                form,
                case,
                mixed_letters,
                written: text.to_string(),
            });
            gap_start = end;
            i = j;
        }
        gaps.push(rest.get(gap_start..).unwrap_or_default().to_string());
        Line {
            tag,
            speaker,
            words,
            gaps,
        }
    }

    /// Rebuild the exact text (inverse of `parse`).
    pub fn render(&self) -> String {
        let mut out = String::new();
        match self.tag {
            TagKind::Childes => {
                out.push('*');
                out.push_str(self.speaker.as_deref().unwrap_or_default());
                out.push_str(":\t");
            }
            TagKind::Swb => {
                out.push_str(self.speaker.as_deref().unwrap_or_default());
                out.push_str(":\t");
            }
            TagKind::Dash => out.push_str("- "),
            TagKind::None => {}
        }
        for (i, g) in self.gaps.iter().enumerate() {
            out.push_str(g);
            if let Some(w) = self.words.get(i) {
                out.push_str(&w.written);
            }
        }
        out
    }

    /// Positions (0..=words.len()) where a fragment may start or end: the
    /// line ends, and gaps that hold anything other than whitespace.
    pub fn boundaries(&self) -> Vec<bool> {
        let n = self.words.len();
        (0..=n)
            .map(|i| {
                i == 0
                    || i == n
                    || self
                        .gaps
                        .get(i)
                        .is_some_and(|g| g.chars().any(|c| !c.is_whitespace()))
            })
            .collect()
    }

    /// Positions where a strong sentence end (`.`, `!`, `?`) separates words:
    /// fragments never span these.
    pub fn hard_stops(&self) -> Vec<bool> {
        let n = self.words.len();
        (0..=n)
            .map(|i| {
                i > 0
                    && i < n
                    && self
                        .gaps
                        .get(i)
                        .is_some_and(|g| g.chars().any(|c| matches!(c, '.' | '!' | '?')))
            })
            .collect()
    }

    /// The case position class of word `i`.
    pub fn case_class(&self, i: usize) -> u64 {
        if i == 0 {
            CASE_LINE_START
        } else if self
            .gaps
            .get(i)
            .is_some_and(|g| g.chars().any(|c| matches!(c, '.' | '!' | '?')))
        {
            CASE_SENTENCE_START
        } else {
            CASE_OTHER
        }
    }
}

/// Restore a word's written form.
pub fn recase(w: &Word) -> String {
    match w.case {
        Case::Lower | Case::Raw => w.form.clone(),
        Case::Cap => {
            let mut chars = w.form.chars();
            let mut out = String::new();
            if let Some(f) = chars.next() {
                out.extend(f.to_uppercase());
                out.push_str(chars.as_str());
            }
            out
        }
        Case::Upper => w.form.to_uppercase(),
        // Mixed case needs the per-letter bits, which a decoder reads from
        // the code; for display the form is shown capitalized.
        Case::Mixed => w.form.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_round_trip_and_split_sensibly() {
        for raw in [
            "*MOT:\tyour new clean blanket's up in your bed.",
            "A:\tI'm sure they are.",
            "- Here's some more wood,Jim.",
            "= = = Corcelles-près-Payerne = = =",
            "\"Don’t you reckon a bird’s got feelin’s, same’s you?\" she would ask.",
            "Erm  what would twelve  twelve four?",
            "",
            "][laughs]",
            "USA and NASA, McDonald",
        ] {
            let line = Line::parse(raw);
            assert_eq!(line.render(), raw, "{line:?}");
            for w in &line.words {
                if w.case != Case::Mixed {
                    assert_eq!(recase(w), w.written);
                }
            }
        }
        let l = Line::parse("*CHI:\tMom where my new blanket?");
        assert_eq!(l.tag, TagKind::Childes);
        assert_eq!(l.speaker.as_deref(), Some("CHI"));
        let forms: Vec<&str> = l.words.iter().map(|w| w.form.as_str()).collect();
        assert_eq!(forms, ["mom", "where", "my", "new", "blanket"]);
        assert_eq!(l.words[0].case, Case::Cap);
        let l = Line::parse("it's Fraser's, don't");
        let forms: Vec<&str> = l.words.iter().map(|w| w.form.as_str()).collect();
        assert_eq!(forms, ["it", "'s", "fraser", "'s", "don", "'t"]);
        assert_eq!(l.gaps[1], "");
        let l = Line::parse("McDonald");
        assert_eq!(l.words[0].case, Case::Mixed);
        assert_eq!(l.words[0].form, "mcdonald");
    }
}
