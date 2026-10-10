//! Splitting a plain-text source into documents.

use std::io::BufRead;

use anyhow::{Context, Result};

/// Longest document, in lines. Longer runs become consecutive documents.
pub const MAX_DOC_LINES: usize = 64;

/// Where documents start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRule {
    /// After a run of blank (whitespace-only) lines. The blank lines end the
    /// document before them.
    Blank,
    /// As [`SplitRule::Blank`], and also at a Simple Wikipedia header line
    /// `= = = Title = = =`.
    BlankOrWikiHeader,
}

/// Is this a Simple Wikipedia article header `= = = Title = = =`?
pub fn is_wiki_header(line: &str) -> bool {
    line.len() > "= = =  = = =".len() && line.starts_with("= = = ") && line.ends_with(" = = =")
}

fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// Incremental document splitter: push lines, get finished documents back.
#[derive(Debug)]
pub struct DocSplitter {
    rule: SplitRule,
    current: Vec<String>,
    prev_blank: bool,
}

impl DocSplitter {
    pub fn new(rule: SplitRule) -> Self {
        Self {
            rule,
            current: Vec::new(),
            prev_blank: false,
        }
    }

    /// Adds the next line; returns the previous document if this line starts
    /// a new one.
    pub fn push(&mut self, line: String) -> Option<Vec<String>> {
        let blank = is_blank(&line);
        let header = self.rule == SplitRule::BlankOrWikiHeader && is_wiki_header(&line);
        let starts_new = !self.current.is_empty()
            && (self.current.len() >= MAX_DOC_LINES || (self.prev_blank && !blank) || header);
        let done = starts_new.then(|| std::mem::take(&mut self.current));
        self.current.push(line);
        self.prev_blank = blank;
        done
    }

    /// The last document, if any.
    pub fn finish(self) -> Option<Vec<String>> {
        (!self.current.is_empty()).then_some(self.current)
    }
}

/// Streams a plain-text source (UTF-8 required) and calls `f` on each
/// document. Lines lose their trailing `\n` only.
pub fn for_each_doc<R: BufRead>(
    mut reader: R,
    rule: SplitRule,
    name: &str,
    mut f: impl FnMut(Vec<String>) -> Result<()>,
) -> Result<()> {
    let mut splitter = DocSplitter::new(rule);
    let mut buf = Vec::new();
    let mut line_no = 0u64;
    loop {
        buf.clear();
        if reader
            .read_until(b'\n', &mut buf)
            .with_context(|| format!("reading {name}"))?
            == 0
        {
            break;
        }
        line_no += 1;
        if buf.last() == Some(&b'\n') {
            buf.pop();
        }
        let line = String::from_utf8(std::mem::take(&mut buf))
            .with_context(|| format!("{name}: invalid UTF-8 on line {line_no}"))?;
        if let Some(doc) = splitter.push(line) {
            f(doc)?;
        }
    }
    if let Some(doc) = splitter.finish() {
        f(doc)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str, rule: SplitRule) -> Vec<Vec<String>> {
        let mut docs = Vec::new();
        for_each_doc(text.as_bytes(), rule, "test", |d| {
            docs.push(d);
            Ok(())
        })
        .unwrap();
        docs
    }

    #[test]
    fn splitting_rules() {
        // Blank lines (also whitespace-only, also runs of them) end the
        // document before them.
        let text = "a\nb\n\n \nc\n= = = T = = =\nd\n\n= = = U = = =\ne\n";
        let plain = split(text, SplitRule::Blank);
        assert_eq!(
            plain,
            vec![
                vec!["a", "b", "", " "],
                vec!["c", "= = = T = = =", "d", ""],
                vec!["= = = U = = =", "e"],
            ]
        );
        let wiki = split(text, SplitRule::BlankOrWikiHeader);
        assert_eq!(
            wiki,
            vec![
                vec!["a", "b", "", " "],
                vec!["c"],
                vec!["= = = T = = =", "d", ""],
                vec!["= = = U = = =", "e"],
            ]
        );
        // No line is lost or changed.
        assert_eq!(
            wiki.concat()
                .iter()
                .map(|l| format!("{l}\n"))
                .collect::<String>(),
            text
        );

        // Long runs are capped.
        let long: String = (0..150).map(|i| format!("{i}\n")).collect();
        let sizes: Vec<usize> = split(&long, SplitRule::Blank)
            .iter()
            .map(Vec::len)
            .collect();
        assert_eq!(sizes, vec![64, 64, 22]);
    }
}
