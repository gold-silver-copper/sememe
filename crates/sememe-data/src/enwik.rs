//! enwik8 / enwik9: the raw bytes (for the Hutter Prize comparison), the
//! article titles (to remove them from the Wikipedia training data), and a
//! cleaned-text version.
//!
//! Cleaning, per `<page>` (redirects are skipped):
//! 1. take the `<text>` body and undo XML escaping (`&lt;` `&gt;` `&quot;`
//!    `&apos;` `&amp;`);
//! 2. drop HTML comments, `{{templates}}` and `{| tables |}` (nested), and
//!    `<ref>`, `<math>`, `<gallery>` and `<timeline>` elements with their
//!    contents; other tags are dropped, keeping their text;
//! 3. `[[Category:…]]`, `[[Image:…]]`, `[[File:…]]` and interlanguage links
//!    are dropped; `[[target|label]]` becomes `label`, `[[target]]` becomes
//!    `target`; `[http://… label]` becomes `label`;
//! 4. `''` and `'''` are dropped; per line, heading `=` signs, leading list
//!    markers (`*#:;`) and surrounding whitespace are trimmed, `----` lines
//!    dropped, and runs of empty lines collapsed to one;
//! 5. HTML entities (`&nbsp;`, `&mdash;`, `&#123;`, …) are decoded.
//!
//! Each article becomes one document: its title, then its cleaned lines.
//! The titles file lists every non-redirect page.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::docs::{DocWriter, FileStats, SEPARATOR};

/// What [`extract`] wrote for one enwik file.
#[derive(Debug, Clone, Serialize)]
pub struct EnwikOutput {
    pub raw_bytes: u64,
    pub raw_sha256: String,
    pub pages: u64,
    pub redirects: u64,
    pub invalid_utf8_lines: u64,
    pub clean: FileStats,
}

/// Extracts `<name>.zip` from `input` into `output`: `<name>` (raw bytes),
/// `<name>-titles.txt` (one title per line, unescaped) and
/// `<name>-clean.docs`.
pub fn extract(input: &Path, output: &Path, name: &str) -> Result<EnwikOutput> {
    use sha2::{Digest, Sha256};

    std::fs::create_dir_all(output)?;
    let zip_path = input.join(format!("{name}.zip"));
    let file = File::open(&zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))?;
    let entry = archive.by_name(name)?;
    let mut reader = BufReader::with_capacity(1 << 20, entry);

    let raw_path = output.join(name);
    let mut raw = BufWriter::with_capacity(1 << 20, File::create(&raw_path)?);
    let mut titles = BufWriter::new(File::create(output.join(format!("{name}-titles.txt")))?);
    let mut clean = DocWriter::create(&output.join(format!("{name}-clean.docs")))?;
    let mut hasher = Sha256::new();
    let mut out = EnwikOutput {
        raw_bytes: 0,
        raw_sha256: String::new(),
        pages: 0,
        redirects: 0,
        invalid_utf8_lines: 0,
        clean: FileStats::default(),
    };

    let mut buf = Vec::new();
    let mut title = String::new();
    let mut body = String::new();
    let mut in_text = false;
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        raw.write_all(&buf)?;
        hasher.update(&buf);
        out.raw_bytes += buf.len() as u64;
        let line = match std::str::from_utf8(&buf) {
            Ok(s) => s.to_string(),
            Err(_) => {
                out.invalid_utf8_lines += 1;
                String::from_utf8_lossy(&buf).into_owned()
            }
        };
        let mut rest = line.as_str();
        if !in_text {
            let t = rest.trim_start();
            if let Some(after) = t.strip_prefix("<title>") {
                title = xml_unescape(after.split("</title>").next().unwrap_or(after));
                continue;
            }
            if t.starts_with("<text") {
                let Some(start) = t.find('>') else { continue };
                if t.get(..start).is_some_and(|tag| tag.ends_with('/')) {
                    continue; // <text ... /> : empty body
                }
                rest = t.get(start + 1..).unwrap_or("");
                in_text = true;
                body.clear();
            } else if t.starts_with("</page>") {
                out.pages += 1;
                let text = xml_unescape(&body);
                if text
                    .trim_start()
                    .to_ascii_uppercase()
                    .starts_with("#REDIRECT")
                {
                    out.redirects += 1;
                } else {
                    titles.write_all(title.as_bytes())?;
                    titles.write_all(b"\n")?;
                    let mut lines = vec![title.clone()];
                    lines.extend(clean_wikitext(&text));
                    lines.retain(|l| l != SEPARATOR);
                    clean.write_doc(&lines)?;
                }
                body.clear();
                continue;
            } else {
                continue;
            }
        }
        if let Some(end) = rest.find("</text>") {
            body.push_str(rest.get(..end).unwrap_or(""));
            in_text = false;
        } else {
            body.push_str(rest);
        }
    }
    raw.flush()?;
    titles.flush()?;
    out.raw_sha256 = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    out.clean = clean.finish()?;
    Ok(out)
}

/// Undoes XML escaping.
pub fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Converts MediaWiki markup to plain lines (see the module docs).
pub fn clean_wikitext(text: &str) -> Vec<String> {
    let stripped = strip_markup(text);
    let mut lines: Vec<String> = Vec::new();
    for line in stripped.lines() {
        let mut l = line.trim();
        // Horizontal rules and table rows left over from tables the simple
        // nesting count could not delimit.
        if l.starts_with("----") || l.starts_with('|') || l.starts_with('!') {
            continue;
        }
        if l.starts_with('=') && l.ends_with('=') {
            l = l.trim_matches('=').trim();
        }
        l = l.trim_start_matches(['*', '#', ':', ';']).trim();
        let l = html_unescape(l);
        let l = l.trim();
        if l.is_empty() && lines.last().is_none_or(|p| p.is_empty()) {
            continue;
        }
        lines.push(l.to_string());
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines
}

/// Index just past the balanced `open … close` that `s` starts with (or the
/// end of `s` if unbalanced).
fn balanced_end(s: &str, open: &str, close: &str) -> usize {
    let mut depth = 0usize;
    let mut i = 0;
    while let Some(rest) = s.get(i..) {
        if rest.is_empty() {
            break;
        }
        if rest.starts_with(open) {
            depth += 1;
            i += open.len();
        } else if rest.starts_with(close) {
            depth = depth.saturating_sub(1);
            i += close.len();
            if depth == 0 {
                return i;
            }
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    s.len()
}

const DROPPED_NAMESPACES: [&str; 6] = [
    "category",
    "image",
    "file",
    "media",
    "wikipedia",
    "template",
];

fn is_dropped_link(target: &str) -> bool {
    let Some((prefix, _)) = target.split_once(':') else {
        return false;
    };
    let p = prefix.trim().to_ascii_lowercase();
    DROPPED_NAMESPACES.contains(&p.as_str())
        // interlanguage links: [[de:…]], [[zh-min-nan:…]]
        || (p.len() >= 2
            && p.len() <= 12
            && prefix.chars().all(|c| c.is_ascii_lowercase() || c == '-'))
}

fn strip_markup(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(c) = rest.chars().next() {
        if rest.starts_with("<!--") {
            rest = rest
                .find("-->")
                .and_then(|i| rest.get(i + 3..))
                .unwrap_or("");
        } else if rest.starts_with("{{") {
            rest = rest.get(balanced_end(rest, "{{", "}}")..).unwrap_or("");
        } else if rest.starts_with("{|") {
            rest = rest.get(balanced_end(rest, "{|", "|}")..).unwrap_or("");
        } else if rest.starts_with("[[") {
            let end = balanced_end(rest, "[[", "]]");
            let inner = rest
                .get(2..end)
                .map(|x| x.strip_suffix("]]").unwrap_or(x))
                .unwrap_or("");
            if !is_dropped_link(inner) {
                let label = inner.rsplit('|').next().unwrap_or(inner);
                out.push_str(&strip_markup(label));
            }
            rest = rest.get(end..).unwrap_or("");
        } else if ["[http://", "[https://", "[ftp://", "[mailto:"]
            .iter()
            .any(|p| rest.starts_with(p))
        {
            let end = rest.find(']').unwrap_or(rest.len());
            let inner = rest.get(1..end).unwrap_or("");
            if let Some((_, label)) = inner.split_once(' ') {
                out.push_str(&strip_markup(label));
            }
            rest = rest.get(end + 1..).unwrap_or("");
        } else if let Some(r) = skip_element(rest) {
            rest = r;
        } else if rest.starts_with("'''") {
            rest = rest.get(3..).unwrap_or("");
        } else if rest.starts_with("''") {
            rest = rest.get(2..).unwrap_or("");
        } else {
            out.push(c);
            rest = rest.get(c.len_utf8()..).unwrap_or("");
        }
    }
    out
}

/// Skips an HTML-like tag at the start of `s`; for `<ref>`, `<math>`,
/// `<gallery>` and `<timeline>` also skips their contents.
fn skip_element(s: &str) -> Option<&str> {
    let after_lt = s.strip_prefix('<')?;
    let first = after_lt.chars().next()?;
    if !(first.is_ascii_alphabetic() || first == '/') {
        return None;
    }
    let close = s.find('>').filter(|i| *i < 400)?;
    let tag = s.get(1..close)?;
    let rest = s.get(close + 1..)?;
    let name: String = tag
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if ["ref", "math", "gallery", "timeline"].contains(&name.as_str()) && !tag.ends_with('/') {
        let end_tag = format!("</{name}");
        let lower = rest.to_ascii_lowercase();
        return Some(match lower.find(&end_tag) {
            Some(i) => rest
                .get(i..)
                .and_then(|r| r.find('>').and_then(|j| r.get(j + 1..)))
                .unwrap_or(""),
            None => "",
        });
    }
    Some(rest)
}

/// Decodes HTML character references: numeric ones and a few common names.
pub fn html_unescape(s: &str) -> String {
    const NAMED: [(&str, &str); 34] = [
        ("nbsp", " "),
        ("amp", "&"),
        ("lt", "<"),
        ("gt", ">"),
        ("quot", "\""),
        ("apos", "'"),
        ("ndash", "\u{2013}"),
        ("mdash", "\u{2014}"),
        ("minus", "\u{2212}"),
        ("times", "\u{d7}"),
        ("divide", "\u{f7}"),
        ("plusmn", "\u{b1}"),
        ("deg", "\u{b0}"),
        ("hellip", "\u{2026}"),
        ("middot", "\u{b7}"),
        ("lsquo", "\u{2018}"),
        ("rsquo", "\u{2019}"),
        ("ldquo", "\u{201c}"),
        ("rdquo", "\u{201d}"),
        ("larr", "\u{2190}"),
        ("rarr", "\u{2192}"),
        ("harr", "\u{2194}"),
        ("rArr", "\u{21d2}"),
        ("hArr", "\u{21d4}"),
        ("radic", "\u{221a}"),
        ("infin", "\u{221e}"),
        ("le", "\u{2264}"),
        ("ge", "\u{2265}"),
        ("ne", "\u{2260}"),
        ("asymp", "\u{2248}"),
        ("sup2", "\u{b2}"),
        ("sup3", "\u{b3}"),
        ("frac12", "\u{bd}"),
        ("prime", "\u{2032}"),
    ];
    // Greek letters, in Unicode order (U+0391… and U+03B1…).
    const GREEK: [&str; 25] = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigmaf", "sigma", "tau", "upsilon",
        "phi", "chi", "psi", "omega",
    ];
    let greek = |name: &str| -> Option<String> {
        let lower = name.to_ascii_lowercase();
        let i = GREEK.iter().position(|g| *g == lower)? as u32;
        let upper = name.chars().next()?.is_ascii_uppercase();
        if upper && lower == "sigmaf" {
            return None;
        }
        char::from_u32(if upper { 0x391 } else { 0x3b1 } + i).map(String::from)
    };
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(rest.get(..i).unwrap_or(""));
        let tail = rest.get(i + 1..).unwrap_or("");
        let decoded = tail.find(';').filter(|j| *j <= 10).and_then(|j| {
            let name = tail.get(..j)?;
            let ch = if let Some(num) = name.strip_prefix('#') {
                let code = match num.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                    None => num.parse().ok()?,
                };
                char::from_u32(code)?.to_string()
            } else {
                match NAMED.iter().find(|(n, _)| *n == name) {
                    Some((_, v)) => v.to_string(),
                    None => greek(name)?,
                }
            };
            Some((ch, j + 1))
        });
        match decoded {
            Some((ch, len)) => {
                out.push_str(&ch);
                rest = tail.get(len..).unwrap_or("");
            }
            None => {
                out.push('&');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikitext_cleaning() {
        let text = "{{Infobox|a={{b}}}}'''Paris''' is the [[capital city|capital]] of \
                    [[France]].<ref>A {{cite}}</ref> See [http://x.org the site].\n\n\n\
                    == History ==\n* It is old&nbsp;&mdash; very.<!-- c -->\n\
                    [[Category:Cities]][[de:Paris]]\n{| class=x\n| a || b\n|}\n";
        assert_eq!(
            clean_wikitext(text),
            vec![
                "Paris is the capital of France. See the site.",
                "",
                "History",
                "It is old \u{2014} very.",
            ]
        );
    }
}
