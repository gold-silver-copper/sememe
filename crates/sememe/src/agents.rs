//! The slow tier: headless Opus coding agents, each in its own git
//! worktree, at most four at once. An agent reads the program as text plus
//! usage tables, and writes typed operations to `agent/ops.jsonl`. It never
//! writes the program: the owner replays its operations on the current
//! version, scores each group exactly, and keeps the ones that lower the
//! total cost (renames and docs are free and always kept).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context as _, Result};
use rustc_hash::FxHashMap;
use sememe_engine::cost::{self, TAG_ENTRY, Table};
use sememe_engine::term::{Head, TermNames};
use sememe_engine::text::Line;
use serde::{Deserialize, Serialize};

use crate::ops::Op;
use crate::trainer::Trainer;

/// Slow-tier tasks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Merge duplicate concepts and build the type hierarchy.
    Consolidate,
    /// Name and document concepts and entries.
    Name,
    /// Replace many specific entries with general rules and inflections.
    Generalize,
    /// Agents alone: edit the program directly to explain expensive text.
    Direct,
}

impl Task {
    fn brief(self) -> &'static str {
        match self {
            Task::Consolidate => {
                "Consolidate the ontology. Parallel teachers and the learner invent the same concept under \
                 different names (`Girl` and `FemaleChild`; `CtxDoggy` learned from context and `Dog`). Find \
                 concepts that mean the same thing and merge them (`merge_concepts`, keeping the better name). \
                 Build the hierarchy: concepts sitting directly under `Thing` that belong under a more specific \
                 type should be moved (`set_parent`); add intermediate concepts (`add_concept`) where several \
                 share a natural supertype (`Dog`, `Cat`, `Horse` < `Animal`). Do not merge concepts that differ \
                 in meaning just because they are related."
            }
            Task::Name => {
                "Name and document. Concepts named `Ctx...` were learned from context without a name: look at \
                 their words (usage.tsv, lexicon.sem) and the example lines, and give each a real name (`rename`, \
                 kind `concept`) and a one-line doc (`set_doc`). If a `Ctx` concept is the same as an existing \
                 concept, merge it instead (`merge_concepts`). Also document the most-used entries, rules and \
                 concepts that have no doc. Readability is the point: someone should be able to read the \
                 program like source code."
            }
            Task::Generalize => {
                "Generalize. Find places where many specific entries could be replaced by one general rule or \
                 entry, and propose that: an inflection (`add_infl`, e.g. category `N`, features `num=many`, strip \
                 \"\", add \"s\") with the full-form entries it makes redundant removed (`remove_entry`); a \
                 type-change rule (`add_rule` \"tc N NP\") when many nouns have both an `N` and an `NP` entry, with \
                 the redundant `NP` entries removed; composition (`fcomp`, `bcomp`), type raising (`ftr NP S`) or \
                 coordination rules where the lexicon has many entries that only exist to work around a missing \
                 rule. Put a rule and the removals it enables in the same group: each group is scored as one \
                 change."
            }
            Task::Direct => {
                "Explain the expensive text. unknowns.tsv lists the most frequent words the program cannot \
                 explain yet, with example lines. Edit the program directly so it explains them: add concepts \
                 (`add_concept`), roles and features if needed, and lexical entries (`add_entry`) with the right \
                 category and meaning, following the conventions you see in the existing lexicon (if it is \
                 empty, start them: atomic categories S, NP, N; a transitive verb is `(S\\NP)/NP : \\y x. \
                 Eat(agent: x, patient: y)`, a determiner `NP/N : \\x. x + def=yes`). Each group is scored by \
                 how much it lowers the total cost; group an entry with the concepts it needs."
            }
        }
    }
}

/// One operation line written by an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentOp {
    /// Change group: operations in one group are scored together.
    pub group: String,
    /// Why.
    #[serde(default)]
    pub why: String,
    /// The operation.
    pub op: Op,
}

const OPS_DOC: &str = r#"## Writing operations

Write `agent/ops.jsonl`: one JSON object per line, `{"group": "g1", "why": "...", "op": {...}}`.
Operations in the same group are scored together as one change; each group is kept only if it lowers
the total cost (program bits + bits of the training text). Renames and docs are free and always kept.
The operations (field `op` is the kind):

- `{"op": "add_concept", "name": "Dog", "parent": "Animal", "doc": "A domestic canine."}`
- `{"op": "add_role", "name": "agent", "domain": "Event", "range": "Thing", "doc": "..."}`
- `{"op": "add_feature", "name": "num", "domain": "Thing", "values": ["one", "many"], "doc": "..."}`
- `{"op": "add_entry", "form": "dog", "cat": "N", "sem": "Dog", "doc": "..."}` (meanings in the lexicon's notation)
- `{"op": "remove_entry", "id": 12}` (the number after `e` in lexicon.sem)
- `{"op": "add_name", "cat": "NP", "concept": "Person", "doc": "..."}` (any capitalized word may name a Person)
- `{"op": "add_infl", "cat": "N", "feats": "num=many", "strip": "", "add": "s", "doc": "..."}`
- `{"op": "add_rule", "rule": "tc N NP", "doc": "..."}` (also `fcomp`, `bcomp`, `fxcomp`, `bxcomp`, `ftr NP S`, `btr NP S`, `coord CONJ`)
- `{"op": "remove_rule", "id": 3}`
- `{"op": "add_atom", "name": "CONJ", "doc": "..."}`
- `{"op": "merge_concepts", "from": "FemaleChild", "into": "Girl"}`
- `{"op": "rename", "kind": "concept", "from": "CtxDoggy", "to": "Dog"}` (kinds: concept, role, feature)
- `{"op": "set_parent", "concept": "Dog", "parent": "Animal"}`
- `{"op": "set_doc", "kind": "concept", "item": "Dog", "doc": "..."}` (kinds: concept, role, feature, entry `e12`, rule `g3`)
- `{"op": "add_fact", "term": "Capital(of: #France, city: #Paris)", "doc": "..."}`
- `{"op": "add_entity", "name": "Paris", "concept": "City", "doc": "..."}`

Write nothing else outside `agent/`. Do not run commands. When done, stop.
"#;

fn uses_by_entry(t: &Trainer) -> FxHashMap<u32, u64> {
    let mut uses: FxHashMap<u32, u64> = FxHashMap::default();
    for ((tb, _), cc) in &t.program.counts.tables {
        if *tb != Table::Lex as u8 {
            continue;
        }
        for (o, q) in &cc.outs {
            if o & 0xffff_ffff_0000_0000 == TAG_ENTRY {
                *uses.entry(*o as u32).or_default() += cost::level_value(*q);
            }
        }
    }
    uses
}

/// Write the brief files an agent reads into `dir/agent/`.
pub fn write_brief(t: &Trainer, dir: &Path, task: Task) -> Result<()> {
    let a = dir.join("agent");
    std::fs::create_dir_all(&a)?;
    crate::store::save(&t.program, &dir.join("program"))?;
    let p = &t.program;
    let cn = p.cat_names();
    let uses = uses_by_entry(t);
    let mut s = String::from("id\tform\tcategory\tmeaning\tuses\n");
    let mut entries: Vec<_> = p.entries.iter().collect();
    entries.sort_by_key(|(id, _)| std::cmp::Reverse(uses.get(id).copied().unwrap_or(0)));
    for (id, e) in entries {
        let _ = writeln!(
            s,
            "e{id}\t{}\t{}\t{}\t{}",
            e.form,
            e.cat.show(&cn),
            e.sem.show(p),
            uses.get(id).copied().unwrap_or(0)
        );
    }
    std::fs::write(a.join("usage.tsv"), s)?;
    // Concepts with how many entries mention them.
    let mut mention: BTreeMap<u32, u64> = BTreeMap::new();
    for (id, e) in &p.entries {
        let u = uses.get(id).copied().unwrap_or(0);
        e.sem.for_each_node(&mut |n| {
            if let Head::Concept(c) = n.head {
                *mention.entry(c).or_default() += u;
            }
        });
    }
    let mut s = String::from("concept\tparent\tuses\tdoc\n");
    for (id, c) in &p.concepts {
        let _ = writeln!(
            s,
            "{}\t{}\t{}\t{}",
            c.name,
            p.concept_name(c.parent),
            mention.get(id).copied().unwrap_or(0),
            c.meta.doc.replace('\n', " ")
        );
    }
    std::fs::write(a.join("concepts.tsv"), s)?;
    // Frequent unknown words with example training lines.
    let known: rustc_hash::FxHashSet<&str> = p.entries.values().map(|e| e.form.as_str()).collect();
    let mut unk: Vec<(&String, &u64)> = t
        .freq
        .iter()
        .filter(|(w, _)| !known.contains(w.as_str()))
        .collect();
    unk.sort_by(|x, y| y.1.cmp(x.1).then(x.0.cmp(y.0)));
    let mut s = String::from("word\tcount\texamples\n");
    for (w, n) in unk.into_iter().take(400) {
        let mut ex = Vec::new();
        if let Some(docs) = t.index.get(w) {
            'docs: for d in docs.iter().take(20) {
                if let Some(doc) = t.corpus.docs.get(*d as usize) {
                    for l in &doc.lines {
                        if l.len() < 120 && Line::parse(l).words.iter().any(|x| &x.form == w) {
                            ex.push(l.replace('\t', " "));
                            if ex.len() >= 3 {
                                break 'docs;
                            }
                            break;
                        }
                    }
                }
            }
        }
        let _ = writeln!(s, "{w}\t{n}\t{}", ex.join(" || "));
    }
    std::fs::write(a.join("unknowns.tsv"), s)?;
    let readme = format!(
        "# Slow-tier task: {task:?}\n\n{}\n\nThe program is in `program/` (text form: `ontology.sem`, `grammar.sem`, `lexicon.sem`, \
         `knowledge.sem`, `counts.tsv`). Usage counts per entry are in `agent/usage.tsv`, concepts with how much \
         their entries are used in `agent/concepts.tsv`, frequent unknown words with example lines in \
         `agent/unknowns.tsv`. The engine and cost code are in `crates/sememe-engine/` (read `src/cost.rs` to see how \
         bits are counted); the meaning conventions are in `docs/meaning-guide.md`.\n\n{OPS_DOC}",
        task.brief()
    );
    std::fs::write(a.join("README.md"), readme)?;
    Ok(())
}

/// The settings file: read and search anywhere in the worktree, write only
/// under `agent/`, no commands, and the data folder (held-out text) denied.
pub fn settings(dir: &Path, data_dir: &Path) -> Result<PathBuf> {
    let data = data_dir.to_string_lossy().to_string();
    let v = serde_json::json!({
        "permissions": {
            "allow": ["Read", "Glob", "Grep", "Write(agent/**)", "Edit(agent/**)"],
            "deny": [
                format!("Read(/{data}/**)"),
                format!("Grep(/{data}/**)"),
                format!("Glob(/{data}/**)"),
                "Bash", "WebFetch", "WebSearch"
            ],
            "defaultMode": "dontAsk"
        }
    });
    let path = dir.join("agent").join("settings.json");
    std::fs::create_dir_all(dir.join("agent"))?;
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    Ok(path)
}

/// Make a git worktree for an agent.
pub fn worktree(name: &str) -> Result<PathBuf> {
    let dir = PathBuf::from("worktrees").join(name);
    if dir.exists() {
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(&dir)
            .status();
    }
    let ok = Command::new("git")
        .args(["worktree", "add", "--detach", "--force"])
        .arg(&dir)
        .arg("HEAD")
        .status()
        .context("git worktree add")?;
    if !ok.success() {
        anyhow::bail!("git worktree add failed");
    }
    Ok(std::fs::canonicalize(&dir)?)
}

/// Remove an agent worktree.
pub fn remove_worktree(dir: &Path) {
    let _ = Command::new("git")
        .args(["worktree", "remove", "--force"])
        .arg(dir)
        .status();
}

/// Run one agent on a task; returns its operations.
pub fn run(t: &Trainer, task: Task, k: usize) -> Result<Vec<AgentOp>> {
    let round = t.state.round + 1;
    let name = format!("{}-r{round}-{task:?}-{k}", t.cfg.run).to_lowercase();
    let dir = worktree(&name)?;
    write_brief(t, &dir, task)?;
    let data = std::fs::canonicalize("data")?;
    let settings = settings(&dir, &data)?;
    let prompt = format!(
        "You are a slow-tier agent improving sememe's learned program. Read agent/README.md first: it says your \
         task, where everything is, and the exact format of the operations to write to agent/ops.jsonl. You cannot \
         run commands. Work through the program text and the usage tables, then write your operations. Aim for \
         changes that make the program smaller or explain more text, and that a person reading the program would \
         recognize as right. Write at most 300 operations."
    );
    let row = t.opus.agent(
        round,
        &format!("agent-{task:?}"),
        &dir,
        &settings,
        &prompt,
        Duration::from_secs(7200),
    )?;
    eprintln!("agent {name}: {} ({:.0}s)", row.outcome, row.secs);
    let text = std::fs::read_to_string(dir.join("agent/ops.jsonl")).unwrap_or_default();
    let mut out = Vec::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<AgentOp>(l) {
            Ok(o) => out.push(o),
            Err(e) => eprintln!("agent {name}: bad op line ({e}): {l}"),
        }
    }
    // Keep the agent's output with the run, then clean up.
    let keep = t.dir.join("agents").join(&name);
    std::fs::create_dir_all(&keep)?;
    let _ = std::fs::copy(dir.join("agent/ops.jsonl"), keep.join("ops.jsonl"));
    remove_worktree(&dir);
    Ok(out)
}
