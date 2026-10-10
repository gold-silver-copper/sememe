//! The command line for the exported model: talk, explain, speak, inspect,
//! score. No LLM, no network.

use std::io::BufRead as _;

use sememe_engine::context::Context;
use sememe_engine::cost::float;
use sememe_engine::generate::{Rng, sample_meaning, sentence, speak};
use sememe_engine::parse::{Deriv, LeafKind, Parser, apply_mentions};
use sememe_engine::reason::{Answer, answer};
use sememe_engine::term::{Head, Term, TermNames};

const USAGE: &str = "usage:
  sememe-model talk <passage file> [question ...]   read a passage, then answer questions (from the arguments or stdin)
  sememe-model explain <text>                        show how a line is understood and what each part costs
  sememe-model speak <meaning>                       say a meaning, e.g. 'Eat(agent: Girl[def=yes], patient: Cake)'
  sememe-model sample [n] [seed]                     sample n meanings from the world model and say them
  sememe-model inspect <concept or word>             a concept's place in the ontology and its words, or a word's entries
  sememe-model score <file>                          bits per byte of a text file (one document)";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    let code = match cmd {
        "talk" => talk(&rest),
        "explain" => explain(&rest.join(" ")),
        "speak" => speak_cmd(&rest.join(" ")),
        "sample" => sample(&rest),
        "inspect" => inspect(rest.first().copied().unwrap_or("")),
        "score" => score(rest.first().copied().unwrap_or("")),
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

fn show(t: &Term) -> String {
    t.show(sememe_model::program().as_ref())
}

fn say(t: &Term) -> String {
    match speak(sememe_model::model(), t) {
        Some(p) => sentence(&p.words),
        None => format!("(cannot say {})", show(t)),
    }
}

fn talk(args: &[&str]) -> i32 {
    let Some(path) = args.first() else {
        eprintln!("{USAGE}");
        return 2;
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("cannot read {path}");
        return 1;
    };
    let model = sememe_model::model();
    let p = sememe_model::program();
    let mut parser = Parser::new(model);
    let mut ctx = Context::new();
    for line in text.lines() {
        let a = parser.parse_line(&mut ctx, line);
        for f in &a.fragments {
            println!("read: {}", show(&f.meaning));
        }
    }
    let mut questions: Vec<String> = args.iter().skip(1).map(|s| s.to_string()).collect();
    if questions.is_empty() {
        questions = std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .collect();
    }
    for q in questions {
        // Understand the question in the context, without adding it.
        let mut qctx = ctx.clone();
        let a = parser.parse_line(&mut qctx, &q);
        let Some(f) = a.fragments.first() else {
            println!("Q: {q}\nA: (no words)");
            continue;
        };
        println!("Q: {q}\n   meaning: {}", show(&f.meaning));
        match answer(&p, &ctx, &f.meaning) {
            Answer::Thing(t, fact) => println!("A: {}\n   because: {}", say(&t), show(&fact)),
            Answer::Yes(fact) => println!("A: Yes.\n   because: {}", show(&fact)),
            Answer::No(fact) => println!("A: No.\n   because: {}", show(&fact)),
            Answer::Unknown => println!("A: I don't know."),
        }
        apply_mentions(&mut ctx, &[]);
    }
    0
}

fn print_deriv(d: &Deriv, depth: usize, p: &sememe_engine::program::Program, words: &[String]) {
    let pad = "  ".repeat(depth);
    let cn = p.cat_names();
    match d {
        Deriv::Leaf {
            word,
            kind,
            cat,
            sem,
            stem,
        } => {
            let w = words.get(*word).cloned().unwrap_or_default();
            let what = match kind {
                LeafKind::Entry { id, infl } => {
                    let round = p.entries.get(id).map_or(0, |e| e.meta.round);
                    let inf = infl
                        .map(|i| format!(" + inflection i{i}"))
                        .unwrap_or_default();
                    format!("entry e{id}{inf} (added in round {round})")
                }
                LeafKind::Name { id } => format!("name entry n{id}"),
                LeafKind::Unknown { .. } => format!("UNKNOWN word, spelled \"{stem}\""),
            };
            println!(
                "{pad}\"{w}\" {} : {}   [{what}]",
                cat.show(&cn),
                sem.show(p)
            );
        }
        Deriv::Unary {
            rule, child, cat, ..
        } => {
            println!("{pad}{} by rule g{rule}", cat.show(&cn));
            print_deriv(child, depth + 1, p, words);
        }
        Deriv::Binary {
            rule,
            left,
            right,
            cat,
            label,
            ..
        } => {
            let rname = p
                .rules
                .get(rule)
                .map(|r| format!("{:?}", r.kind))
                .unwrap_or_default();
            println!(
                "{pad}{} by g{rule} {rname} (split {:?} {})",
                cat.show(&cn),
                label.kind,
                label.name
            );
            print_deriv(left, depth + 1, p, words);
            print_deriv(right, depth + 1, p, words);
        }
        Deriv::Coord {
            rule,
            left,
            conj,
            right,
            cat,
            ..
        } => {
            println!("{pad}{} by coordination g{rule}", cat.show(&cn));
            print_deriv(left, depth + 1, p, words);
            print_deriv(conj, depth + 1, p, words);
            print_deriv(right, depth + 1, p, words);
        }
    }
}

fn explain(text: &str) -> i32 {
    let model = sememe_model::model();
    let p = sememe_model::program();
    let mut parser = Parser::new(model);
    let mut ctx = Context::new();
    let a = parser.parse_line(&mut ctx, text);
    let bytes = text.len() + 1;
    println!(
        "\"{text}\": {:.1} bits ({:.2} bits/byte; formatting {:.1} bits). Bits are the cheapest analysis found (a Viterbi path), an upper bound on the model's cost.",
        float(a.bits),
        float(a.bits) / bytes as f64,
        float(a.format_bits)
    );
    for f in &a.fragments {
        println!(
            "\nfragment words {}..{}: language {:.1} bits, meaning {:.1} bits",
            f.start,
            f.end,
            float(f.lang_bits),
            float(f.world_bits)
        );
        println!("meaning: {}", show(&f.meaning));
        let modes: Vec<&str> = f
            .modes
            .iter()
            .map(|m| match m {
                0 => "fresh",
                1 => "refer back",
                2 => "restate",
                _ => "infer",
            })
            .collect();
        println!("how each node was filled: {}", modes.join(", "));
        println!("derivation:");
        print_deriv(&f.deriv, 1, &p, &a.words);
    }
    0
}

fn speak_cmd(text: &str) -> i32 {
    let p = sememe_model::program();
    match Term::parse(text, p.as_ref())
        .ok()
        .and_then(|t| t.normalize())
    {
        Some(t) => {
            println!("{}", say(&t));
            0
        }
        None => {
            eprintln!("cannot read the meaning `{text}`");
            1
        }
    }
}

fn sample(args: &[&str]) -> i32 {
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(5);
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let model = sememe_model::model();
    let mut rng = Rng::new(seed);
    let mut said = 0;
    for _ in 0..n * 50 {
        if said >= n {
            break;
        }
        let Some(m) = sample_meaning(model, &mut rng) else {
            continue;
        };
        if let Some(p) = speak(model, &m) {
            println!("{}   <- {}", sentence(&p.words), show(&m));
            said += 1;
        }
    }
    0
}

fn inspect(what: &str) -> i32 {
    let p = sememe_model::program();
    let cn = p.cat_names();
    if let Some(c) = p.concept_by_name(what) {
        let Some(con) = p.concepts.get(&c) else {
            return 1;
        };
        println!(
            "concept {} (c{c}), added in round {}",
            con.name, con.meta.round
        );
        println!("  {}", con.meta.doc);
        let chain: Vec<String> = p.ancestors(c).iter().map(|a| p.concept_name(*a)).collect();
        println!("  is a: {}", chain.join(" < "));
        let kids: Vec<String> = p
            .concepts
            .iter()
            .filter(|(k, x)| x.parent == c && **k != c)
            .map(|(_, x)| x.name.clone())
            .collect();
        if !kids.is_empty() {
            println!("  kinds: {}", kids.join(", "));
        }
        for (id, e) in &p.entries {
            let mut uses = false;
            e.sem
                .for_each_node(&mut |n| uses |= n.head == Head::Concept(c));
            if uses {
                println!(
                    "  word e{id} \"{}\" {} : {}",
                    e.form,
                    e.cat.show(&cn),
                    e.sem.show(p.as_ref())
                );
            }
        }
        return 0;
    }
    let mut found = false;
    for (id, e) in &p.entries {
        if e.form == what.to_lowercase() {
            found = true;
            println!(
                "e{id} \"{}\" {} : {}   (round {}){}",
                e.form,
                e.cat.show(&cn),
                e.sem.show(p.as_ref()),
                e.meta.round,
                if e.meta.doc.is_empty() {
                    String::new()
                } else {
                    format!(" -- {}", e.meta.doc)
                }
            );
        }
    }
    if !found {
        println!("no concept or word `{what}`");
    }
    0
}

fn score(path: &str) -> i32 {
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("cannot read {path}");
        return 1;
    };
    let mut parser = Parser::new(sememe_model::model());
    let mut ctx = Context::new();
    let mut bits = 0;
    for line in text.lines() {
        bits += parser.parse_line(&mut ctx, line).bits;
    }
    println!(
        "{:.4} bits/byte ({:.1} bits, {} bytes)",
        float(bits) / text.len().max(1) as f64,
        float(bits),
        text.len()
    );
    0
}
