//! Build-order check: an empty program costs exactly the all-unknown cost
//! computed by hand, and a small hand-written lexicon lowers it.

use std::sync::Arc;

use sememe_engine::context::Context;
use sememe_engine::cost::{ONE_BIT, TAG_ENTRY, Table, cat_key, lg, sem_key, spell};
use sememe_engine::model::Model;
use sememe_engine::parse::Parser;
use sememe_engine::program::{Concept, CtxCounts, Entry, Meta, Program};
use sememe_engine::term::Term;

fn cost(p: Program, line: &str) -> u64 {
    let m = Model::new(Arc::new(p));
    let mut parser = Parser::new(&m);
    parser.parse_line(&mut Context::new(), line).bits
}

#[test]
#[allow(clippy::identity_op)]
fn empty_program_costs_exactly_the_all_unknown_cost() {
    // The line "no" in the seed program (3 atoms, 2 rules, 3 roles, 2
    // concepts, no features, no stored counts), by hand:
    let atoms = 3;
    let expected = lg(4)                 // tag kind: uniform over 4
        + ONE_BIT                        // line is not empty: 1 bit
        + ONE_BIT + lg(atoms)            // root category S: atom flag + which atom
        + lg(2 + 1)                      // expand S: <lex> among 2 rules + lex
        + 0                              // inflection: uniform over 0 + 1
        + spell(b"no")                   // the unknown word, spelled
        + lg(4)                          // world: mode fresh
        + lg(2 + 0 + 3)                  // head Unknown among 2 concepts + 3 specials
        + lg(0 + 1)                      // no features: <absent> among 0 values + 1
        + lg(3 + 1)                      // no roles: <absent> among 3 roles + 1
        + ONE_BIT                        // no more fragments
        + lg(5)                          // case: lower among 5
        + spell(b"")                     // gap before: empty
        + spell(b""); // gap after: empty
    assert_eq!(cost(Program::seed(), "no"), expected);
}

#[test]
fn a_hand_written_lexicon_lowers_the_cost() {
    let line = "the cat sat on the mat";
    let empty = cost(Program::seed(), line);
    let mut p = Program::seed();
    for (id, name) in [(10, "Cat"), (11, "Mat"), (12, "Sit")] {
        p.concepts.insert(
            id,
            Concept {
                name: name.into(),
                parent: 0,
                meta: Meta::default(),
            },
        );
    }
    p.reindex();
    let cn = p.cat_names();
    for (id, form, cat, sem) in [
        (0, "cat", "N", "Cat"),
        (1, "mat", "N", "Mat"),
        (2, "sat", "S\\N", "\\x. Sit(arg: x)"),
        (3, "the", "N/N", "\\x. x"),
        (4, "on", "(S\\S)/N", "\\y x. x + mod: y"),
    ] {
        let cat = sememe_engine::category::Cat::parse(cat, &cn).unwrap();
        let sem = Term::parse(sem, &p).unwrap();
        // One stored use each, as one counting pass would give.
        let mut cc = CtxCounts::default();
        cc.outs.insert(TAG_ENTRY | id as u64, 1);
        p.counts
            .tables
            .insert((Table::Lex as u8, [cat_key(&cat), sem_key(&sem)]), cc);
        p.entries.insert(
            id,
            Entry {
                form: form.into(),
                cat,
                sem,
                meta: Meta::default(),
            },
        );
    }
    let known = cost(p, line);
    // A stored entry costs 1 bit (count 1 against the escape) instead of
    // its spelling (16 letters at about 4 bits each plus end markers).
    assert!(
        known + 40 * ONE_BIT < empty,
        "known {known} vs empty {empty}"
    );
}

#[test]
fn what_the_engine_speaks_it_understands_back() {
    use sememe_engine::generate::speak;
    let mut p = Program::seed();
    for (id, name) in [(10, "Girl"), (11, "Cake"), (12, "Eat"), (13, "Big")] {
        p.concepts.insert(
            id,
            Concept {
                name: name.into(),
                parent: 0,
                meta: Meta::default(),
            },
        );
    }
    for (id, name) in [(5, "agent"), (6, "patient"), (7, "attr")] {
        p.roles.insert(
            id,
            sememe_engine::program::Role {
                name: name.into(),
                domain: 0,
                range: 0,
                meta: Meta::default(),
            },
        );
    }
    p.feats.insert(
        0,
        sememe_engine::program::Feature {
            name: "def".into(),
            domain: 0,
            values: vec!["yes".into(), "no".into()],
            meta: Meta::default(),
        },
    );
    p.reindex();
    let cn = p.cat_names();
    for (id, form, cat, sem) in [
        (0, "girl", "N", "Girl"),
        (1, "cake", "N", "Cake"),
        (2, "ate", "(S\\NP)/NP", "\\y x. Eat(agent: x, patient: y)"),
        (3, "the", "NP/N", "\\x. x + def=yes"),
        (4, "a", "NP/N", "\\x. x + def=no"),
        (5, "big", "N/N", "\\x. x + attr: Big"),
    ] {
        let cat = sememe_engine::category::Cat::parse(cat, &cn).unwrap();
        let sem = Term::parse(sem, &p).unwrap();
        let mut cc = CtxCounts::default();
        cc.outs.insert(TAG_ENTRY | id as u64, 1);
        p.counts
            .tables
            .insert((Table::Lex as u8, [cat_key(&cat), sem_key(&sem)]), cc);
        p.entries.insert(
            id,
            Entry {
                form: form.into(),
                cat,
                sem,
                meta: Meta::default(),
            },
        );
    }
    let meaning = Term::parse(
        "Eat(agent: Girl[def=yes], patient: Cake[def=no](attr: Big))",
        &p,
    )
    .unwrap()
    .normalize()
    .unwrap();
    let m = Model::new(Arc::new(p));
    let phrase = speak(&m, &meaning).expect("speakable");
    assert_eq!(phrase.words.join(" "), "the girl ate a big cake");
    let mut parser = Parser::new(&m);
    let a = parser.parse_line(&mut Context::new(), &phrase.words.join(" "));
    assert_eq!(a.fragments.len(), 1);
    assert_eq!(a.fragments[0].meaning, meaning);
}
