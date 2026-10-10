//! The exported crate and the training engine give the same cost on the
//! fixture text.

use std::sync::Arc;

use sememe_engine::context::Context;
use sememe_engine::model::Model;
use sememe_engine::parse::Parser;
use sememe_engine::program::{Program, ProgramFiles};

fn cost(model: &Model, text: &str) -> u64 {
    let mut parser = Parser::new(model);
    let mut total = 0;
    for doc in text.split("\u{1e}\n") {
        let mut ctx = Context::new();
        for line in doc.lines() {
            total += parser.parse_line(&mut ctx, line).bits;
        }
    }
    total
}

#[test]
fn exported_crate_and_engine_agree() {
    let text = include_str!("../../../tests/fixture/fixture.docs");
    let mut files = ProgramFiles::new();
    for name in [
        "ontology.sem",
        "grammar.sem",
        "lexicon.sem",
        "knowledge.sem",
        "counts.tsv",
    ] {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../program/").to_string() + name;
        files.insert(
            name.to_string(),
            std::fs::read_to_string(path).unwrap_or_default(),
        );
    }
    let engine = Model::new(Arc::new(Program::from_files(&files).unwrap()));
    let exported = cost(sememe_model::model(), text);
    assert_eq!(exported, cost(&engine, text));
    assert!(exported > 0);
}
