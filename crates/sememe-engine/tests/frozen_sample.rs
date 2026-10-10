//! Engine changes must not change costs within a run: the bits of the
//! fixture text under a fixed program are recorded, and CI fails if they
//! change. Changing them means starting a new run: rerun with
//! `SEMEME_BLESS=1` and record the new cost hash in `runs/current-run.json`.

use std::sync::Arc;

use sememe_engine::context::Context;
use sememe_engine::model::Model;
use sememe_engine::parse::Parser;
use sememe_engine::program::{Program, ProgramFiles};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixture/");

#[test]
fn fixture_bits_are_frozen() {
    let mut files = ProgramFiles::new();
    for name in [
        "ontology.sem",
        "grammar.sem",
        "lexicon.sem",
        "knowledge.sem",
        "counts.tsv",
    ] {
        let text = std::fs::read_to_string(format!("{ROOT}program/{name}")).unwrap();
        files.insert(name.to_string(), text);
    }
    let model = Model::new(Arc::new(Program::from_files(&files).unwrap()));
    let mut parser = Parser::new(&model);
    let text = std::fs::read_to_string(format!("{ROOT}fixture.docs")).unwrap();
    let mut bits = 0u64;
    for doc in text.split("\u{1e}\n") {
        let mut ctx = Context::new();
        for line in doc.lines() {
            bits += parser.parse_line(&mut ctx, line).bits;
        }
    }
    let path = format!("{ROOT}expected-bits.txt");
    if std::env::var("SEMEME_BLESS").is_ok() {
        std::fs::write(&path, format!("{bits}\n")).unwrap();
    }
    let want: u64 = std::fs::read_to_string(&path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        bits, want,
        "engine change altered costs: start a new run (see this test's doc)"
    );
}
