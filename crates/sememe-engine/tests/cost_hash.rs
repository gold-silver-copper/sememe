//! The cost code is frozen for a run: CI fails if its hash differs from the
//! one the active run was started with (`runs/current-run.json`).

#[test]
fn cost_code_matches_the_active_run() {
    let text = include_str!("../../../runs/current-run.json");
    let want = text
        .split("\"cost_hash\"")
        .nth(1)
        .and_then(|s| s.split('"').nth(1))
        .unwrap_or_default();
    assert_eq!(
        sememe_engine::cost::COST_HASH,
        want,
        "cost.rs changed: start a new run and record its hash in runs/current-run.json"
    );
}
