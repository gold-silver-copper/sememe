//! Records the version hash of the frozen cost code (`src/cost.rs` and the
//! character model it embeds) so every metric can carry it.
use sha2::{Digest, Sha256};

fn main() {
    let mut hasher = Sha256::new();
    for path in ["src/cost.rs", "src/charmodel.tsv"] {
        println!("cargo:rerun-if-changed={path}");
        match std::fs::read(path) {
            Ok(bytes) => hasher.update(&bytes),
            Err(err) => println!("cargo:warning=cannot read {path}: {err}"),
        }
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    println!("cargo:rustc-env=SEMEME_COST_HASH={hex}");
}
