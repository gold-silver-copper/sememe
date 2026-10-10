# sememe

A compiler from text to a readable language model.

An LLM is trained by compressing text: training searches for the weights that
make the corpus cheapest to write down. sememe uses the same objective, but the
model is a **program** instead of weights:

> total cost = size of the program + bits needed to write down all the
> training text using the program

The program is a typed ontology, a combinatory categorial grammar lexicon,
facts and inference rules, each with a readable name. It is learned by
counting (EM), by an Opus teacher that writes down what sentences mean, and by
Opus agents that refactor the program. The learned program is committed as
text in `program/` and exported as a Rust crate in `crates/sememe-model/`.

The working spec is `prompts/text-to-code-compiler.md`.
