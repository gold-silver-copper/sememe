# References

Research related to sememe, downloaded October 2026. The PDFs are
gitignored, since most of them can't be redistributed. Each one can be
downloaded again from the link listed with it.

## Understanding English with hand-written code

- `1971-winograd-shrdlu.pdf`: Terry Winograd, *Procedures as a Representation
  for Data in a Computer Program for Understanding Natural Language* (MIT PhD
  thesis, SHRDLU). <https://dspace.mit.edu/handle/1721.1/7095>. Scanned, no
  text layer.
  The closest ancestor. It reads English commands and questions about a
  small world, acts, answers and explains itself, all with hand-written
  procedures. It is impressive inside its world, it was never tested on input
  its author hadn't seen, and it never grew beyond that world.
- `2000-copestake-flickinger-english-resource-grammar.pdf`: Copestake &
  Flickinger, *An open source grammar development environment and
  broad-coverage English grammar using HPSG* (LREC 2000).
  <https://aclanthology.org/L00-1276/>
  A large hand-written English grammar that both parses and generates through
  a structured meaning representation. Shows what broad symbolic coverage
  takes.
- `2023-lenat-marcus-what-llms-might-learn-from-cyc.pdf`: Lenat & Marcus,
  *Getting from Generative AI to Trustworthy AI: What LLMs might learn from
  Cyc*. <https://arxiv.org/abs/2308.04445>
  The Cyc project's own account of 40 years of hand-built knowledge, and what
  it thinks symbolic systems offer that LLMs don't.
- `2019-sutton-bitter-lesson.html`: Rich Sutton, *The Bitter Lesson*.
  <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>
  The main argument against this project: general methods that scale with
  computing power have so far beaten hand-built knowledge.

## Grammars that grow from use

- `2015-wang-berant-liang-semantic-parser-overnight.pdf`: Wang, Berant &
  Liang, *Building a Semantic Parser Overnight* (ACL 2015).
  <https://aclanthology.org/P15-1129/>
  A small grammar plus paraphrases written for the task, used to learn a
  parser. Also a warning: paraphrases written for a task don't sound like real
  users.
- `2017-wang-et-al-naturalizing-programming-language-voxelurn.pdf`: Wang,
  Ginn, Liang & Manning, *Naturalizing a Programming Language via Interactive
  Learning* (ACL 2017). <https://aclanthology.org/P17-1086/>
  A command language whose grammar grows as users teach it new phrasings.
- `2023-wang-et-al-grammar-prompting.pdf`: Wang et al., *Grammar Prompting for
  Domain-Specific Language Generation with Large Language Models* (NeurIPS
  2023). <https://arxiv.org/abs/2305.19234>
  An LLM picks the minimal subset of an expert-written grammar for each
  example, then generates within it. It shows LLMs can work well with formal
  grammars, but they choose rules here; they don't write new ones.

## LLMs growing programs under an automatic scorer

- `2023-romera-paredes-et-al-funsearch.pdf`: Romera-Paredes et al.,
  *Mathematical discoveries from program search with large language models*
  (Nature 2023, FunSearch).
  <https://www.nature.com/articles/s41586-023-06924-6>
- `2025-novikov-et-al-alphaevolve.pdf`: Novikov et al., *AlphaEvolve: A coding
  agent for scientific and algorithmic discovery*.
  <https://arxiv.org/abs/2506.13131>. Open-source version: OpenEvolve,
  <https://github.com/codelion/openevolve>
  An LLM proposes code changes and a scorer keeps the better ones, in a loop
  with no human in between. The model for sememe's improvement loop.
- `2023-wang-et-al-voyager.pdf`: Wang et al., *Voyager: An Open-Ended Embodied
  Agent with Large Language Models*. <https://arxiv.org/abs/2305.16291>
  An agent stores what it learns as a growing library of code skills, kept
  only after it checks that they work. The closest to "knowledge stored as a
  codebase that LLMs grow".
- `2021-ellis-et-al-dreamcoder.pdf`: Ellis et al., *DreamCoder: Growing
  generalizable, interpretable knowledge with wake-sleep Bayesian program
  learning* (PLDI 2021). <https://arxiv.org/abs/2006.08381>
- `2023-grand-et-al-lilo.pdf`: Grand et al., *LILO: Learning Interpretable
  Libraries by Compressing and Documenting Code* (ICLR 2024).
  <https://arxiv.org/abs/2310.19791>
  Learn reusable helper functions, keeping those that shrink the total of the
  library plus every solution rewritten to use it (the library itself grows).
  Also shows that unnamed helpers hurt LLMs and that auto-written docs fix
  this. The basis for sememe's size-for-gain acceptance rule and its
  compression rounds.

## Turning LLM or neural knowledge into readable rules and programs

- `2024-wang-et-al-llms-are-interpretable-learners.pdf`: Wang et al., *Large
  Language Models are Interpretable Learners*.
  <https://arxiv.org/abs/2406.17224>
- `2026-distilling-asp-theories-from-llms.pdf`: Higuera Ruiz, Hofmarcher et
  al., *Distilling Answer Set Programming Theories from Large Language Models*
  (NeSy 2026). <https://arxiv.org/abs/2607.28086>
  An LLM's knowledge extracted as readable logic rules. The closest current
  work to "the LLM is the teacher, the code is what's learned", on small
  tasks.
- `2026-symtorch-symbolic-distillation.pdf`: Tan, Soubki & Cranmer, *SymTorch:
  Symbolic Distillation of Neural Networks*.
  <https://arxiv.org/abs/2602.21307>
- `2021-weiss-et-al-thinking-like-transformers-rasp.pdf`: Weiss, Goldberg &
  Yahav, *Thinking Like Transformers* (ICML 2021, RASP).
  <https://arxiv.org/abs/2106.06981>
- `2023-lindner-et-al-tracr.pdf`: Lindner et al., *Tracr: Compiled
  Transformers as a Laboratory for Interpretability*.
  <https://arxiv.org/abs/2301.05062>
- `2023-friedman-et-al-learning-transformer-programs.pdf`: Friedman, Wettig &
  Chen, *Learning Transformer Programs* (NeurIPS 2023).
  <https://arxiv.org/abs/2306.01128>
  Neural networks built so they can be converted into readable programs, or
  the reverse: the "decompile the model" path sememe doesn't take.

## Universal grammar and cross-linguistic structure

- `2002-bender-flickinger-oepen-grammar-matrix.pdf`: Bender, Flickinger &
  Oepen, *The Grammar Matrix* (COLING Workshop on Grammar Engineering 2002).
  <https://aclanthology.org/W02-1502/>
  A shared core of grammar code, taken from the English Resource Grammar,
  that every new language's grammar starts from, adding only what is specific
  to that language. The direct model for per-language grammar crates over one
  shared base.
- `2004-ranta-grammatical-framework.pdf`: Aarne Ranta, *Grammatical
  Framework: A Type-Theoretical Grammar Formalism* (Journal of Functional
  Programming 2004). <https://www.cse.chalmers.se/~aarne/articles/gf-jfp.pdf>.
  No usable text layer.
  One typed abstract syntax holds the meaning, and each language has a
  concrete syntax mapping it to and from text, with the same grammar used to
  parse and to generate. The closest existing design to sememe's typed
  meaning layer plus one grammar crate per language.
- `2009-ranta-gf-resource-grammar-library.pdf`: Aarne Ranta, *The GF Resource
  Grammar Library* (LiLT 2009).
  <https://journals.colorado.edu/index.php/lilt/article/view/1205>
  Grammars for many languages behind one shared interface, so application
  code is written once for all of them. Shows how much grammar can be shared
  across languages and what has to stay separate.
- `2009-angelov-incremental-parsing-pmcfg.pdf`: Krasimir Angelov,
  *Incremental Parsing with Parallel Multiple Context-Free Grammars* (EACL
  2009). <https://aclanthology.org/E09-1009/>
  GF's parser, which reads one word at a time and can list every word that
  may come next. The technique behind grammar-driven Tab completion. The
  later runtime-format paper (Angelov, Bringert & Ranta, *PGF*, JoLLI 2010,
  <https://doi.org/10.1007/s10849-009-9112-y>) has no open copy.
- `2021-de-marneffe-et-al-universal-dependencies.pdf`: de Marneffe, Manning,
  Nivre & Zeman, *Universal Dependencies* (Computational Linguistics 2021).
  <https://aclanthology.org/2021.cl-2.11/>
  One set of word classes and grammatical relations used across 100+
  languages. The main working example of a cross-linguistic inventory, and
  of where it has to give way to language-specific detail.
- `2010-haspelmath-comparative-concepts.pdf`: Martin Haspelmath,
  *Comparative concepts and descriptive categories in crosslinguistic
  studies* (Language 2010). <https://zenodo.org/records/1303024>
  A language's own categories (English "adjective") are a different kind of
  thing from the concepts used to compare languages. Supports keeping
  language-neutral meaning types separate from each grammar crate's
  categories.
- `2009-evans-levinson-myth-of-language-universals.pdf`: Evans & Levinson,
  *The myth of language universals* (Behavioral and Brain Sciences 2009,
  with peer commentaries). <https://doi.org/10.1017/S0140525X0999094X>
  The strongest case that very few grammatical features hold in every
  language. A warning against an English-shaped "neutral" layer.
- `2019-chomsky-gallego-ott-generative-grammar-faculty-of-language.pdf`:
  Chomsky, Gallego & Ott, *Generative Grammar and the Faculty of Language:
  Insights, Questions, and Challenges* (Catalan Journal of Linguistics 2019).
  <https://lingbuzz.net/lingbuzz/003507>
  A recent overview of the universal-grammar position: a small shared core of
  combining operations, with differences between languages placed in the
  lexicon and the sound system. Read it against Evans & Levinson.
- `2021-wierzbicka-semantic-primitives-fifty-years-later.pdf`: Anna
  Wierzbicka, *"Semantic Primitives", fifty years later* (Russian Journal of
  Linguistics 2021). <https://doi.org/10.22363/2687-0088-2021-25-2-317-342>
  About 65 "semantic primes" (SOMEONE, DO, WANT, BECAUSE…) claimed to exist as
  words in every language and able to define everything else. A tested
  candidate for the smallest building blocks of an interlingua.

## Language-neutral meaning representations

- `2013-banarescu-et-al-amr.pdf`: Banarescu et al., *Abstract Meaning
  Representation for Sembanking* (Linguistic Annotation Workshop 2013).
  <https://aclanthology.org/W13-2322/>
  Sentence meaning as a small graph of concepts and roles, independent of
  word order and syntax. Widely used, but English-centric.
- `2021-van-gysel-et-al-uniform-meaning-representation.pdf`: Van Gysel et
  al., *Designing a Uniform Meaning Representation for Natural Language
  Processing* (KI 2021). <https://par.nsf.gov/biblio/10288269>
  AMR extended to work across languages, adding tense, aspect, modality and
  links across a whole document, and stating where languages may differ. The
  most current design for a typed language-neutral meaning layer.
- `2005-copestake-et-al-minimal-recursion-semantics.pdf`: Copestake,
  Flickinger, Pollard & Sag, *Minimal Recursion Semantics: An Introduction*
  (Research on Language and Computation 2005).
  <https://www.cl.cam.ac.uk/~aac10/papers/mrs.pdf>
  The meaning format the English Resource Grammar and Grammar Matrix produce
  and generate from. It can leave choices such as quantifier scope open,
  which the meaning layer needs.
- `2017-abzianidze-et-al-parallel-meaning-bank.pdf`: Abzianidze et al., *The
  Parallel Meaning Bank* (EACL 2017). <https://aclanthology.org/E17-2039/>
  Translations in English, German, Italian and Dutch annotated with one
  shared formal meaning. A test of "same meaning, different language".
- `2020-vrandecic-multilingual-wikipedia.pdf`: Denny Vrandečić,
  *Architecture for a Multilingual Wikipedia* (2020).
  <https://arxiv.org/abs/2004.04733>
  Abstract Wikipedia: content stored once as language-neutral constructors
  over Wikidata items, with one renderer per language. A large current
  attempt at sememe's architecture, in the generating direction.
- `1992-nyberg-mitamura-kant.pdf`: Nyberg & Mitamura, *The KANT System*
  (COLING 1992). <https://aclanthology.org/C92-3168/>
  Interlingua machine translation that shipped, because it was limited to
  one domain and a controlled input language. Evidence that a defined world
  makes a neutral layer much easier.
- `2013-bond-foster-open-multilingual-wordnet.pdf`: Bond & Foster, *Linking
  and Extending an Open Multilingual Wordnet* (ACL 2013).
  <https://aclanthology.org/P13-1133/>
  Wordnets in 150+ languages linked through one shared index of concepts. A
  model for word meanings each language's lexicon points to (research only:
  sememe doesn't use WordNet data).
- `1969-schank-tesler-conceptual-dependency-parser.pdf`: Schank & Tesler, *A
  Conceptual Dependency Parser for Natural Language* (COLING 1969).
  <https://aclanthology.org/C69-0201/>
  Hand-written parsing straight into language-independent meaning, later
  reduced to a small set of action primitives. An ancestor of the meaning
  layer, showing both the appeal and the limits of a fixed primitive set.

## Ontologies

- `1993-gruber-translation-approach-ontologies.pdf`: Tom Gruber, *A
  Translation Approach to Portable Ontology Specifications* (Knowledge
  Acquisition 1993). <http://tomgruber.org/writing/ontolingua-kaj-1993.pdf>
  The standard definition of an ontology ("an explicit specification of a
  conceptualization"), and sharing one across systems by translation.
- `1998-guarino-formal-ontology.pdf`: Nicola Guarino, *Formal Ontology and
  Information Systems* (FOIS 1998).
  <http://www.loa.istc.cnr.it/old/Papers/FOIS98.pdf>
  Top-level, domain, task and application ontologies, and how each should
  depend on the others. Maps onto the meaning layer versus each world's kinds.
- `2001-niles-pease-sumo.pdf`: Niles & Pease, *Towards a Standard Upper
  Ontology* (FOIS 2001).
  <https://www.ontologyportal.org/professional/FOIS.pdf>
  SUMO: a large free top-level ontology in logic, mapped to WordNet. A worked
  example of general things, events and relations a lexicon can link into.
- `2002-gangemi-et-al-dolce.pdf`: Gangemi, Guarino, Masolo, Oltramari &
  Schneider, *Sweetening Ontologies with DOLCE* (EKAW 2002).
  <http://www.loa.istc.cnr.it/old/Papers/DOLCE-EKAW.pdf>
  A top-level ontology built around the categories of natural language and
  human thinking rather than physics. The closest one to a meaning layer
  behind language.
- `2022-otte-beverley-ruttenberg-bfo.pdf`: Otte, Beverley & Ruttenberg,
  *BFO: Basic Formal Ontology* (Applied Ontology 2022, authors' preprint).
  <https://philarchive.org/rec/OTTBBF>
  About 36 top-level classes, an ISO standard, extended by 350+ domain
  ontologies. Shows how small a shared top level can be.
- `2014-vrandecic-krotzsch-wikidata.pdf`: Vrandečić & Krötzsch, *Wikidata: A
  Free Collaborative Knowledgebase* (CACM 2014).
  <https://iccl.inf.tu-dresden.de/web/Article4002>
  Items are language-neutral IDs with a label in each language: things kept
  separate from the words for them. A likely future world.

Some of these files are the authors' preprints or accepted manuscripts
rather than the final published layout (Gruber, Guarino, Van Gysel, BFO,
Ranta 2004). The SUMO and BFO files come from the Internet Archive's saved
copies of the links above, which are behind bot checks.

## Learning a grammar from data: CCG induction, EM and MDL

- `2011-steedman-baldridge-combinatory-categorial-grammar.pdf`: Steedman &
  Baldridge, *Combinatory Categorial Grammar* (chapter in Borsley & Börjars,
  eds., *Non-Transformational Syntax*, Blackwell 2011; authors' copy).
  <https://homepages.inf.ed.ac.uk/steedman/papers/ccg/SteedmanBaldridgeNTSyntax.pdf>
  The standard introduction to CCG: every word carries a category saying
  what it combines with, and a handful of fixed combining rules (application,
  composition, type-raising) do the rest. This is the design of sememe's
  fixed engine, where all the language-specific knowledge sits in the lexicon.
- `2005-zettlemoyer-collins-ccg-logical-form.pdf`: Zettlemoyer & Collins,
  *Learning to Map Sentences to Logical Form: Structured Classification with
  Probabilistic Categorial Grammars* (UAI 2005).
  <https://arxiv.org/abs/1207.1420>
  The first system to learn a CCG lexicon from sentences paired with their
  meanings. It proposes candidate word entries from a small set of
  hand-written templates and keeps the ones that help parse the training
  pairs. The starting point for sememe's mechanical lexicon learner.
- `2010-kwiatkowski-et-al-ccg-higher-order-unification.pdf`: Kwiatkowski,
  Zettlemoyer, Goldwater & Steedman, *Inducing Probabilistic CCG Grammars
  from Logical Form with Higher-Order Unification* (EMNLP 2010).
  <https://aclanthology.org/D10-1119/>
  Removes the hand-written templates: it splits each sentence's meaning into
  pieces for its words by working backwards through the CCG rules, so the
  same learner works across languages. The closest published method to
  inducing lexicon entries from (sentence, meaning) pairs with no
  language-specific setup.
- `2011-kwiatkowski-et-al-ccg-lexical-generalization.pdf`: Kwiatkowski,
  Zettlemoyer, Goldwater & Steedman, *Lexical Generalization in CCG Grammar
  Induction for Semantic Parsing* (EMNLP 2011).
  <https://aclanthology.org/D11-1140/>
  Splits each lexicon entry into a word stem and a shared template, so a
  pattern learned for one word carries over to words seen rarely. A
  direct way to make a learned lexicon smaller, which is what an MDL score
  rewards.
- `2013-artzi-zettlemoyer-weakly-supervised-semantic-parsing.pdf`: Artzi &
  Zettlemoyer, *Weakly Supervised Learning of Semantic Parsers for Mapping
  Instructions to Actions* (TACL 2013). <https://aclanthology.org/Q13-1005/>
  Learns a CCG parser when the training signal is only whether the parsed
  instruction does the right thing, not a written-out meaning. Relevant if
  sememe ever checks meanings by running them in a world instead of trusting
  the LLM's written meanings.
- `2012-bisk-hockenmaier-ccg-grammar-induction.pdf`: Bisk & Hockenmaier,
  *Simple Robust Grammar Induction with Combinatory Categorial Grammars*
  (AAAI 2012). <https://doi.org/10.1609/aaai.v26i1.8355>
  Learns a CCG from part-of-speech-tagged text with no meanings at all,
  starting from a few general principles and using EM to estimate how likely
  each category is. Shows how far text without meanings gets a CCG learner,
  and how EM over rule counts behaves on it.
- `1998-friedman-bayesian-structural-em.pdf`: Nir Friedman, *The Bayesian
  Structural EM Algorithm* (UAI 1998). <https://arxiv.org/abs/1301.7373>
  EM that changes the model's structure as well as its numbers: each round
  fills in hidden information with the current model, then picks the
  structure with the best score given that filled-in data. The theory behind
  sememe alternating EM on rule counts with score-based decisions about
  which lexicon entries exist.
- `1996-de-marcken-unsupervised-language-acquisition.pdf`: Carl de Marcken,
  *Unsupervised Language Acquisition* (MIT PhD thesis 1996).
  <https://arxiv.org/abs/cmp-lg/9611002>
  Learns a hierarchical lexicon of words and phrases from raw text by
  minimizing the bits to store the lexicon plus the bits to encode the text
  with it, using EM inside an add-and-delete search. The closest ancestor of
  sememe's whole objective, and a careful account of how such a search goes
  wrong.
- `1996-grunwald-mdl-grammar-inference` (link only, no open copy): Peter
  Grünwald, *A Minimum Description Length Approach to Grammar Inference*
  (in Wermter, Riloff & Scheler, eds., *Connectionist, Statistical and
  Symbolic Approaches to Learning for Natural Language Processing*, LNAI
  1040, Springer 1996). <https://doi.org/10.1007/3-540-60925-3_48>
  Grammar learning scored as grammar size plus the size of the corpus
  encoded with it, applied to word classes. The author's old FTP copy is
  gone and not archived.
- `2004-grunwald-mdl-tutorial.pdf`: Peter Grünwald, *A Tutorial Introduction
  to the Minimum Description Length Principle* (in *Advances in Minimum
  Description Length*, MIT Press 2005). <https://arxiv.org/abs/math/0406077>
  The open substitute for the paper above: what "bits for the model plus bits
  for the data" means precisely, and when it can be trusted. The reference for
  how sememe should count program bits so the score isn't gamed.
- `2002-creutz-lagus-unsupervised-morpheme-discovery.pdf`: Creutz & Lagus,
  *Unsupervised Discovery of Morphemes* (ACL SIGPHON workshop 2002).
  <https://aclanthology.org/W02-0603/>
  Splits words into stems and endings by picking the set of pieces that
  makes the piece list plus the text encoded with it shortest. The method
  sememe's MDL segmentation of word forms follows. The longer journal
  version (Creutz & Lagus, *Unsupervised Models for Morpheme Segmentation
  and Morphology Learning*, ACM TSLP 2007,
  <https://doi.org/10.1145/1187415.1187418>) has no open copy.
- `2013-virpioja-et-al-morfessor-2.pdf`: Virpioja, Smit, Grönroos & Kurimo,
  *Morfessor 2.0: Python Implementation and Extensions for Morfessor
  Baseline* (Aalto University technical report 2013).
  <http://urn.fi/URN:ISBN:978-952-60-5501-5>
  The current reference for Morfessor Baseline, with the exact cost formulas,
  the search, and a weight for trading lexicon size against corpus cost. A
  practical guide for implementing and tuning sememe's word-form learner.
- `2023-deletang-et-al-language-modeling-is-compression.pdf`: Delétang et
  al., *Language Modeling Is Compression* (ICLR 2024).
  <https://arxiv.org/abs/2309.10668>
  Any model that predicts text can compress it, and the bits it needs are a
  fair way to compare models of very different kinds. It also shows that
  counting a large model's own size changes which model wins. The basis for
  comparing sememe with neural models on bits per text, model size included.
- `2012-kwiatkowski-et-al-probabilistic-syntactic-semantic-acquisition.pdf`:
  Kwiatkowski, Goldwater, Zettlemoyer & Steedman, *A Probabilistic Model of
  Syntactic and Semantic Acquisition from Child-Directed Utterances and their
  Meanings* (EACL 2012). <https://aclanthology.org/E12-1024/>
  A generative CCG model that learns words and grammar from child-directed
  speech paired with meanings. The closest existing model to sememe's
  P(text | meaning), trained on the same kind of text as stage 1.
- `2002-hockenmaier-steedman-generative-ccg.pdf`: Hockenmaier & Steedman,
  *Generative Models for Statistical Parsing with Combinatory Categorial
  Grammar* (ACL 2002). <https://aclanthology.org/P02-1043/>
  How to give a CCG derivation a probability top-down, which is what turns a
  CCG lexicon into a model that can count bits.
- `1996-eisner-ccg-normal-form-parsing.pdf`: Jason Eisner, *Efficient
  Normal-Form Parsing for Combinatory Categorial Grammar* (ACL 1996).
  <https://aclanthology.org/P96-1011/>
  CCG allows many equivalent derivations of the same reading. Normal-form
  derivations keep one per reading, so no probability (and no bits) is wasted
  on duplicates.
