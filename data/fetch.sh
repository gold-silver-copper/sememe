#!/usr/bin/env bash
# Downloads sememe's training and evaluation data into this folder.
# Resumable: rerun after an interruption and finished files are skipped.
# Every file except this script and README.md is gitignored.
set -euo pipefail
cd "$(dirname "$0")"

UA="sememe-data-fetch"
HF="https://huggingface.co/datasets"

fetch() { # fetch <url> <dest>
  local url=$1 dest=$2
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && ! -f "$dest.part" ]]; then echo "skip  $dest"; return; fi
  echo "fetch $dest"
  touch "$dest.part"
  curl -fL --retry 10 --retry-delay 5 -C - -A "$UA" -o "$dest" "$url"
  rm -f "$dest.part"
}

# Stage 1: BabyLM 2025 (strict 100M and strict-small 10M), from OSF.
fetch https://osf.io/download/eswy2/ babylm/README.md
fetch https://osf.io/download/y7djq/ babylm/train_10M.zip
fetch https://osf.io/download/ywea7/ babylm/train_100M.zip
fetch https://osf.io/download/uyd7v/ babylm/dev.zip
fetch https://osf.io/download/ftwu3/ babylm/test.zip

# BLiMP grammar minimal pairs (67 phenomena), CC BY 4.0.
fetch https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip blimp/blimp-master.zip

# BabyLM's own evaluation sets (as used for its published scores), from OSF.
fetch https://osf.io/download/s5xjh/ blimp/babylm/blimp_filtered/adjunct_island.jsonl
fetch https://osf.io/download/r24kg/ blimp/babylm/blimp_filtered/anaphor_gender_agreement.jsonl
fetch https://osf.io/download/37dpw/ blimp/babylm/blimp_filtered/anaphor_number_agreement.jsonl
fetch https://osf.io/download/cezf8/ blimp/babylm/blimp_filtered/animate_subject_passive.jsonl
fetch https://osf.io/download/je7mt/ blimp/babylm/blimp_filtered/animate_subject_trans.jsonl
fetch https://osf.io/download/pgaxr/ blimp/babylm/blimp_filtered/causative.jsonl
fetch https://osf.io/download/8gsu3/ blimp/babylm/blimp_filtered/complex_NP_island.jsonl
fetch https://osf.io/download/r2k68/ blimp/babylm/blimp_filtered/coordinate_structure_constraint_complex_left_branch.jsonl
fetch https://osf.io/download/dk3t9/ blimp/babylm/blimp_filtered/coordinate_structure_constraint_object_extraction.jsonl
fetch https://osf.io/download/tbr68/ blimp/babylm/blimp_filtered/determiner_noun_agreement_1.jsonl
fetch https://osf.io/download/681a0460d854624ef30a7bba/ blimp/babylm/blimp_filtered/determiner_noun_agreement_2.jsonl
fetch https://osf.io/download/jg8n5/ blimp/babylm/blimp_filtered/determiner_noun_agreement_irregular_1.jsonl
fetch https://osf.io/download/681a0457720c176b81c0dd8a/ blimp/babylm/blimp_filtered/determiner_noun_agreement_irregular_2.jsonl
fetch https://osf.io/download/681a0458becda878d4c615df/ blimp/babylm/blimp_filtered/determiner_noun_agreement_with_adj_2.jsonl
fetch https://osf.io/download/f2qyd/ blimp/babylm/blimp_filtered/determiner_noun_agreement_with_adj_irregular_1.jsonl
fetch https://osf.io/download/87c6m/ blimp/babylm/blimp_filtered/determiner_noun_agreement_with_adj_irregular_2.jsonl
fetch https://osf.io/download/yh9s6/ blimp/babylm/blimp_filtered/determiner_noun_agreement_with_adjective_1.jsonl
fetch https://osf.io/download/681a044f409790fe07c6146c/ blimp/babylm/blimp_filtered/distractor_agreement_relational_noun.jsonl
fetch https://osf.io/download/681a044fb84cb119f3c614b3/ blimp/babylm/blimp_filtered/distractor_agreement_relative_clause.jsonl
fetch https://osf.io/download/3n9sx/ blimp/babylm/blimp_filtered/drop_argument.jsonl
fetch https://osf.io/download/681a044af6551570e60a7a4e/ blimp/babylm/blimp_filtered/ellipsis_n_bar_1.jsonl
fetch https://osf.io/download/681a0447797fe5af6c0b6a15/ blimp/babylm/blimp_filtered/ellipsis_n_bar_2.jsonl
fetch https://osf.io/download/hjazf/ blimp/babylm/blimp_filtered/existential_there_object_raising.jsonl
fetch https://osf.io/download/jr63p/ blimp/babylm/blimp_filtered/existential_there_quantifiers_1.jsonl
fetch https://osf.io/download/681a0446179cd8348d1595f0/ blimp/babylm/blimp_filtered/existential_there_quantifiers_2.jsonl
fetch https://osf.io/download/9y8k4/ blimp/babylm/blimp_filtered/existential_there_subject_raising.jsonl
fetch https://osf.io/download/681a0440f0a0bdb408c614eb/ blimp/babylm/blimp_filtered/expletive_it_object_raising.jsonl
fetch https://osf.io/download/x9qhv/ blimp/babylm/blimp_filtered/inchoative.jsonl
fetch https://osf.io/download/681a043e51f0f70ab20b6a56/ blimp/babylm/blimp_filtered/intransitive.jsonl
fetch https://osf.io/download/681a043a720c176b81c0dd82/ blimp/babylm/blimp_filtered/irregular_past_participle_adjectives.jsonl
fetch https://osf.io/download/ctxfz/ blimp/babylm/blimp_filtered/irregular_past_participle_verbs.jsonl
fetch https://osf.io/download/681a0437ff83c3467ac0de12/ blimp/babylm/blimp_filtered/irregular_plural_subject_verb_agreement_1.jsonl
fetch https://osf.io/download/681a0435720c176b81c0dd80/ blimp/babylm/blimp_filtered/irregular_plural_subject_verb_agreement_2.jsonl
fetch https://osf.io/download/681a043214ccdc3f661596b6/ blimp/babylm/blimp_filtered/left_branch_island_echo_question.jsonl
fetch https://osf.io/download/673rc/ blimp/babylm/blimp_filtered/left_branch_island_simple_question.jsonl
fetch https://osf.io/download/681a0437f6551570e60a7a48/ blimp/babylm/blimp_filtered/matrix_question_npi_licensor_present.jsonl
fetch https://osf.io/download/681a0433becda878d4c615dc/ blimp/babylm/blimp_filtered/npi_present_1.jsonl
fetch https://osf.io/download/pr58v/ blimp/babylm/blimp_filtered/npi_present_2.jsonl
fetch https://osf.io/download/wxph8/ blimp/babylm/blimp_filtered/only_npi_licensor_present.jsonl
fetch https://osf.io/download/681a046791a6c4b6d41598a9/ blimp/babylm/blimp_filtered/only_npi_scope.jsonl
fetch https://osf.io/download/681a046afb0bb08fb80b6ac2/ blimp/babylm/blimp_filtered/passive_1.jsonl
fetch https://osf.io/download/681a04258e54dcb3cac61526/ blimp/babylm/blimp_filtered/passive_2.jsonl
fetch https://osf.io/download/681a042721ca347e2015967d/ blimp/babylm/blimp_filtered/principle_A_c_command.jsonl
fetch https://osf.io/download/681a042c409790fe07c6145e/ blimp/babylm/blimp_filtered/principle_A_case_1.jsonl
fetch https://osf.io/download/ed2s8/ blimp/babylm/blimp_filtered/principle_A_case_2.jsonl
fetch https://osf.io/download/kfwab/ blimp/babylm/blimp_filtered/principle_A_domain_1.jsonl
fetch https://osf.io/download/681a042a48eb0e2c02c0dd5a/ blimp/babylm/blimp_filtered/principle_A_domain_2.jsonl
fetch https://osf.io/download/7mvc3/ blimp/babylm/blimp_filtered/principle_A_domain_3.jsonl
fetch https://osf.io/download/z9yvq/ blimp/babylm/blimp_filtered/principle_A_reconstruction.jsonl
fetch https://osf.io/download/ft765/ blimp/babylm/blimp_filtered/regular_plural_subject_verb_agreement_1.jsonl
fetch https://osf.io/download/681a041ed18724e30ec0dbeb/ blimp/babylm/blimp_filtered/regular_plural_subject_verb_agreement_2.jsonl
fetch https://osf.io/download/8tvjq/ blimp/babylm/blimp_filtered/sentential_negation_npi_licensor_present.jsonl
fetch https://osf.io/download/681a0416982066dcb5c0de76/ blimp/babylm/blimp_filtered/sentential_negation_npi_scope.jsonl
fetch https://osf.io/download/6sfz7/ blimp/babylm/blimp_filtered/sentential_subject_island.jsonl
fetch https://osf.io/download/681a0410ddab4984e60a7a51/ blimp/babylm/blimp_filtered/superlative_quantifiers_1.jsonl
fetch https://osf.io/download/gst6j/ blimp/babylm/blimp_filtered/superlative_quantifiers_2.jsonl
fetch https://osf.io/download/681a0466ddab4984e60a7a64/ blimp/babylm/blimp_filtered/tough_vs_raising_1.jsonl
fetch https://osf.io/download/681a040e0ce5a8a6880a7cc5/ blimp/babylm/blimp_filtered/tough_vs_raising_2.jsonl
fetch https://osf.io/download/c8xsv/ blimp/babylm/blimp_filtered/transitive.jsonl
fetch https://osf.io/download/681a040991a6c4b6d415989f/ blimp/babylm/blimp_filtered/wh_island.jsonl
fetch https://osf.io/download/681a040c797fe5af6c0b6a01/ blimp/babylm/blimp_filtered/wh_questions_object_gap.jsonl
fetch https://osf.io/download/681a0408409790fe07c61453/ blimp/babylm/blimp_filtered/wh_questions_subject_gap.jsonl
fetch https://osf.io/download/cwt9s/ blimp/babylm/blimp_filtered/wh_questions_subject_gap_long_distance.jsonl
fetch https://osf.io/download/t6va4/ blimp/babylm/blimp_filtered/wh_vs_that_no_gap.jsonl
fetch https://osf.io/download/6yu39/ blimp/babylm/blimp_filtered/wh_vs_that_no_gap_long_distance.jsonl
fetch https://osf.io/download/cnvm2/ blimp/babylm/blimp_filtered/wh_vs_that_with_gap.jsonl
fetch https://osf.io/download/md9vg/ blimp/babylm/blimp_filtered/wh_vs_that_with_gap_long_distance.jsonl
fetch https://osf.io/download/kf2jz/ blimp/babylm/supplement_filtered/hypernym.jsonl
fetch https://osf.io/download/2kcj8/ blimp/babylm/supplement_filtered/qa_congruence_easy.jsonl
fetch https://osf.io/download/zuq8y/ blimp/babylm/supplement_filtered/qa_congruence_tricky.jsonl
fetch https://osf.io/download/se2n9/ blimp/babylm/supplement_filtered/subject_aux_inversion.jsonl
fetch https://osf.io/download/9uyra/ blimp/babylm/supplement_filtered/turn_taking.jsonl

# Stage 3 benchmark: enwik8 / enwik9 (Hutter Prize).
fetch https://mattmahoney.net/dc/enwik8.zip enwik/enwik8.zip
fetch https://mattmahoney.net/dc/enwik9.zip enwik/enwik9.zip

# Stage 4: UN Parallel Corpus v1.0, English-Russian (OPUS, Moses format).
fetch https://object.pouta.csc.fi/OPUS-UNPC/v1.0/moses/en-ru.txt.zip unpc/en-ru.txt.zip

# Stage 3: English Wikipedia, 2023-11-01, plain text (wikimedia/wikipedia).
for i in $(seq -f "%05g" 0 40); do
  fetch "$HF/wikimedia/wikipedia/resolve/main/20231101.en/train-$i-of-00041.parquet" \
        "wikipedia/train-$i-of-00041.parquet"
done

# Stage 2: FineWeb-Edu, the 10B-token sample.
for i in $(seq -f "%03g" 0 13); do
  fetch "$HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/${i}_00000.parquet" \
        "fineweb-edu/sample-10BT/${i}_00000.parquet"
done

echo "all downloads done"
