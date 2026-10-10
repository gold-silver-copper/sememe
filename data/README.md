# Data

Training and evaluation text for sememe. `fetch.sh` downloads everything here
and can be rerun to resume. Everything in this folder except `fetch.sh` and
this README is gitignored: the data is never committed.

Downloaded 2026-10-09: 41 GB in total, all archives tested and parquet files checked.

| Folder | What | Size | Source | License |
|---|---|---|---|---|
| `babylm/` | BabyLM 2025 text data (the archives are dated 2024-04-08: the same text as the 2024 release): `train_10M`, `train_100M`, `dev`, `test`. Child-directed speech, children's books, Simple Wikipedia, subtitles, dialogue | ~235 MB zipped | <https://osf.io/ryjfm/> | Mixed by source. Research use; see `babylm/README.md` |
| `blimp/` | BLiMP: 67 sets of 1,000 grammatical/ungrammatical sentence pairs | small | <https://github.com/alexwarstadt/blimp> | CC BY 4.0 |
| `blimp/babylm/` | BabyLM's own evaluation sets: `blimp_filtered` (BLiMP restricted to BabyLM's vocabulary, 67 files) and `supplement_filtered` (5 extra tests). Used for BabyLM's published scores | small | <https://osf.io/ryjfm/> | as BLiMP |
| `fineweb-edu/sample-10BT/` | FineWeb-Edu, 10B-token sample (14 parquet files), educational web pages, each with an educational-quality score | ~28.5 GB | <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu> | ODC-By 1.0, plus Common Crawl's terms of use |
| `wikipedia/` | English Wikipedia, 2023-11-01 dump, plain text (41 parquet files) | ~11.6 GB | <https://huggingface.co/datasets/wikimedia/wikipedia> | CC BY-SA 4.0 / GFDL |
| `enwik/` | enwik8 (100 MB) and enwik9 (1 GB): the start of an English Wikipedia XML dump, the Hutter Prize benchmark | ~0.36 GB zipped | <https://mattmahoney.net/dc/textdata.html> | Wikipedia's (CC BY-SA) |
| `unpc/` | UN Parallel Corpus v1.0, English–Russian, sentence-aligned (Moses format) | ~2.6 GB zipped | <https://opus.nlpl.eu/UNPC> | UN terms: free use with attribution (see OPUS) |

Not downloaded:

- **The other UN languages** (Chinese, Arabic, French, Spanish) are on the
  same OPUS page when needed.
- **The 2026 BabyLM English and Russian sets on Hugging Face**
  (`BabyLM-community/babylm-eng`, `babylm-rus`) are gated behind a login, so
  the open 2025 release on OSF is used instead.
