# Verifying the dataset sizes (full-scale smoke test)

`tests/test_full_scale.py` re-derives the size of every official split with the normalized parsers and, when the SALT assets are available, the
SALT coverage. It is skipped unless `AUDIOSET_TOOLS_DATA` points to a folder with the layout below, so the normal test-suite never needs the data.

```bash
# AUDIOSET_TOOLS_SALT_DIR is optional: a folder with salt_event_mapping.tsv and salt_event_roots.json (see audioset-tools fetch-salt)
AUDIOSET_TOOLS_DATA=/path/to/datasets AUDIOSET_TOOLS_SALT_DIR=/path/to/salt pytest -m smoke -s tests/test_full_scale.py
```

Expected layout of `AUDIOSET_TOOLS_DATA` (the files and checksums of the validation run: the official AudioSet segment CSVs, whose headers read `# Segments csv created Sun Mar  5 2017`, the official AudioSet
Strong TSVs and `vggsound.csv` of the VGGSound project):

| File | SHA-256 |
|---|---|
| `AudioSet_meta/eval_segments.csv` | `f7055f53935f70e74e39494a5c8c4c78f1700fce275028bd36e31186f33d45db` |
| `AudioSet_meta/balanced_train_segments.csv` | `3b83e5ff9ed72cfda37d4e36ce08250e107c1ea70f109f6ab6668ac200db5e94` |
| `AudioSet_meta/unbalanced_train_segments.csv` | `bb2dfd5e4f1ced39db9480633c510f813d7899f4b7472aa83a411b4a43a12698` |
| `AudioSet_meta/class_labels_indices.csv` | `cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429` |
| `AudioSet_Strong_meta/audioset_eval_strong.tsv` | `ffda064647a3e7c7eb473d5fcde944f4800c5ce3a7c1f3dfcf131b95dcd30f60` |
| `AudioSet_Strong_meta/audioset_train_strong.tsv` | `29e4646c847c0285cde744cf884843de1fbb647fc24762eb2b244628d276c92b` |
| `AudioSet_Strong_meta/mid_to_display_name.tsv` | `6aeaa2144223933eab75566ca478366ad80e59f381611e308cbe536913d7ddf7` |
| `VGGsound_meta/vggsound.csv` | `c1816c00a237afa4994e873e88f56bac206cbb285fddb05c564184b9c3d6e6ce` |

## Sizes (all asserted by the test)

| Source | Records | Event rows | Official count (file header or file length) | Parse time |
|---|---|---|---|---|
| AudioSet eval | 20,371 | | `num_ytids=20371` in the CSV header | 0.1 s |
| AudioSet balanced train | 22,160 | | `num_ytids=22160` | 0.1 s |
| AudioSet unbalanced train | 2,041,789 | | `num_ytids=2041789` | 10.6 s |
| AudioSet Strong eval | 16,996 segments | 139,538 | 139,538 data rows in the TSV | 2.2 s |
| AudioSet Strong train | 103,463 segments | 934,821 | 934,821 data rows in the TSV | 4.3 s |
| VGGSound | 199,467 | | 199,467 rows in the CSV | 2.9 s |

Times and memory: Apple silicon laptop, Python 3.10.11, numpy 2.2.6, machine otherwise idle, single process; they vary by a factor of about two with the machine load.
The complete smoke run (10 tests, all six files kept in memory) took 21.8 s inside pytest and peaked at about **3.8 GB** (3,841,458,176 bytes, `/usr/bin/time -l`);
loading **only** the unbalanced split takes 10.4 s and about **2.5 GB** (2,450,128,896 bytes).

## SALT coverage (records with at least one SALT label)

With the upstream files at commit `e89d13b65f59bf621539faaf4dd3ad7ff81232c6` (and with the author's cleaned copy of the TSV, which gives the same numbers):

| Source | With >= 1 SALT label | Share |
|---|---|---|
| AudioSet eval | 20,367 of 20,371 | 99.98 % |
| AudioSet Strong eval | 16,996 of 16,996 | 100.00 % |
| VGGSound | 198,775 of 199,467 | 99.65 % |

## Also exercised

`filter_labels(["Speech"])` on the eval split gives 5,324 clips; a three-label `rebalance(random_state=0)` returns 636 clips twice with identical content and order (0.08 s).
The sizes above are the official split sizes of the three sources.

Regenerating this page: `AUDIOSET_TOOLS_DATA=... pytest -m smoke -s` prints one `[smoke]` line per source with the count and the time; the validation record is in [`validation.md`](validation.md).
