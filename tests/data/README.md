# Test data

Small excerpts of the official metadata files (no audio) and synthetic SALT files. They make the test-suite self-contained.

| File | Origin | Rows |
|---|---|---|
| `audioset/eval_segments_excerpt.csv` | `AudioSet_meta/eval_segments.csv` (the three `#` comment lines + the first 30 segments) | 30 |
| `audioset/class_labels_indices_excerpt.csv` | rows of `class_labels_indices.csv` whose MID occurs in the excerpt above | 51 |
| `audioset_strong/audioset_eval_strong_excerpt.tsv` | `AudioSet_Strong_meta/audioset_eval_strong.tsv` (header + the first 7 whole segments covering at least 40 event rows) | 43 events |
| `audioset_strong/mid_to_display_name_excerpt.tsv` | rows of `mid_to_display_name.tsv` used by the excerpt above | 20 |
| `vggsound/vggsound_excerpt.csv` | first 40 rows of `VGGsound_meta/vggsound.csv` | 40 |
| `salt/salt_event_mapping.tsv`, `salt/salt_event_roots.json` | **synthetic**, written for this repository (see `salt/README.md`) | 41 rows / small hierarchy |

`make_fixtures.py` regenerates the first five groups from the full files: `python tests/data/make_fixtures.py /path/to/datasets` (the folder contains
`AudioSet_meta`, `AudioSet_Strong_meta`, `VGGsound_meta`; the excerpts of this repository were produced from the official files as kept in `gpat_framework/datasets`; the AudioSet CSV headers read `# Segments csv created Sun Mar  5 2017`). Origin, attribution and terms: [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
