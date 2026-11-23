# Synthetic SALT fixtures

`salt_event_mapping.tsv` and `salt_event_roots.json` are **synthetic**: small hand-written files with the same
layout as the SALT assets (`standard_event`, `dataset_label`, `dataset`; nested dictionary of standard events), the
real dataset keys (`AudioSet`, `AudioSetStrong`, `VggSound`, `Freesound50k`) and invented, generic standard-event
names. They are not an excerpt of the SALT files, which are not distributed with this repository (the upstream
repository declares no licence; see `NOTICE` and `docs/salt.md`). The last rows of the TSV are deliberately
malformed (case/whitespace variant, missing standard event, tab-only and one-column rows) to test the loader.
