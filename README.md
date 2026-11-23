<p align="center">
  <img src="assets/disim_logo.png" alt="University of L'Aquila - Department of Information Engineering, Computer Science and Mathematics (DISIM)" width="280">
</p>

# AudioSet-Tools v2

*Taxonomy-aware curation of AudioSet, AudioSet Strong and VGGSound metadata: one normalized record model, SALT label spaces, deterministic rebalancing, lossless exports, event tracks and a YouTube clip downloader.*

[![CI](https://github.com/StefanoGiacomelli/audioset-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/StefanoGiacomelli/audioset-tools/actions/workflows/ci.yml)
[![article](https://img.shields.io/badge/article-10.1186%2Fs13636--025--00436--z-blue)](https://doi.org/10.1186/s13636-025-00436-z)
![python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776AB)

> **Status.** This is the standalone **version 2** of the framework. Version 1 (the code of the published article and of the AudioSet-EV v1 case study, with its data, loaders and notebook) is preserved under the **`v1.0.0` tag** and in the Zenodo records of the data; the mapping of every v1 function and script to v2 is in [`docs/MIGRATION_FROM_V1.md`](docs/MIGRATION_FROM_V1.md). Licence: **GNU GPLv3**, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

AudioSet-Tools v2 moves the abstraction boundary of the framework upstream: each supported metadata source is parsed once into a shared representation (`AudioRecord`, `TemporalEvent`, `AudioDataset`) and every later operation (label-aware filtering, rebalancing, merging, statistics, export, event tracks, retrieval jobs) acts on that representation instead of on schema-specific branches. It implements the architecture presented in the EUSIPCO 2026 poster (see [Related publications](#related-publications-and-how-to-cite)). It was extracted from the prototype in `gpat_framework/taxonomy/audioset_tools`, verified against it on the complete official metadata (65 of 65 compared results identical, [`docs/validation.md`](docs/validation.md)) and completed with tests, fixes and the retrieval features of v1.

## What you can do

* **Parse** AudioSet Standard (official CSVs and curated v1 CSVs), AudioSet Strong (TSV), VGGSound (CSV) and the package's own normalized JSON/CSV; the format is detected automatically.
* **Filter** in four query spaces — `id` (MIDs), `original` (display names), `salt` (standardized events, with optional descendant expansion) and `auto` — with `any` / `all` / `none` matching.
* **Rebalance** multi-label datasets deterministically (`random_state`), with documented multi-label semantics; select ranges, group, count, look up YouTube ids, deduplicate and merge on a stable `record_id`.
* **Export** lossless normalized JSON/CSV; derive binary event tracks and piano rolls from Strong annotations.
* **Retrieve** clips with `YouTubeDatasetDownloader` (yt-dlp + resampy + soundfile): resampling, trimming, channel policy, JSON reports, and — opt-in — resume by file, random pauses, shadow-ban / consecutive-failure halts, cookies and a cookie-refresh hook.

## Supported sources

| Source | Files | Record |
|---|---|---|
| AudioSet Standard | `eval_segments.csv`, `balanced_train_segments.csv`, `unbalanced_train_segments.csv` (+ `class_labels_indices.csv`); curated v1 CSVs, also with a `downloaded` column | one clip, several labels |
| AudioSet Strong | `audioset_eval_strong.tsv`, `audioset_train_strong.tsv` (+ `mid_to_display_name.tsv`) | one segment with its `TemporalEvent`s |
| VGGSound | `vggsound.csv` | one 10-s clip, one label, split |
| Normalized | JSON / CSV written by `AudioDataset.to_json` / `to_csv` | the stored records |
| SALT (optional label space) | `salt_event_mapping.tsv`, `salt_event_roots.json` — fetched on demand, **not distributed here** | `salt_labels` of every record |

The metadata files are not included (only excerpts for the tests, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)); the official sizes (20,371 / 22,160 / 2,041,789 AudioSet clips, 16,996 / 103,463 Strong segments with 139,538 / 934,821 events, 199,467 VGGSound rows) are re-derived by the optional full-scale smoke test.

## Installation

Python >= 3.10. Core dependencies: `numpy`, `tqdm`.

```bash
git clone https://github.com/StefanoGiacomelli/audioset-tools.git
cd audioset-tools
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,download,plot]"      # extras: dev, download, plot, torch, salt (see the table below)
```

| Extra | Adds | Needed for |
|---|---|---|
| `download` | `yt-dlp`, `soundfile`, `resampy` (+ the `ffmpeg` executable on `PATH`) | `YouTubeDatasetDownloader` with `dry_run=False` |
| `plot` | `matplotlib` >= 3.6 | `plot_events_pianoroll` |
| `torch` | `torch` | `generate_event_tracks(..., out_type="pt")` |
| `salt` | the author's `py-salt` fork, pinned (Python < 3.12, numpy < 2) | `SaltResolver.from_py_salt()` only |
| `dev` | `pytest`, `ruff`, `build`, `nbstripout` | tests, lint, packaging |

SALT assets (optional): `audioset-tools fetch-salt` ([`docs/salt.md`](docs/salt.md)).

## Quick start

```bash
pytest                                   # offline test-suite on the excerpts of tests/data (no network, no other repository)
```

```python
from audioset_tools import AudioDataset, SaltResolver

salt = SaltResolver.from_mapping_file("tests/data/salt/salt_event_mapping.tsv",   # synthetic SALT-like files for the demo
                                      "tests/data/salt/salt_event_roots.json")
ds = AudioDataset.from_file("tests/data/vggsound/vggsound_excerpt.csv", salt_resolver=salt)
print(ds)
vehicles = ds.filter_labels(["vehicle"], label_space="salt", include_descendants=True)
print(len(vehicles), sorted({r.original_labels[0] for r in vehicles}))
vehicles.to_csv("vehicles.csv")                                    # lossless: loads back to equal records
print(AudioDataset.from_file("vehicles.csv").records == vehicles.records)
print(vehicles[0].source_id, vehicles[0].record_id)                # stable SHA-1 content identifier
print(ds.rebalance(focus_labels=["foghorn", "wind noise"], random_state=0).label_counts())
```

Expected output:

```
AudioDataset(records=40, source_format='VggSound', salt_resolver=yes)
5 ['airplane', 'ambulance siren', 'car engine starting']
True
--8puiAGLhs_30 a32f04bdcb8c1cb234191120359b19acb6f244b3
Counter({'foghorn': 1, 'wind noise': 1})
```

More: [`docs/usage.md`](docs/usage.md) (formats, query spaces, operations, export, events), [`examples/audioset_tools_demo.ipynb`](examples/audioset_tools_demo.ipynb) (executed copy with outputs: [`docs/audioset_tools_demo_executed.ipynb`](docs/audioset_tools_demo_executed.ipynb)).

## Repository layout

```
audioset-tools/
├── src/audioset_tools/      # the package: models, parsers, dataset, filters, utils, events, salt, downloader, cli (+ py.typed)
├── tests/                   # pytest suite; tests/data = excerpts of the official files + synthetic SALT files
├── examples/                # demo notebook (outputs stripped)
├── docs/                    # usage, api, downloader, salt, rebalancing, MIGRATION_FROM_V1, validation, verify_dataset_statistics, executed notebook
├── assets/                  # DISIM and funding logos
├── .github/workflows/ci.yml # ruff + pytest on Python 3.10-3.13, wheel build job
├── pyproject.toml, LICENSE, CITATION.cff, CHANGELOG.md, NOTICE, THIRD_PARTY_NOTICES.md, .gitignore
```

## Reproducibility

* **Data.** Not included. Download the official AudioSet and AudioSet Strong files from their web site and `vggsound.csv` from the VGGSound project; the test-suite needs none of them.
* **Offline tests.** `pytest` uses only `tests/data` and a fake `yt_dlp`: it never touches the network.
* **Full-scale check.** `AUDIOSET_TOOLS_DATA=/path/to/datasets pytest -m smoke -s` (folder with `AudioSet_meta/`, `AudioSet_Strong_meta/`, `VGGsound_meta/`) reproduces the sizes above and, with the SALT assets, the SALT coverage (99.98 % of the AudioSet eval clips, 100 % of the Strong eval segments, 99.65 % of the VGGSound clips). The 2,041,789-clip unbalanced split parses in about 10 s and needs about 2.5 GB of memory; the whole smoke test (all six files kept in memory) peaks at about 3.8 GB ([`docs/verify_dataset_statistics.md`](docs/verify_dataset_statistics.md)).
* **Seeds and identity.** `rebalance(random_state=s)` gives the same selection and order on any machine; `record_id` is a content hash independent of the Python hash seed.
* **What cannot be re-run offline.** Anything that talks to YouTube. The downloader's network behaviour (yt-dlp, cookies, bans, rate limits) was **not** exercised in the validation; see [`docs/downloader.md`](docs/downloader.md). The `fetch-salt` download is tested against a local server; one real download (2026-10-27) fetched both files with matching checksums.

## Documentation

[`docs/usage.md`](docs/usage.md) · [`docs/api.md`](docs/api.md) · [`docs/downloader.md`](docs/downloader.md) · [`docs/salt.md`](docs/salt.md) · [`docs/rebalancing.md`](docs/rebalancing.md) · [`docs/MIGRATION_FROM_V1.md`](docs/MIGRATION_FROM_V1.md) · [`docs/verify_dataset_statistics.md`](docs/verify_dataset_statistics.md) · [`docs/validation.md`](docs/validation.md) · [`CHANGELOG.md`](CHANGELOG.md)

## Current boundaries

* **Evidence.** Normalization does not improve the quality of the annotations: AudioSet labels remain clip-level and weak, temporally precise information exists only where the source provides it, and a SALT correspondence is an interpretation of a label, not a guarantee of identity.
* **Rebalancing** bounds each label's contribution to the smallest bucket; for multi-label clips the final per-label counts are not exactly equal (documented, with examples). Strong rebalancing works on segments, not on event rows.
* **YouTube** is not an archival source: availability, authentication and encodings drift. The downloader reports what happened; it cannot make retrieval reliable. Respect the YouTube terms of service and the rights of the video owners.
* **SALT assets** are not redistributed because the upstream repository declares no licence; they must be fetched (network) or supplied by the user. The optional `salt` extra requires Python < 3.12 and numpy < 2 (py-salt requirement).
* **v1 stereo-resampling defect.** The retrieval code of v1 resampled multichannel clips along the channel axis before pull request #1; **the waveforms of about 95 % of the released AudioSet-EV v1 clips are affected** (verified 2026-10-27 against the independently produced clips of AudioSet-EV v2; [`docs/MIGRATION_FROM_V1.md`](docs/MIGRATION_FROM_V1.md#5-the-stereo-resampling-defect-of-v1)). v2 uses `axis=0` and tests it.
* **Memory.** Records are Python objects: the unbalanced AudioSet split (2,041,789 clips) needs about 2.5 GB of memory.
* `AudioRecord` objects are immutable but not hashable (they hold dictionaries): use `record_id` as a key.

## Funding

Research activities were mainly funded under the Ministerial Decree (DM) 118/2023, Mission 4, Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) – “PNRR Research” – CUP: E11I23000100001.

Research activities and publications were also partially supported by the European Union under the Italian National Recovery and Resilience Plan (PNRR) of NextGenerationEU, within the partnership on “Telecommunications of the Future” (PE00000001 – RESTART), and by the European Union – NextGenerationEU under the Italian Ministry of University and Research (MUR), National Innovation Ecosystem grant ECS00000041 – VITALITY, CUP: E13C22001060006. These two project lines are the ones stated in the funding section of the AudioSet-Tools article (see below).

<p>
  <img src="assets/italiadomani_logo.png" alt="Italia domani" height="70">&nbsp;&nbsp;
  <img src="assets/next_generation_eu_logo.png" alt="Funded by the European Union - NextGenerationEU" height="70">&nbsp;&nbsp;
  <img src="assets/mur_blue_logo.jpg" alt="Ministero dell'Universita e della Ricerca" height="70">
</p>

## Related publications and how to cite

If you use this software, please cite the article (preferred citation, version 1 of the framework) and, for the version 2 architecture, the EUSIPCO 2026 poster; for the data, cite the AudioSet-EV records. `CITATION.cff` carries the same references. A DOI for the v2.0.0 software release will be added when it is minted.

```bibtex
@article{audioset_tools,
    author = {Giacomelli, Stefano and Giordano, Marco and Rinaldi, Claudia and Graziosi, Fabio},
    title = {{AudioSet-Tools: A Python Framework for Taxonomy-Aware AudioSet Curation and Reproducible Audio Research}},
    journal = {EURASIP Journal on Audio, Speech, and Music Processing},
    volume = {2026},
    number = {1},
    eid = {2},
    year = {2026},
    publisher = {Springer Nature},
    doi = {10.1186/s13636-025-00436-z}
}

@misc{audioset_tools_eusipco_poster,
  author = {Giacomelli, Stefano and Giordano, Marco and Rinaldi, Claudia and Graziosi, Fabio},
  title = {{AudioSet-Tools: A Python Framework for Taxonomy-Aware AudioSet Curation and Reproducible Audio Research}},
  year = {2026},
  note = {Poster presentation, European Signal Processing Conference (EUSIPCO 2026), Session ASMSP-P8.1, Bruges, Belgium, 3 September 2026},
  url = {https://cmsworkshops.com/eusipco2026/view_paper.php?SessionID=1012&PaperNum=2208}
}
```

Related datasets (the data records on Zenodo are licensed CC BY-SA 4.0):

```bibtex
@misc{audioset_ev_dataset,
    author = {Giacomelli, Stefano and Rinaldi, Claudia},
    title = {{AudioSet-EV: An AudioSet-Derived Distribution of Emergency Vehicle Siren Sounds}},
    publisher = {Zenodo},
    year = {2025},
    doi = {10.5281/zenodo.14882314},
    url = {https://zenodo.org/doi/10.5281/zenodo.14882314},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
}

@misc{audioset_ev_v2_dataset,
    author = {Giacomelli, Stefano and Rinaldi, Claudia},
    title = {{AudioSet-EV v2: A Refined AudioSet-Derived Distribution of Emergency Vehicle Siren Sounds}},
    month = feb,
    year = {2026},
    publisher = {Zenodo},
    version = {0.2.0},
    doi = {10.5281/zenodo.18668076},
    url = {https://zenodo.org/records/18668076},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```

## License

The code and the documentation of this repository are released under the [GNU General Public License, version 3 or later (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl-3.0.html); the licence text is in [`LICENSE`](LICENSE) and the notice in [`NOTICE`](NOTICE). Third-party material (AudioSet, AudioSet Strong and VGGSound excerpts, SALT, logos) keeps its own terms: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The AudioSet-EV data are not part of this repository: their Zenodo records state the licence CC BY-SA 4.0, and the official AudioSet metadata files are released by Google under CC BY 4.0 (the ontology file under CC BY-SA 4.0).

## Acknowledgments

The `axis=0` correction of the resampling call was contributed by [@Davvv24](https://github.com/Davvv24) (pull request #1 of v1). AudioSet is a project of Google (Gemmeke et al., ICASSP 2017; Hershey et al., ICASSP 2021 for the strong labels); SALT is by Stamatiadis, Olvera and Essid (DCASE 2024); VGGSound by Chen et al. (ICASSP 2020).

## Contact

Stefano Giacomelli — University of L'Aquila, Department of Information Engineering, Computer Science and Mathematics (DISIM), Italy — <stefano.giacomelli@graduate.univaq.it> — [GitHub @StefanoGiacomelli](https://github.com/StefanoGiacomelli) — ORCID [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748).
