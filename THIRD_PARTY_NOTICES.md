# Third-party notices

This file is documentation only: it records the origin and the terms, as far as they are known to the author, of the
third-party material that is referenced by or included in this repository. It is not legal advice; check the sources.
The code and the documentation of this repository are released under the GNU General Public License, version 3 or (at your option) any later version (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)); nothing here changes the terms of the third-party material.

## Included in the repository

| Material | Where | What is included | Terms and attribution |
|---|---|---|---|
| **AudioSet** segment lists and class labels (Google) | `tests/data/audioset/` | the three `#` comment lines and the first 30 rows of `eval_segments.csv`; the rows of `class_labels_indices.csv` that those 30 rows use | Distributed by Google with the AudioSet release (data: CC BY 4.0; ontology: CC BY-SA 4.0, as stated by Google and by the README of `github.com/audioset/ontology`; check the current terms at research.google.com/audioset). Attribution: J. F. Gemmeke et al., "Audio Set: An Ontology and Human-Labeled Dataset for Audio Events", ICASSP 2017, pp. 776-780, doi 10.1109/ICASSP.2017.7952261 |
| **AudioSet Strong** temporally-strong labels (Google) | `tests/data/audioset_strong/` | the header and the first 7 whole segments (43 event rows) of `audioset_eval_strong.tsv`; the rows of `mid_to_display_name.tsv` that they use | Released by Google together with AudioSet; terms: see the AudioSet web site. Attribution: S. Hershey et al., "The Benefit of Temporally-Strong Labels in Audio Event Classification", ICASSP 2021, pp. 366-370, doi 10.1109/ICASSP39728.2021.9414579 |
| **VGGSound** metadata | `tests/data/vggsound/` | the first 40 rows (YouTube id, start second, label, split) of `vggsound.csv` | Terms: see the VGGSound project page (not verified by the author of this repository). Attribution: H. Chen et al., "VGGSound: A Large-Scale Audio-Visual Dataset", ICASSP 2020, pp. 721-725, doi 10.1109/ICASSP40776.2020.9053174 |
| Synthetic SALT-like files | `tests/data/salt/` | hand-written, same layout as SALT, invented standard-event names | Written for this repository; not derived from the SALT files |
| Logos | `assets/` | DISIM / University of L'Aquila, "Italia domani", NextGenerationEU, MUR | Property of the respective institutions; used to acknowledge affiliation and funding |

No audio, no video and no YouTube content is included in this repository.

## Referenced but not included

| Material | Role | Notes |
|---|---|---|
| **SALT** (Standardized Audio event Label Taxonomy; `github.com/tpt-adasp/salt`) | optional `salt` label space (mapping TSV and hierarchy JSON) | The upstream repository declares **no licence** (GitHub API `license: null`; no LICENSE/COPYING/NOTICE file; checked on 2026-10-26 at commit `e89d13b65f59bf621539faaf4dd3ad7ff81232c6`), so the files are **not redistributed**; `audioset-tools fetch-salt` downloads them from upstream to the user's machine with a checksum check. Attribution: P. Stamatiadis, M. Olvera, S. Essid, "SALT: Standardized Audio Event Label Taxonomy", DCASE 2024 Workshop, arXiv:2409.11746. The optional `salt` extra installs the author's fork of the Python package (`github.com/StefanoGiacomelli/salt_exp`, commit `2edbadb9fae33e46751a40de1f76b190ecb2b5e4`), which inherits the licence situation of upstream |
| **FSD50K** | the SALT mapping covers its labels (dataset key `Freesound50k`) | not used or shipped here. E. Fonseca et al., IEEE/ACM TASLP 30, 829-852, 2022, doi 10.1109/TASLP.2021.3133208 |
| **YouTube** content | what the downloader retrieves on request | Subject to the YouTube Terms of Service and to the rights of the owners of the videos; the user is responsible for the use made of the downloader and of the retrieved audio |
| **Python dependencies** | `numpy`, `tqdm` (core); `yt-dlp`, `soundfile`, `resampy` (`download`); `matplotlib` (`plot`); `torch` (`torch`); `pytest`, `ruff`, `build`, `nbstripout` (`dev`) | installed from PyPI, each under its own licence (read from the installed distributions during the validation: yt-dlp Unlicense, soundfile BSD 3-Clause, resampy ISC, numpy BSD, tqdm MPL-2.0 and MIT, matplotlib PSF-based licence) |

## Contributors

The `axis=0` correction of the resampling call (pull request #1 of AudioSet-Tools v1, commit `21729fb`) was contributed by [@Davvv24](https://github.com/Davvv24).
