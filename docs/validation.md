# Validation record of version 2.0.0

What was checked before this tree was staged as the successor of AudioSet-Tools v1, how to repeat it, and what was **not** checked. Date of the run: 2026-10-26, on an Apple-silicon
laptop. The code was extracted from the prototype `taxonomy/audioset_tools` of the `gpat_framework` repository; "in-tree v2" below means an unmodified copy of that prototype.

## 1. Summary

| Check | Result |
|---|---|
| Test-suite in a **clean venv, Python 3.10.11**, `pip install -e ".[dev,download,plot]"` (numpy 2.2.6, matplotlib 3.10.9, resampy 0.4.3, soundfile 0.14.0, yt-dlp 2026.10.21) | `ruff check .` clean; **132 passed, 11 skipped** (10 full-scale smoke tests need `AUDIOSET_TOOLS_DATA`, 1 needs `torch`) in about 15 s |
| The same tests against the **built wheel** installed into a second clean venv, **Python 3.13.3** (numpy 2.5.4, matplotlib 3.11.2); `import audioset_tools` from a neutral directory resolves to `site-packages` | 132 passed, 11 skipped |
| Clean editable installs on **Python 3.11 and 3.12** | 132 passed, 11 skipped on each |
| The **sdist** unpacked, installed into a clean Python 3.12 venv, tests run inside the unpacked tree (checks `MANIFEST.in`) | 132 passed, 11 skipped |
| Wheel | 18 files: the nine modules, `cli.py`, `__main__.py`, `py.typed`, metadata and `licenses/LICENSE`; no data files (no SALT assets); `License-Expression: GPL-3.0-or-later` (the first build of 2026-10-26, before the licence was added, had 17 files and no licence field; the licence-dependent checks were repeated on 2026-11-03, section 7) |
| **Full-scale smoke test** on the official metadata (10 tests) | 10 passed in 21.8 s; sizes and SALT coverage: [`verify_dataset_statistics.md`](verify_dataset_statistics.md) |
| **Mutation test**: `axis=0` removed from the `resampy.resample` call (the v1 defect) | 4 of the stereo regression tests fail (48 kHz and 44.1 kHz stereo, `mono`, `mono_split`); the 32 kHz case (no resampling) still passes |
| **Side-by-side with the in-tree v2**, same inputs and SALT files, separate processes, complete official metadata | **65 of 65 compared results identical**: parse of the six sources with a digest of every field of every record (AudioSet 20,371 / 22,160 / 2,041,789, Strong 16,996 / 103,463 with 139,538 / 934,821 events, VGGSound 199,467), 17 filter cases (`id`/`original`/`salt`/`auto`, `any`/`all`/`none`, descendants on and off, functional wrappers), ranges, lookups, grouping, `rebalance` (seeds 0, 7, 42; four configurations plus all labels; method and wrapper), merge/deduplicate/`merge_files`, statistics, event tracks (3 bin sizes x 3 label spaces), JSON export (without the new `record_id` key) and the first nine CSV columns. Parse times equal within noise. Differences by design: the new CSV columns and the fixes of section 3 |
| **v1 cross-checks** (modules of the v1 cleaned snapshot, standard library only) | `rebalancing_filter(deterministic=True, seed=s)` equals `rebalance(random_state=s)` in selection and order: 18 of 18 combinations (eval and balanced splits; focus sets of 5 labels, 6 labels, all 527 labels; seeds 0, 7, 42). `compute_stats` label occurrences equal `label_counts("original")` on the stored `EV_Positives.csv` (8,409 clips, 111 labels) and `EV_Negatives.csv` (8,502, 312). The AudioSet-EV v1 positives rebuilt with v2 operations (recipe in [`MIGRATION_FROM_V1.md`](MIGRATION_FROM_V1.md)) are the stored `EV_Positives.csv`: 8,409 clips, identical as a set and as an ordered list |
| **Example notebook** | executed in a real Jupyter kernel (nbclient 0.11 / ipykernel 7.3, Python 3.10.11, **numpy 1.26.4**): no error, and the standard output of all 23 code cells and the 4 figures equal those of the executed copy shipped in `docs/` |
| SALT | upstream licence check (none declared), pinned SHA-256, upstream and cleaned TSV resolve identically, `descendants()` fix: section 4 |

## 2. How to repeat

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,download,plot]"
ruff check . && pytest                                   # 132 passed, 11 skipped
python -m build                                          # sdist + wheel in dist/
AUDIOSET_TOOLS_DATA=/path/to/datasets pytest -m smoke -s # full-scale sizes; add AUDIOSET_TOOLS_SALT_DIR for the SALT coverage
```

The offline suite is organized by topic: `test_parsing` (formats, parsers), `test_roundtrip` (JSON/CSV), `test_identity` (`record_id`), `test_operations` (filters,
ranges, grouping, merge, rebalance), `test_events`, `test_salt` (resolver, assets, fetch against a local server, `py-salt` stub), `test_downloader` (fake `yt_dlp`),
`test_examples` (the notebook), `test_packaging` (metadata consistency), `test_full_scale` (optional).

Worked rebalance examples of [`rebalancing.md`](rebalancing.md) (synthetic: buckets 15 / 10 / 29, `m` = 10, 30 selected, per-label 11 / 10 / 16; AudioSet eval, five labels: buckets 5,324 / 5,695 / 212 / 230 / 189,
`m` = 189, 945 selected, per-label 363 / 269 / 189 / 190 / 189) were executed with the package.

## 3. Defects of the prototype found and fixed (each with a regression test)

| ID | Defect | Test |
|---|---|---|
| F-1 | `to_csv` output could not be read back (header not recognized, JSON cells and floats not decoded) and dropped `events`, `raw`, `source_path` | `test_roundtrip.py` |
| F-2 | `resampy` imported lazily but not declared; first real download failed late | `test_downloader.py::test_missing_*` |
| F-3 | deprecated `matplotlib.cm.get_cmap` | `test_events.py::test_piano_roll_uses_the_current_matplotlib_colormap_api` |
| F-4 | three copies of `_ensure_dataset` | `test_events.py::test_ensure_dataset_has_a_single_definition` |
| F-5 | private `explorer._std_col` | `test_salt.py::test_from_py_salt_*` |
| F-6 | repeated runs silently duplicated files | `test_downloader.py` (`on_exists`) |
| F-7 | multi-label `rebalance` semantics were implicit | documented; `test_operations.py::test_rebalance_*` |
| F-8 | curated v1 CSV files **with a `downloaded` column** (the published `EV_*_downloaded.csv`) produced garbage labels: the list string and `True`/`False` became "labels" (877 and 2,008 distinct "labels" instead of 111 and 326). Reproduced on the unmodified prototype with a two-row fixture and on the four stored v1 files, fixed | `test_parsing.py::test_curated_v1_files_*` |
| F-9 | `SaltResolver.descendants()` **never returned** for the ten top-level SALT roots (`music`, `animal`, `human_sounds`, ...): the real roots file nests each root inside itself and the breadth-first search had no visited set; it also listed repeated nodes many times (497 entries for `vehicle`, 81 distinct). The fixed version equals the old list de-duplicated for the other 844 of the 854 labels of the real file | `test_salt.py::test_descendants_*` |

## 4. SALT checks

* **Licence.** On 2026-10-26 the GitHub API reports `license: null` for `tpt-adasp/salt`, the `/license` endpoint answers 404, and the tree of `main` (commit `e89d13b65f59bf621539faaf4dd3ad7ff81232c6`) holds no
  `LICENSE`, `COPYING` or `NOTICE` file; no README or `setup.py` states a licence. The two asset files are therefore not vendored ([`salt.md`](salt.md)).
* **Checksums.** Computed from the upstream git objects of that commit (values in `salt.md`). Fetch code tested against a local HTTP server; one real download on 2026-10-27 (Python 3.13.3, macOS): both checksums matched.
* **Upstream versus the author's cleaned TSV** (fork commit `2358c17`): 9,928 rows read upstream versus 8,871 in the cleaned file (1,057 blank rows removed); the 8,869 usable rows are identical; the resolver
  mapping (1,919 `(dataset, label)` keys) and the hierarchy index are equal; the roots file is byte-identical.
* **Fork.** `StefanoGiacomelli/salt_exp`, branch `gpat/py-salt-local-dev`, commit `2edbadb9fae33e46751a40de1f76b190ecb2b5e4`, is pushed (GitHub API). The upstream `py-salt/setup.py` is not installable
  (`find_packages(include=['pysalt*'])`, a missing comma between two requirements); the fork fixes the packaging, hence the pinned `salt` extra.

## 5. What was **not** validated

* Anything that talks to **YouTube**: real yt-dlp runs, current yt-dlp behaviour, cookies (file, browser, refresh hook), the shadow-ban message strings of v1 against the live service, rate limits. The downloader tests use a fake `yt_dlp`; only
  the existence of the yt-dlp option names used was checked against the installed yt-dlp.
* `SaltResolver.from_py_salt()` with a real `py-salt` (stub explorer only). The `fetch-salt` download was validated against a local server; a single real download on 2026-10-27 (Python 3.13.3, macOS) matched both checksums, and the same command fails with `CERTIFICATE_VERIFY_FAILED` on a python.org interpreter that has no certificate bundle installed (see [`salt.md`](salt.md)).
* `out_type="pt"` (torch is not installed in the validation environments; the test is skipped).
* The **GitHub Actions workflow** (a file; it cannot run here); **static type checking** (`py.typed` ships, no checker was run); Windows; Python 3.14.
* The effect of the v1 stereo-resampling defect on the released AudioSet-EV v1 waveforms was checked separately on 2026-10-27 against the clips of AudioSet-EV v2 (result: affected; see [`MIGRATION_FROM_V1.md`](MIGRATION_FROM_V1.md#5-the-stereo-resampling-defect-of-v1)); it is not part of this validation.
* Time and memory figures are single measurements on one machine.

## 6. Re-verification of the release tree (2026-10-27 and 2026-10-28)

Before the `v2.0.0` tag the checks were repeated on the final tree (licence, package metadata and documentation finalised; no source file changed):

| Check | Result |
|---|---|
| `ruff check .` | clean |
| `pytest`, editable installs on Python 3.10.11, 3.11.2, 3.12.8 and 3.13.3 | 132 passed, 11 skipped on each |
| Built wheel (18 files, `licenses/LICENSE`, `License-Expression: GPL-3.0-or-later`, no data files; rebuilt on 2026-11-03 when the licence was fixed, section 7) installed into a clean Python 3.13.3 environment, tests run against it | 132 passed, 11 skipped |
| sdist (77 entries, with `LICENSE` and `NOTICE`) unpacked and installed into a clean Python 3.12.8 environment, tests run in the unpacked tree | 132 passed, 11 skipped |
| Full-scale smoke test (`AUDIOSET_TOOLS_DATA` = the official metadata, SALT files fetched by `audioset-tools fetch-salt`) | 10 passed in 21.2 s; same sizes and SALT coverage as in section 1 |
| `audioset-tools fetch-salt`, real download (Python 3.13.3, macOS) | both files written, SHA-256 as pinned in [`salt.md`](salt.md) |
| README quick start (Python snippet) | output equal to the documented one |

On a python.org Python 3.10 without a certificate bundle the same `fetch-salt` command stops with `CERTIFICATE_VERIFY_FAILED` and writes nothing ([`salt.md`](salt.md)). The GitHub Actions workflow was not run.

## 7. Licence change (2026-11-03)

On 2026-11-03 the licence of the code and the documentation was fixed as the **GNU General Public License, version 3** (SPDX `GPL-3.0-or-later`). `LICENSE` is the official text
fetched from <https://www.gnu.org/licenses/gpl-3.0.txt> on 2026-11-03 (35,149 bytes, SHA-256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`). The licence is stated in `LICENSE`, `CITATION.cff`,
`pyproject.toml`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `README.md`, `CHANGELOG.md` and asserted in `tests/test_packaging.py`; no source file of the package changed. The data keep deferring to the licence of their
Zenodo records (CC BY-SA 4.0). The checks that depend on the licence were repeated on a copy of the tree:

| Check | Result |
|---|---|
| `ruff check .` and `pytest`, editable install on Python 3.10.11 | ruff clean; 132 passed, 11 skipped; installed metadata `License-Expression: GPL-3.0-or-later`, `License-File: LICENSE`, no `License ::` classifier |
| Rebuilt wheel (18 files, `licenses/LICENSE`, `License-Expression: GPL-3.0-or-later`) installed into a clean Python 3.13.3 environment, tests run against it | 132 passed, 11 skipped |
| Rebuilt sdist (77 entries) unpacked and installed into a clean Python 3.12.8 environment, tests run in the unpacked tree | 132 passed, 11 skipped |
