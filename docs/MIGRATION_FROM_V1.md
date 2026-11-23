# Migrating from AudioSet-Tools v1

AudioSet-Tools v1 is the framework of the published article (*EURASIP J. Audio, Speech, and Music Processing* 2026, doi
[10.1186/s13636-025-00436-z](https://doi.org/10.1186/s13636-025-00436-z)) and the code that built **AudioSet-EV v1**. v2 is a re-design,
not a patch: one normalized record model instead of parallel Standard/Strong function families, label spaces
(`id`/`original`/`salt`/`auto`), a typed downloader with JSON reports, and tests that run offline.

**Where v1 lives.** Everything of v1 (the five modules, the raw metadata snapshots, the three entry scripts, `AudioSet_EV_data/`,
`AudioSet_EV_strong/`, `EV-benchmark/`, the figure notebook, its documentation) stays available under the **`v1.0.0` tag** of
`github.com/StefanoGiacomelli/audioset-tools` and in the Zenodo records of the data:
AudioSet-EV v1 [10.5281/zenodo.14882314](https://doi.org/10.5281/zenodo.14882314), AudioSet-EV v2
[10.5281/zenodo.18668076](https://doi.org/10.5281/zenodo.18668076). **Use the tag or Zenodo for the published EV benchmark files** (mind the stereo-resampling defect of the v1 waveforms, section 5); the
v1 data, loaders and notebooks are *not* part of this repository.

## 1. Function and script mapping

### Metadata functions (`filters.py`, `utils.py`; the `*_strong.py` versions map to the same v2 members)

| v1 | v2 | Notes |
|---|---|---|
| `select_by_label(labels_file, data_file, target_labels, out_filename)` | `AudioDataset.filter_labels(labels, label_space="original" or "id", match="any")` | the display-name to MID step is replaced by loading with `labels_file=`; v1 silently dropped labels missing from the label file, v2 simply matches nothing |
| `blacklist_by_label(...)` | `filter_labels(..., match="none")` or `blacklist_labels(...)` | |
| `select_by_samp_idx(data_file, start_idx, end_idx, out)` | `select_range(start, end)` | **index convention differs**: v1 counted the header as row 0 (first data row = 1), v2 is zero-based over records: v1 `(2, 5)` = v2 `(1, 4)` |
| `reselect_by_label(...)`, `reblacklist_by_label(...)` (processed CSV with MID lists) | the same `filter_labels` on the curated CSV loaded with `load_dataset` | curated v1 CSVs, including the ones with the `downloaded` column, are read directly (see section 4) |
| `rebalancing_filter(...)` | `AudioDataset.rebalance(focus_labels, label_space, random_state)` | deterministic; see section 3 and [`rebalancing.md`](rebalancing.md). Strong: v1 balanced **event rows** (single label per row), v2 balances **segments** (records with several events) |
| `find_samps_by_samps(targets_file, data_file)` | `find_samples_by_samples(targets, dataset_or_path)` / `find_by_youtube_ids(ids)` | v1 returned `(row index, yt_id)` tuples (Strong: one hit per annotated **event**); v2 returns a dataset of **records** (one per clip, events inside); v2 splits Strong ids at the **last** `_` (v1 split at the first one and mis-parsed ids containing `_`) |
| `compute_stats(data_file, labels_file, save_json)` | `compute_stats(...)` / `AudioDataset.summary()` and `label_counts(label_space)` | label occurrences are identical (checked on the stored `EV_Positives.csv`: 8,409 clips, 111 labels); v1's statistics conditioned on the `downloaded` flag are not reproduced: the flag is only kept in `raw["downloaded"]` |
| `merge_sets(files, output_file)` | `merge_files(paths)` / `merge_datasets(*datasets)` then `to_csv`/`to_json` | v1 deduplicated on the whole CSV row, v2 on the record identity (`record_id` fields) |
| `utils_strong.generate_event_tracks(...)` | `generate_event_tracks(dataset_or_path, bin_size=...)` | same floor rule `int(10 / bin_size)` (33 bins for 0.3 s) |
| `utils_strong.plot_events_pianoroll(...)` | `plot_events_pianoroll(...)` | needs the `plot` extra |
| (SALT only in the notebook `salt_paper_figs.ipynb`) | `SaltResolver`, the `salt` label space, `include_descendants` | [`salt.md`](salt.md) |

### Retrieval (`downloaders.py`)

| v1 `StandardDownloader` | v2 `YouTubeDatasetDownloader` |
|---|---|
| `__init__(data_file, labels_file, target_sr, channels_proc, normalize, cookies_file, verbose, sleep_interval_requests, browser_command)`; context manager; reads the metadata CSV itself | `__init__(output_dir, target_sr, channels, normalize, cookies_file, retries, sleep_seconds, verbose, dry_run, *, ...)`; jobs come from `dataset.to_download_jobs()`; `download_dataset` / `download_jobs` / `download_job` |
| `channels_proc='stereo' / 'mono_split' / 'mono_red'` | `channels="stereo" / "mono_split" / "mono"` |
| output `AudioSet_<csv stem>_downloads/<yt_id>_Original.wav`, `_Reduced.wav` (mono reduction), `_Left.wav`, `_Right.wav` | `<output_dir>/<source_id>.wav` (mono and stereo), `<source_id>_Left.wav` / `_Right.wav`; the stem is the sanitized `source_id` (the benchmark loaders of v1 list `*.wav` files, so both layouts are readable) |
| in-place `downloaded` column in the metadata CSV | no mutation of metadata; state = output files (`on_exists="skip"`) and the JSON report |
| three text reports (`Missing_samples_report.txt`, `Success_samples_report.txt`, `Success_labels_report.txt`) | one JSON report: status, category, message, retries, UTC timestamp, paths, labels (`write_download_report`, `read_download_report`) |
| resume (flag **and** `*<yt_id>*.wav`) | `on_exists="skip"` |
| random 5-20 s pause | `random_pause=(5, 20)` (opt-in) |
| shadow-ban halt (three messages) | `halt_on_shadow_ban=True` raises `ShadowBanError` (opt-in) |
| `refresh_cookies()` (Firefox, `pgrep`) | `cookie_refresh_hook`, `cookies_from_browser` (yt-dlp), `cookies_file`; **not** reproduced literally |
| `sleep_requests: 1.25` (not a yt-dlp option: never applied) | `ydl_options={"sleep_interval_requests": 1.25}` |
| `resampy.resample(...)` with `axis=0` (after PR #1) | `resampy.resample(..., axis=0)` kept, with tests |

Methods of `StandardDownloader` one by one: `load_data` / `load_labels` -> `load_dataset(path, labels_file=...)` (then `to_download_jobs()`); `create_output_folder` -> the `output_dir` argument (created if missing, no
`AudioSet_<stem>_downloads` folder is invented); `download_and_process` -> `download_jobs` / `download_dataset`; `download_and_process_audio` and `process_audio` -> `download_job` (the DSP chain is private);
`refresh_cookies` -> `cookie_refresh_hook`; `generate_reports` -> `write_download_report` (JSON); `__enter__`/`__exit__` (reports written on exit) -> not needed: `download_jobs(..., report_path=...)` writes the report even if the run is halted.

Details and the exact semantics: [`downloader.md`](downloader.md).

### Entry scripts

| v1 script | In v2 |
|---|---|
| `main_as-ev_standard_processing.py` (AudioSet-EV metadata) | the recipe below (positives) and `rebalance(random_state=...)` for the negatives. The *published* negatives are one hash-order-dependent draw of the legacy rebalancing and cannot be regenerated bit for bit by any version; a deterministic draw with the same procedure is `rebalance` |
| `main_as-ev_standard_download.py` (retrieval) | `YouTubeDatasetDownloader(..., target_sr=32000, channels="mono", on_exists="skip", random_pause=(5, 20), cookies_file="cookies.txt")` |
| `main_as_strong_processing.py` (Strong subset, piano roll, tracks) | `load_dataset(strong_tsv, labels_file=...)`, `filter_labels`, `generate_event_tracks(bin_size=0.3)`, `plot_events_pianoroll` |

Recipe (the AudioSet-EV v1 positives with v2; it reproduces the stored `EV_Positives.csv`, see [`validation.md`](validation.md)):

```python
from audioset_tools import load_dataset, merge_datasets

labels = "class_labels_indices.csv"
splits = [load_dataset(f"{name}_segments.csv", labels_file=labels) for name in ("balanced_train", "unbalanced_train", "eval")]
positives = (merge_datasets(*splits)
             .filter_labels(["Emergency vehicle", "Police car (siren)", "Ambulance (siren)",
                             "Fire engine, fire truck (siren)"], label_space="original")
             .filter_labels(["Civil defense siren"], label_space="original", match="none"))
positives.to_csv("EV_Positives.normalized.csv")      # 8,409 clips on the official CSVs of the v1 case study (headers: created Sun Mar 5 2017)
```

## 2. What v2 deliberately does not retain

* The in-place mutation of the metadata CSV (`downloaded` column), the three text reports and the `_Original`/`_Reduced` naming.
* The Firefox/`pgrep` cookie refresh (replaced by a hook), the 5-20 s pause as a *default* (now opt-in), the (ineffective) 1.25 s request delay.
* Separate Standard/Strong function families and `*_strong.py` modules; the "processed CSV" distinction (`re*` functions).
* Event-row rebalancing of Strong TSVs (v2 balances segments).
* The EV case study and its data (`AudioSet_EV_data`, `AudioSet_EV_strong`), the benchmark loaders (`EV-benchmark/`: `AudioSetEV_*`, `ESC50_*`, `sireNNet_*`, `LSSiren_*`, `UrbanSound8K_*`, `FSD50K_*`, `data_demo.py`,
  `FSD50K/parser.py`, `ESC-50/create_folds.py`), the statistics script (`analysis/ev_statistics.py`), the figure notebook (`salt_paper_figs.ipynb`) and the raw metadata
  snapshots (`original_csv_01-11-2024`, `strong_tsv_12-03-2025`): they remain under the `v1.0.0` tag / Zenodo.
* Pre-vendored SALT assets (see [`salt.md`](salt.md)).

## 3. Rebalancing: determinism

v1's published rebalancing visited labels in the insertion order of a dictionary built from `set` intersections and returned `list(set(...))`
(both depend on `PYTHONHASHSEED` and on the file-listing order of the calling script). v2 sorts the labels and uses a private
`random.Random(random_state)`: the same seed gives the same selection **and** order on any machine. The `v1.0.0` snapshot added
`deterministic=True, seed=...`, which is the same procedure (identical selection and order on the full eval and balanced splits, 18 combinations;
[`validation.md`](validation.md)). A draw made with the legacy v1 mode has no v2 equivalent.

## 4. Reading v1 files with v2

* Raw official AudioSet CSVs (with `#` comment lines) and the hand-edited-header CSVs of v1: read directly.
* **Curated v1 CSVs with a `downloaded` column** (the published `EV_Positives_downloaded.csv`, `EV_Negatives_downloaded.csv`): read correctly since v2.0.0; the flag is kept in
  `record.raw["downloaded"]` (`"True"`/`"False"`). The first v2 prototype joined that column into the label cell (877 and 2,008 spurious "labels" on the two files
  instead of 111 and 326); this is regression-tested.
* v1 Strong TSVs: read directly; `AudioDataset` groups the event rows by segment.

## 5. The stereo-resampling defect of v1

**Status: the code path is confirmed and the effect on the released AudioSet-EV v1 waveforms is VERIFIED: about 95 % of the clips are affected (check of 2026-10-27).**

* **What the code did.** Until commit `21729fb` (pull request #1 by `@Davvv24`, 2026-06-18, one line), `StandardDownloader.process_audio` called
  `resampy.resample(data, sr, target_sr)` without `axis=0`. `soundfile` returns stereo clips as `(frames, channels)` and resampy resamples the **last** axis by default, i.e. the channel axis. With a
  source rate between 32 kHz and 64 kHz (YouTube streams: 44.1 kHz or 48 kHz) the result has **one** channel at the **original** rate and no error is raised; the
  following trimming then used indices computed at the target rate, and the file was stamped with the target rate: the audio is a mislocated, slowed (x1.5 at 48 kHz) left-weighted mono excerpt.
  The file length is still exactly the requested duration, so the header and the duration do not reveal it. Mono sources are not affected. Executed demonstration with synthetic
  signals (resampy 0.4.3): [`docs/known_issue_stereo_resampling.md` in the `v1.0.0` tag](https://github.com/StefanoGiacomelli/audioset-tools/blob/v1.0.0/docs/known_issue_stereo_resampling.md).
* **What the check found.** The archive files were written on 4-6 January 2025 by the pre-fix code (the downloader of `init_commit`, identical at the Zenodo release); the fix of pull request #1 (merged in `2c37407`) postdates them by 17 months. Of 894 sampled clips that also exist in AudioSet-EV v2 (independently produced), 338 are proven affected (source-rate signal stamped 32 kHz, slowed by 1.378 / 1.5, wrong window), 165 are correct (almost all from mono sources) and 391 are undetermined; estimate: about 95 % (roughly 13,300 of 14,026) affected. Not known: the status of each individual clip and the channel count of each source (never logged). Details, per-clip table and reproduction script: [`docs/stereo_resampling_check_results.md` in the `v1.0.0` tag](https://github.com/StefanoGiacomelli/audioset-tools/blob/v1.0.0/docs/stereo_resampling_check_results.md).
* **What v2 does.** `downloader.py` resamples with `axis=0`. Regression tests in `tests/test_downloader.py` use synthetic stereo signals at 48, 44.1 and 32 kHz (left 440 Hz, right 1,000 Hz, a marker at a known time): they check the output rate, the number
  of frames, both channels' frequencies and the marker position for the `stereo`, `mono` and `mono_split` policies, and they **fail when `axis=0` is removed** (validation evidence in [`validation.md`](validation.md)).
  A further test documents the third-party behaviour (the default axis never returns the right shape).
