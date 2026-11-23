# API summary

All names below are exported by `audioset_tools` (`audioset_tools.__all__`; the package is typed, `py.typed`). Modules:
`models`, `parsers`, `dataset`, `filters`, `utils`, `events`, `salt`, `downloader`, `cli`.

## Data model (`models`)

| Name | Description |
|---|---|
| `AudioRecord` | frozen dataclass: `dataset`, `source_id`, `youtube_id`, `start_seconds`, `end_seconds`, `original_label_ids`, `original_labels`, `salt_labels`, `split`, `events`, `raw`, `source_path`; `duration_seconds`, `labels(label_space)`, `record_id`, `to_dict()` |
| `TemporalEvent` | `start_seconds`, `end_seconds`, `original_label_id`, `original_label`, `salt_labels`, `raw`; `to_dict()` |
| `DownloadJob` | request derived from a record (`DownloadJob.from_record`): ids, interval, `output_stem`, labels, `metadata` |
| `DownloadReportEntry` | result of one job: paths, `status`, `success`, `error_category`, `error_message`, `retry_count`, `processing_timestamp`, labels |
| `AUDIOSET`, `AUDIOSET_STRONG`, `VGGSOUND` | dataset names (`"AudioSet"`, `"AudioSetStrong"`, `"VggSound"`); `models.NORMALIZED` = `"Normalized"` |
| `is_valid_youtube_id(value)` | 11 characters of `[A-Za-z0-9_-]` |

## Parsing (`parsers`)

| Name | Description |
|---|---|
| `detect_format(path)` | `AudioSet`, `AudioSetStrong`, `VggSound` or `Normalized`; `FileNotFoundError`/`ValueError` otherwise |
| `load_label_map(labels_file)` | MID -> display name from `class_labels_indices.csv`, `mid_to_display_name.tsv` or `KnowledgeGraphId,Name` files |
| `parse_metadata(path, labels_file=None, salt_resolver=None, dataset=None)` | list of `AudioRecord` |

## Dataset (`dataset`, plus the functional wrappers of `filters` and `utils`)

`AudioDataset(records, source_format=None, salt_resolver=None)` (immutable; iterable, indexable, `len`):

| Member | Description |
|---|---|
| `AudioDataset.from_file(path, labels_file=None, salt_resolver=None, dataset=None)`; `load_dataset(...)` | parse a file |
| `filter_labels(labels, label_space="auto", match="any", include_descendants=False, case_sensitive=False)` | query spaces `id`/`original`/`salt`/`auto`; `any`/`all`/`none` |
| `select_range(start, end)` | zero-based, half-open |
| `group_by(field)`, `label_counts(label_space="original")`, `summary()` | grouped inspection and statistics |
| `find_by_youtube_ids(ids)` | records of those videos |
| `rebalance(focus_labels=None, label_space="original", random_state=None)` | see [`rebalancing.md`](rebalancing.md) |
| `deduplicate()`, `merge(*others, deduplicate=True)`; `merge_datasets(*datasets, deduplicate=True)` | identity = the `record_id` fields |
| `to_json(path)`, `to_csv(path)` | lossless normalized export (readable back) |
| `to_download_jobs()` | tuple of `DownloadJob` |

Functional wrappers (accept a dataset **or a path**): `filter_labels(dataset_or_path, labels, labels_file=None, salt_resolver=None, label_space="auto", match="any", include_descendants=False)`,
`blacklist_labels(...)`, `select_range(dataset_or_path, start, end, **load_kwargs)`, `rebalance(dataset_or_path, focus_labels=None, label_space="original", random_state=None, **load_kwargs)`,
`compute_stats(dataset_or_path, ...)`, `find_samples_by_youtube_ids(dataset_or_path, ids, ...)`,
`find_samples_by_samples(targets, dataset_or_path, ...)` (YouTube ids of a dataset, path or iterable), `merge_files(paths, labels_file=None, salt_resolver=None, deduplicate=True)`.

## Events (`events`)

`generate_event_tracks(dataset_or_path, labels_file=None, salt_resolver=None, bin_size=0.1, label_space="original", out_type="np")`,
`iter_events(dataset_or_path, ..., label_space="original")` (yields `(record, event, label)`),
`plot_events_pianoroll(dataset_or_path, segment_ids=None, ..., label_space="original", save_plots=False, output_dir=None, verbose=False)` (returns matplotlib figures).

## SALT (`salt`)

`SaltResolver(mapping, roots=None)` with `from_mapping_file(mapping_file=None, roots_file=None)`, `from_py_salt(explorer=None)`, `resolve(dataset, labels)`,
`expand_salt_labels(labels, include_descendants=False)`, `descendants(label)`; `fetch_salt_assets(dest=None, *, force=False, ...)`, `default_salt_dir()`,
`SaltAssetError`; constants `SALT_UPSTREAM_REPOSITORY`, `SALT_UPSTREAM_COMMIT`, `SALT_ASSETS` (module `audioset_tools.salt`). See [`salt.md`](salt.md).

## Retrieval (`downloader`)

`YouTubeDatasetDownloader(output_dir, target_sr=44100, channels="stereo", normalize=False, cookies_file=None, retries=1, sleep_seconds=0.0, verbose=False, dry_run=False, *, on_exists="rename", random_pause=None, pause_seed=None, halt_on_shadow_ban=False, shadow_ban_messages=SHADOW_BAN_MESSAGES, max_consecutive_failures=None, cookies_from_browser=None, cookie_refresh_hook=None, ydl_options=None, check_prerequisites=True)`
with `download_dataset(dataset, limit=None, report_path=None)`, `download_jobs(jobs, limit=None, report_path=None)`, `download_job(job)`, `check_prerequisites()`;
`write_download_report(entries, path)`, `read_download_report(path)`;
exceptions `DownloaderSetupError`, `DownloadHaltedError` (base), `ShadowBanError`, `ConsecutiveFailuresError`; constants `SHADOW_BAN_MESSAGES`, `AUTH_REQUIRED_MESSAGES`.
See [`downloader.md`](downloader.md).

## Command line (`cli`)

`audioset-tools --version`; `audioset-tools fetch-salt [--dest DIR] [--force] [--timeout S]`.

## Stability notes

The shared helper that returns a dataset from "dataset or path" is private (`dataset._ensure_dataset`); `events.ensure_dataset` is kept as an
alias for the first v2 prototype. The `record_id` definition is part of the public contract (see [`usage.md`](usage.md#record_id)).
