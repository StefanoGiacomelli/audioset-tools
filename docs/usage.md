# Usage guide

Everything below runs on the excerpts shipped in `tests/data/` (see `tests/data/README.md`) or on your own copies of the
official metadata files. Paths are relative to the repository root.

## 1. Loading metadata

```python
from audioset_tools import AudioDataset, SaltResolver, detect_format

detect_format("tests/data/audioset/eval_segments_excerpt.csv")       # 'AudioSet'

salt = SaltResolver.from_mapping_file("tests/data/salt/salt_event_mapping.tsv",
                                      "tests/data/salt/salt_event_roots.json")   # optional, see docs/salt.md
audioset = AudioDataset.from_file("tests/data/audioset/eval_segments_excerpt.csv",
                                  labels_file="tests/data/audioset/class_labels_indices_excerpt.csv",
                                  salt_resolver=salt)
print(audioset)            # AudioDataset(records=30, source_format='AudioSet', salt_resolver=yes)
```

`detect_format(path)` looks at the file and returns the format; `AudioDataset.from_file` / `load_dataset` / `parse_metadata`
use it unless you pass `dataset=`.

| Format (`detect_format`) | File | Record created |
|---|---|---|
| `AudioSet` | the official segment CSVs (`eval_segments.csv`, `balanced_train_segments.csv`, `unbalanced_train_segments.csv`; the three `#` comment lines are skipped), or curated CSVs with a header `yt_id,start_seconds,end_seconds,positive_labels` where the label cell is a MID list (`"/m/a,/m/b"` or the Python list repr written by AudioSet-Tools v1), with an optional trailing `downloaded` True/False column (kept in `raw["downloaded"]`, never a label) | one record per clip; `source_id = <ytid>_<start>_<end>` (integer times as integers, otherwise `.` written as `p`: `abc_0p5_10`) |
| `AudioSetStrong` | the official strong-label TSVs (`segment_id`, `start_time_seconds`, `end_time_seconds`, `label`) | one record per **segment** (rows grouped by `segment_id`), with one `TemporalEvent` per row; `source_id` = segment id; the clip start is read after the last `_` of the id (YouTube ids may contain `_`) and the clip is 10 s long |
| `VggSound` | `vggsound.csv` (`youtube_id,start_seconds,label,split`, no header) | one record per row; `end = start + 10`; `source_id = <ytid>_<start>`; `split` kept |
| `Normalized` | JSON written by `to_json` or CSV written by `to_csv` (detected by the header columns, whatever their order) | the stored records, unchanged |

Label names: `labels_file` maps MIDs to display names (`class_labels_indices.csv`, `mid_to_display_name.tsv`, or a
`KnowledgeGraphId,Name` CSV). Without it the MIDs are used as names. VGGSound labels are already names.

## 2. The record model

`AudioRecord` (frozen dataclass): `dataset`, `source_id`, `youtube_id`, `start_seconds`, `end_seconds`,
`original_label_ids` (MIDs), `original_labels` (display names), `salt_labels`, `split`, `events` (`TemporalEvent` tuple),
`raw` (the source fields as strings, for traceability), `source_path`, plus `duration_seconds`, `labels(label_space)` and
`record_id`.

### `record_id`

A stable content hash (40 hexadecimal characters). Exact definition:

```python
payload = [dataset, source_id, youtube_id, start_seconds, end_seconds, list(original_label_ids), list(original_labels)]
canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
record_id = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
```

Times enter as `float` (`None` is `null`, `-0.0` is `0.0`). These are the fields used by `deduplicate()`/`merge()`, so
equal `record_id` means "duplicate"; SALT labels, split, events, raw and source path are not part of it, hence a record keeps its
identifier when a SALT mapping is added or changed. It does not depend on the Python hash seed, platform or package
version. It is a fingerprint, not a security feature. `to_json`/`to_csv` write it (`record_id` key/column); the loaders ignore it and
recompute it.

Records are immutable but **not hashable** (they hold dictionaries): use `record_id` (or `source_id`) as a dictionary or set key, not the record.

## 3. Query spaces and matching

`filter_labels(labels, label_space="auto", match="any", include_descendants=False, case_sensitive=False)`:

| `label_space` | Compared with |
|---|---|
| `id` | the MIDs of the record (`/m/09x0r`) |
| `original` | the display names (`Speech`) |
| `salt` | the SALT standard events of the record |
| `auto` | all three together (ids, names and SALT labels; note that a query such as `music` matches both the AudioSet name `Music` and the SALT event `music`) |

`match="any"` keeps records with at least one query label, `"all"` with every query label, `"none"` with none of them (blacklist; also
`blacklist_labels(...)`). Comparison ignores case and repeated whitespace. With `label_space="salt"` and
`include_descendants=True` the query labels are first expanded with their SALT descendants (needs the resolver attached to the dataset;
without one, nothing is expanded). Native labels are never changed.

```python
speech = audioset.filter_labels(["Speech"], label_space="original")
no_music = audioset.filter_labels(["Music"], label_space="original", match="none")
vehicles = audioset.filter_labels(["vehicle"], label_space="salt", include_descendants=True)
```

## 4. Other dataset operations

| Operation | Notes |
|---|---|
| `select_range(i, j)` | zero-based, half-open (v1 `select_by_samp_idx` counted the header as row 0: v1 `(2, 5)` is v2 `(1, 4)`) |
| `group_by(field)` | by any record attribute or raw key; returns `{value: AudioDataset}` |
| `label_counts(label_space)`, `summary()`, `compute_stats(...)` | label histograms; summary = records, datasets, splits, unique YouTube ids, original and SALT label counts |
| `find_by_youtube_ids(ids)`, `find_samples_by_samples(targets, dataset_or_path)` | all records of those videos (one video can have several segments); returns **records**, not event rows |
| `deduplicate()`, `merge(*others, deduplicate=True)`, `merge_datasets`, `merge_files` | identity = `record_id` fields; the first occurrence is kept; merging keeps the order you pass |
| `rebalance(focus_labels, label_space, random_state)` | see [`rebalancing.md`](rebalancing.md) |
| `to_download_jobs()` | `DownloadJob` per record; see [`downloader.md`](downloader.md) |

Every operation returns a new immutable `AudioDataset` (the SALT resolver and `source_format` are carried over). The
functional wrappers (`filter_labels`, `blacklist_labels`, `select_range`, `rebalance`, `compute_stats`, `find_samples_by_*`,
`merge_files`) accept a dataset **or a path**.

## 5. Export and import

```python
audioset.to_json("subset.json")     # list of objects: every AudioRecord field + record_id
audioset.to_csv("subset.csv")       # columns: dataset, source_id, youtube_id, start_seconds, end_seconds,
                                    #   original_label_ids, original_labels, salt_labels, split,
                                    #   events, raw, source_path, record_id
reloaded = AudioDataset.from_file("subset.csv")        # detected as 'Normalized'; also for .json
assert reloaded.records == audioset.records
```

The round trip is lossless in both formats (tested for the three sources, empty datasets, `None` fields, Unicode, quotes and commas in
labels, and newlines in raw values). In the CSV, label tuples, `events` and `raw` are JSON-encoded cells, times are plain floats (empty cell =
`None`), empty `split`/`source_path` cells are `None`; no field is dropped. CSV files of the first prototype (nine columns, which it
could not read back) load too (missing columns default to empty; `source_path` then becomes the file read). A stored
`source_path` of `null` is kept as is; only files without the key fall back to their own path.

## 6. Temporal events (AudioSet Strong)

```python
from audioset_tools import generate_event_tracks, iter_events, plot_events_pianoroll

strong = AudioDataset.from_file("tests/data/audioset_strong/audioset_eval_strong_excerpt.tsv",
                                labels_file="tests/data/audioset_strong/mid_to_display_name_excerpt.tsv")
tracks = generate_event_tracks(strong, bin_size=0.5)        # [(f"{source_id}_{label}", array of 20 ints), ...]
for record, event, label in iter_events(strong, label_space="original"): ...
figures = plot_events_pianoroll(strong, segment_ids="s9d-2nhuJCQ_30000", save_plots=True, output_dir="plots")
```

Tracks have `int(10 / bin_size)` bins over the fixed 10-s clip (20 for 0.5 s, 33 for 0.3 s, 100 for 0.1 s); a bin is active when an event
overlaps it (onset floored, offset ceiled); events with the same label in one segment are merged; `label_space` is `original`, `id` or
`salt`; `out_type="pt"` returns torch tensors (`pip install "audioset-tools[torch]"`). The piano roll needs the `plot` extra
(matplotlib >= 3.6).

## 7. Retrieval, SALT and the command line

* Downloading clips: [`downloader.md`](downloader.md).
* SALT assets, `fetch-salt`, `from_py_salt`: [`salt.md`](salt.md).
* `audioset-tools --version`, `audioset-tools fetch-salt [--dest DIR] [--force]` (also `python -m audioset_tools`).
