# Clip retrieval: `YouTubeDatasetDownloader`

The downloader turns normalized `DownloadJob` objects (one per record: `dataset.to_download_jobs()`) into audio files.
It is deliberately small: yt-dlp fetches the whole stream, everything else (resampling, trimming, normalization,
channel policy, naming, reports, retries, resume, pauses, halts) is done by this package and is covered by offline tests.

> **Validation boundary.** The test-suite replaces `yt_dlp` with a fake module that writes synthetic WAV files or
> raises chosen errors. It therefore proves the behaviour of the code *around* yt-dlp. It does **not** prove anything about
> YouTube, about the current behaviour of the real yt-dlp, about cookies, bans or rate limits: no request to YouTube
> was made in the validation of this release. The yt-dlp option names used here were checked to exist in the embedding
> documentation of yt-dlp 2026.10.21 (`format`, `outtmpl`, `postprocessors`, `cookiefile`, `cookiesfrombrowser`,
> `ffmpeg_location`, `sleep_interval_requests`, `quiet`, `no_warnings`).
> Downloading is subject to the YouTube terms of service and to the rights of the video owners; you are responsible
> for your use.

## Prerequisites

`pip install "audioset-tools[download]"` (yt-dlp, soundfile, resampy) **and** the `ffmpeg` executable on `PATH`
(or `ydl_options={"ffmpeg_location": ...}`). With `dry_run=False` the constructor checks all four and fails
immediately (`ImportError` with the remedy, or `DownloaderSetupError` for ffmpeg); `check_prerequisites=False`
postpones the check, `downloader.check_prerequisites()` repeats it. `dry_run=True` needs none of them.

## Pipeline of one job

1. `invalid_metadata` is reported for a malformed YouTube ID (11 characters of `[A-Za-z0-9_-]`); nothing else is attempted.
2. `dry_run=True` reports `skipped_dry_run` (no network, no file).
3. `on_exists="skip"`: an existing output means `skipped_exists` (no network).
4. yt-dlp downloads `bestaudio/best` into a temporary directory and converts the **full** stream to WAV
   (`FFmpegExtractAudio`). ffmpeg is used only here.
5. `soundfile.read` loads the WAV as `(frames, channels)` (1-D for mono).
6. If the rate differs from `target_sr`: `resampy.resample(data, sr, target_sr, axis=0)`. `axis=0` is the time axis; see
   the stereo defect of v1 in [`MIGRATION_FROM_V1.md`](MIGRATION_FROM_V1.md#5-the-stereo-resampling-defect-of-v1).
7. Trimming by NumPy slicing at `target_sr` (`start_seconds`/`end_seconds` of the job; no interval = whole stream;
   an interval beyond the end of the stream is cut at the end).
8. Optional peak normalization (`normalize=True`; silence is left unchanged).
9. Channel policy: `stereo` keeps the source layout (a mono source stays mono; there is no up-mixing), `mono` averages the
   channels, `mono_split` writes `<stem>_Left.wav` and `<stem>_Right.wav` for stereo sources (a mono source gives the
   single file `<stem>.wav`). `<stem>` is the sanitized `source_id` of the job.
10. The file is written (PCM_16 WAV), and a `DownloadReportEntry` is returned.

### Output names and `on_exists` (F-6)

| `on_exists` | When `<stem>.wav` already exists |
|---|---|
| `"rename"` (default; previous behaviour) | the new file is `<stem>_1.wav`, then `<stem>_2.wav`, ... (repeated runs duplicate files) |
| `"skip"` | no download; status `skipped_exists`, `success=True`, `final_output_path` = the existing file (resume by file) |
| `"overwrite"` | the file is replaced |

For `mono_split` a clip counts as present when `_Left` **and** `_Right` exist (or the single mono file).

### Report entries

`download_job` never raises for a download problem: it returns a `DownloadReportEntry` with `status` in
`downloaded`, `failed`, `invalid_metadata`, `skipped_dry_run`, `skipped_exists`, the requested and final path, the exception
class name (`error_category`) and message, the number of retries used and a UTC timestamp. `download_jobs(jobs,
limit=None, report_path=None)` writes the list as JSON **also when the run is halted or interrupted**.
`read_download_report(path)` loads it.

## Robustness features ported from v1 (all opt-in, off by default)

| Feature | v2 option | What v1 did | Differences |
|---|---|---|---|
| Resume / skip downloaded items | `on_exists="skip"` | in-place `downloaded` flag in the metadata CSV, plus an existing `*<yt_id>*.wav` | state is the output file, not a flag in the metadata (the metadata are never modified); matching is by output stem (`source_id`), not by YouTube ID anywhere in a file name |
| Random pauses between requests | `random_pause=(5, 20)`, `pause_seed` | `random.uniform(5, 20)` seconds after every sample that was not skipped | default off; also available as a fixed `sleep_seconds`; both add up; applied only after jobs that contacted the network and never after the last job |
| Halt on YouTube shadow ban | `halt_on_shadow_ban=True`, `shadow_ban_messages` | three message strings, then the whole run returned silently | raises `ShadowBanError` (with `reason`, `job`, `entries`); the job is not retried; the partial report is written; matching ignores case and typographic apostrophes; messages are configurable |
| Halt after repeated failures | `max_consecutive_failures=N` | not available | raises `ConsecutiveFailuresError`; successes reset the streak, invalid/dry-run/skipped jobs neither count nor reset |
| Cookies | `cookies_file`, `cookies_from_browser` (browser name or `(browser, profile, keyring, container)`), `ydl_options` | `cookies_file` only | yt-dlp's own browser extraction is exposed |
| Cookie refresh | `cookie_refresh_hook(job, error) -> bool` | `refresh_cookies()` launched Firefox (`subprocess.run(["firefox"])`, a `pgrep` loop that never waited) and the job was not retried | not reproduced: no browser or `pgrep` specifics; **you** decide how to refresh (the hook can re-export `downloader.cookies_file`, which is re-read on every attempt); called at most once per job, only for the "Sign in to confirm you're not a bot" error; returning `True` retries immediately and does not consume a configured retry |
| Request delay | `ydl_options={"sleep_interval_requests": 1.25}` | `'sleep_requests': 1.25`, which is **not** a yt-dlp option and was ignored (the documented 1.25 s was never applied) | use the real option name |

Other yt-dlp options (proxy, rate limit, `ffmpeg_location`, ...) go through `ydl_options`; `outtmpl` and `postprocessors`
are managed by the downloader and rejected.

### Examples

Resume a long run (the second call downloads only what is missing and keeps writing one report per call):

```python
from audioset_tools import YouTubeDatasetDownloader, load_dataset

dataset = load_dataset("eval_segments.csv", labels_file="class_labels_indices.csv").filter_labels(["Siren"])
downloader = YouTubeDatasetDownloader("clips", target_sr=32000, channels="mono", on_exists="skip",
                                      random_pause=(5, 20), cookies_file="cookies.txt",
                                      halt_on_shadow_ban=True, max_consecutive_failures=25)
entries = downloader.download_dataset(dataset, report_path="report.json")
```

Handle a halt:

```python
from audioset_tools import DownloadHaltedError

try:
    downloader.download_dataset(dataset, report_path="report.json")
except DownloadHaltedError as halt:
    print(halt.reason, halt.job.youtube_id, len(halt.entries))   # report.json already holds halt.entries
```

A manual cookie refresh, the closest to what v1 asked for (example only, not exercised in the validation):

```python
def refresh(job, error):
    input("Re-export cookies.txt from your browser, then press Enter ... ")
    return True   # retry this clip once

downloader = YouTubeDatasetDownloader("clips", cookies_file="cookies.txt", cookie_refresh_hook=refresh)
```

## What the tests cover

`tests/test_downloader.py`: constructor validation; fail-fast on missing ffmpeg/packages (and the `ffmpeg_location` case);
the DSP path with synthetic signals (stereo 48, 44.1 and 32 kHz sources: output rate, frame count, per-channel
frequencies and the position of a marker, for `stereo`, `mono`, `mono_split`, mono sources, trimming at the end of the
stream, normalization and silence); report content and JSON round trip; `rename`/`skip`/`overwrite`; random and fixed
pauses (recorded, seeded; none after the last job or after skipped/invalid/dry-run jobs); halts (every v1 message, no retry
of a banned job, partial report written, custom messages, streak semantics); cookie/yt-dlp option pass-through; the
refresh hook (retry, once per job, only for authentication errors, declined, failing). Removing `axis=0` from the
resampling call makes the stereo regression tests fail (recorded in the validation evidence).
