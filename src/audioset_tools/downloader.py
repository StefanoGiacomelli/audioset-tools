"""YouTube audio downloader for normalized AudioSet-Tools records.

Retrieval robustness features ported from AudioSet-Tools v1 (all opt-in, all off by default):

* ``on_exists="skip"``: resume by file, i.e. skip clips whose output already exists (no network access);
* ``random_pause=(low, high)``: random pause between requests (v1: 5-20 s);
* ``halt_on_shadow_ban`` / ``max_consecutive_failures``: stop a long run with a clear exception;
* ``cookies_file`` / ``cookies_from_browser`` / ``cookie_refresh_hook``: authentication cookies and a refresh hook.

The network behaviour of this module (YouTube access, yt-dlp, cookies, bans) cannot be exercised offline: the test
suite replaces ``yt_dlp`` with a fake module. See ``docs/downloader.md`` for what that does and does not prove.
"""

from __future__ import annotations

import importlib
import json
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .models import DownloadJob, DownloadReportEntry, is_valid_youtube_id, safe_output_stem

#: Error messages that v1 treated as a YouTube shadow ban (the whole run stopped). Compared case-insensitively
#: after replacing typographic apostrophes.
SHADOW_BAN_MESSAGES: tuple[str, ...] = (
    "This content isn't available, try again later.",
    "Video unavailable. This content isn't available.",
    "The following content is not available on this app.. Watch on the latest version of YouTube.",
)
#: Error messages meaning that YouTube asks for authenticated cookies (v1 launched a browser for them).
AUTH_REQUIRED_MESSAGES: tuple[str, ...] = ("Sign in to confirm you're not a bot",)
ON_EXISTS_CHOICES: tuple[str, ...] = ("rename", "skip", "overwrite")


class DownloaderSetupError(RuntimeError):
    """A prerequisite of real downloads is missing (for example the ``ffmpeg`` executable)."""


class DownloadHaltedError(RuntimeError):
    """A download run was stopped on purpose to protect the account/IP (opt-in).

    Attributes: ``reason`` (``"shadow_ban"`` or ``"consecutive_failures"``), ``job`` (the job that triggered the
    halt), ``entries`` (report entries of every job processed so far, including the failing one) and
    ``consecutive_failures``. If ``download_jobs`` was given a ``report_path`` the partial report has been written
    before the exception propagates.
    """

    def __init__(self,
                 message: str,
                 reason: str,
                 job: DownloadJob | None = None,
                 entries: Sequence[DownloadReportEntry] = (),
                 consecutive_failures: int = 0) -> None:
        super().__init__(message)
        self.reason = reason
        self.job = job
        self.entries = list(entries)
        self.consecutive_failures = consecutive_failures


class ShadowBanError(DownloadHaltedError):
    """An error message matching ``shadow_ban_messages`` was received (YouTube shadow ban suspected)."""


class ConsecutiveFailuresError(DownloadHaltedError):
    """``max_consecutive_failures`` downloads failed in a row."""


class YouTubeDatasetDownloader:
    """Download and post-process YouTube clips from normalized jobs.

    This class adapts the useful behavior from the previous AudioSet-Tools
    downloader while avoiding in-place metadata mutation and text-only reports.

    Parameters
    ----------
    output_dir, target_sr, channels, normalize
        Output folder, sampling rate, channel policy (``stereo`` keeps the source layout, ``mono`` averages the
        channels, ``mono_split`` writes ``_Left``/``_Right`` files for stereo sources) and optional peak
        normalization.
    cookies_file
        Netscape-format cookie file passed to yt-dlp (``cookiefile``).
    retries, sleep_seconds, verbose, dry_run
        Extra attempts per job, fixed pause (seconds) after every job that contacted the network, extra output,
        and a mode that validates the jobs and reports ``skipped_dry_run`` without any network or file access.
    on_exists
        What to do when the output file exists: ``"rename"`` (default, previous behaviour: write ``<stem>_1.wav``,
        ``<stem>_2.wav``, ...), ``"skip"`` (resume: report ``skipped_exists`` without downloading) or
        ``"overwrite"``. For ``channels="mono_split"`` a clip counts as present when both ``_Left`` and ``_Right``
        exist (or the single mono file written for mono sources).
    random_pause
        ``(low, high)`` seconds: an additional uniformly random pause after every job that contacted the network
        (v1: ``(5, 20)``). ``pause_seed`` makes the pauses reproducible.
    halt_on_shadow_ban, shadow_ban_messages
        Raise :class:`ShadowBanError` from ``download_jobs`` when a failure message contains one of the messages
        (default: the three strings of v1) and do not retry that job.
    max_consecutive_failures
        Raise :class:`ConsecutiveFailuresError` after that many failed jobs in a row (successes reset the count;
        invalid metadata, dry-run and skipped jobs neither count nor reset).
    cookies_from_browser
        yt-dlp ``cookiesfrombrowser``: a browser name (``"firefox"``) or a tuple ``(browser, profile, keyring,
        container)``.
    cookie_refresh_hook
        ``hook(job, error) -> bool`` called at most once per job when YouTube asks for authentication; returning
        ``True`` retries the job immediately (the hook can e.g. re-export ``cookies_file``, which is re-read on every
        attempt). v1 launched Firefox and did not retry.
    ydl_options
        Extra yt-dlp options (e.g. ``{"sleep_interval_requests": 1.25, "proxy": ...}``); ``outtmpl`` and
        ``postprocessors`` are managed by this class and cannot be set.
    check_prerequisites
        With ``dry_run=False`` the constructor checks that ``yt_dlp``, ``soundfile`` and ``resampy`` import and that
        ``ffmpeg`` is available, and fails fast otherwise. Pass ``False`` to defer the check.
    """

    def __init__(self,
                 output_dir: str | Path,
                 target_sr: int = 44100,
                 channels: str = "stereo",
                 normalize: bool = False,
                 cookies_file: str | Path | None = None,
                 retries: int = 1,
                 sleep_seconds: float = 0.0,
                 verbose: bool = False,
                 dry_run: bool = False,
                 *,
                 on_exists: str = "rename",
                 random_pause: tuple[float, float] | None = None,
                 pause_seed: int | None = None,
                 halt_on_shadow_ban: bool = False,
                 shadow_ban_messages: Sequence[str] = SHADOW_BAN_MESSAGES,
                 max_consecutive_failures: int | None = None,
                 cookies_from_browser: str | Sequence[str | None] | None = None,
                 cookie_refresh_hook: Callable[[DownloadJob, Exception], bool | None] | None = None,
                 ydl_options: Mapping[str, Any] | None = None,
                 check_prerequisites: bool = True) -> None:

        if channels not in {"stereo", "mono", "mono_split"}:
            raise ValueError("channels must be one of 'stereo', 'mono', or 'mono_split'.")
        if on_exists not in ON_EXISTS_CHOICES:
            raise ValueError(f"on_exists must be one of {ON_EXISTS_CHOICES}.")
        if random_pause is not None:
            low, high = random_pause
            if not 0 <= low <= high:
                raise ValueError("random_pause must be a (low, high) pair with 0 <= low <= high.")
        if max_consecutive_failures is not None and max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be a positive integer or None.")
        managed = {"outtmpl", "postprocessors"}.intersection(ydl_options or {})
        if managed:
            raise ValueError(f"ydl_options cannot set {sorted(managed)}: they are managed by the downloader.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        self.channels = channels
        self.normalize = normalize
        self.cookies_file = Path(cookies_file) if cookies_file else None
        self.retries = max(0, retries)
        self.sleep_seconds = max(0.0, sleep_seconds)
        self.verbose = verbose
        self.dry_run = dry_run
        self.on_exists = on_exists
        self.random_pause = tuple(random_pause) if random_pause is not None else None
        self.halt_on_shadow_ban = halt_on_shadow_ban
        self.shadow_ban_messages = tuple(shadow_ban_messages)
        self.max_consecutive_failures = max_consecutive_failures
        self.cookies_from_browser = cookies_from_browser
        self.cookie_refresh_hook = cookie_refresh_hook
        self.ydl_options = dict(ydl_options or {})
        self._rng = random.Random(pause_seed)
        self._sleep: Callable[[float], None] = time.sleep

        if check_prerequisites and not dry_run:
            self.check_prerequisites()

    def check_prerequisites(self) -> None:
        """Raise if real downloads cannot work: missing Python packages (ImportError) or ``ffmpeg`` (setup error)."""

        _import_download_modules()
        if not self._ffmpeg_available():
            raise DownloaderSetupError("ffmpeg was not found: yt-dlp needs the ffmpeg executable on PATH (or the "
                                       "'ffmpeg_location' yt-dlp option) to extract WAV audio. Install ffmpeg or "
                                       "use dry_run=True.")

    def download_dataset(self,
                         dataset,
                         limit: int | None = None,
                         report_path: str | Path | None = None) -> list[DownloadReportEntry]:
        return self.download_jobs(dataset.to_download_jobs(), limit=limit, report_path=report_path)

    def download_jobs(self,
                      jobs: Iterable[DownloadJob],
                      limit: int | None = None,
                      report_path: str | Path | None = None) -> list[DownloadReportEntry]:
        job_list = list(jobs)
        if limit is not None:
            job_list = job_list[:limit]

        report_entries: list[DownloadReportEntry] = []
        consecutive_failures = 0
        try:
            for index, job in enumerate(tqdm(job_list, desc="Downloading YouTube clips")):
                entry = self.download_job(job)
                report_entries.append(entry)

                if entry.status == "failed":
                    consecutive_failures += 1
                    if self.halt_on_shadow_ban and self._matches(entry.error_message, self.shadow_ban_messages):
                        raise ShadowBanError(f"YouTube shadow ban suspected at {job.youtube_id} "
                                             f"({entry.error_message}); run halted after {len(report_entries)} "
                                             "job(s). Wait, change IP/cookies, then resume with on_exists='skip'.",
                                             "shadow_ban", job, report_entries, consecutive_failures)
                    if (self.max_consecutive_failures is not None
                            and consecutive_failures >= self.max_consecutive_failures):
                        raise ConsecutiveFailuresError(f"{consecutive_failures} consecutive download failures "
                                                       f"(last: {job.youtube_id}: {entry.error_message}); run "
                                                       f"halted after {len(report_entries)} job(s).",
                                                       "consecutive_failures", job, report_entries,
                                                       consecutive_failures)
                elif entry.status == "downloaded":
                    consecutive_failures = 0

                if entry.status in {"downloaded", "failed"} and index < len(job_list) - 1:
                    self._pause()
        finally:
            if report_path is not None:
                write_download_report(report_entries, report_path)

        return report_entries

    def download_job(self, job: DownloadJob) -> DownloadReportEntry:
        requested_path = self._output_path(job)
        if not is_valid_youtube_id(job.youtube_id):
            return self._report(job,
                                requested_path=requested_path,
                                final_path=None,
                                status="invalid_metadata",
                                success=False,
                                error_category="InvalidYouTubeId",
                                error_message=f"Malformed YouTube ID: {job.youtube_id}")
        if self.dry_run:
            return self._report(job,
                                requested_path=requested_path,
                                final_path=None,
                                status="skipped_dry_run",
                                success=True)
        if self.on_exists == "skip":
            existing = self._existing_outputs(requested_path)
            if existing:
                return self._report(job,
                                    requested_path=requested_path,
                                    final_path=existing[0],
                                    status="skipped_exists",
                                    success=True)

        last_error: Exception | None = None
        attempts = 0
        budget = self.retries + 1
        refresh_used = False
        while attempts < budget:
            attempts += 1
            try:
                final_paths = self._download_and_process(job, requested_path)
                return self._report(job,
                                    requested_path=requested_path,
                                    final_path=final_paths[0] if final_paths else None,
                                    status="downloaded",
                                    success=True,
                                    retry_count=attempts - 1)
            except Exception as exc:  # noqa: BLE001 - categorized in report
                last_error = exc
                if self.verbose:
                    print(f"Download failed for {job.youtube_id} on attempt {attempts}: {exc}")
                message = str(exc)
                if (self.cookie_refresh_hook is not None and not refresh_used
                        and self._matches(message, AUTH_REQUIRED_MESSAGES)):
                    refresh_used = True
                    try:
                        refreshed = bool(self.cookie_refresh_hook(job, exc))
                    except Exception as hook_exc:  # noqa: BLE001 - reported as the failure of the job
                        last_error = RuntimeError(f"cookie_refresh_hook failed: {hook_exc}")
                        last_error.__cause__ = hook_exc
                        break
                    if refreshed:
                        budget += 1  # the attempt after a refresh does not consume a configured retry
                        continue
                if self.halt_on_shadow_ban and self._matches(message, self.shadow_ban_messages):
                    break

        return self._report(job,
                            requested_path=requested_path,
                            final_path=None,
                            status="failed",
                            success=False,
                            error_category=type(last_error).__name__ if last_error else "UnknownError",
                            error_message=str(last_error) if last_error else "Unknown error",
                            retry_count=max(0, attempts - 1))

    def _download_and_process(self, job: DownloadJob, output_path: Path) -> list[Path]:
        yt_dlp, sf, resampy = _import_download_modules()

        with tempfile.TemporaryDirectory(prefix="audioset_tools_") as tmp:
            tmp_dir = Path(tmp)
            outtmpl = str(tmp_dir / f"{safe_output_stem(job.youtube_id)}.%(ext)s")
            ydl_opts: dict[str, Any] = {"quiet": not self.verbose,
                                        "no_warnings": not self.verbose,
                                        "format": "bestaudio/best"}
            ydl_opts.update(self.ydl_options)
            ydl_opts["outtmpl"] = outtmpl
            ydl_opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]
            if self.cookies_file:
                ydl_opts["cookiefile"] = str(self.cookies_file)
            if self.cookies_from_browser:
                browser = self.cookies_from_browser
                ydl_opts["cookiesfrombrowser"] = (browser,) if isinstance(browser, str) else tuple(browser)

            url = f"https://www.youtube.com/watch?v={job.youtube_id}"

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            wav_files = list(tmp_dir.glob("*.wav"))
            if not wav_files:
                raise RuntimeError("yt-dlp completed without producing a WAV file.")

            data, sr = sf.read(wav_files[0])

            if sr != self.target_sr:
                # axis=0 is the time axis of soundfile's (frames, channels) arrays. resampy's default (axis=-1)
                # would resample the channel axis of a stereo clip (the AudioSet-Tools v1 defect, PR #1).
                data = resampy.resample(data, sr, self.target_sr, axis=0)

            if job.start_seconds is not None and job.end_seconds is not None:
                start_idx = max(0, int(job.start_seconds * self.target_sr))
                end_idx = max(start_idx, int(job.end_seconds * self.target_sr))
                data = data[start_idx:end_idx]

            if self.normalize:
                peak = np.max(np.abs(data)) if data.size else 0.0
                if peak > 0:
                    data = data / peak

            output_path = self._claim(output_path)

            if self.channels == "mono" and data.ndim == 2:
                data = data.mean(axis=1)
                sf.write(output_path, data, self.target_sr)

                return [output_path]

            if self.channels == "mono_split" and data.ndim == 2 and data.shape[1] > 1:
                left = self._claim(self._split_path(output_path, "Left"))
                right = self._claim(self._split_path(output_path, "Right"))
                sf.write(left, data[:, 0], self.target_sr)
                sf.write(right, data[:, 1], self.target_sr)

                return [left, right]

            sf.write(output_path, data, self.target_sr)

            return [output_path]

    def _output_path(self, job: DownloadJob) -> Path:
        stem = safe_output_stem(job.output_stem or job.source_id or job.youtube_id)
        return self.output_dir / f"{stem}.wav"

    @staticmethod
    def _split_path(output_path: Path, side: str) -> Path:
        return output_path.with_name(f"{output_path.stem}_{side}{output_path.suffix}")

    def _claim(self, path: Path) -> Path:
        """Return the path to write: unchanged for ``overwrite``, the next free name for ``rename``/``skip``."""

        if self.on_exists == "overwrite":
            return path

        return _next_available_path(path)

    def _existing_outputs(self, requested_path: Path) -> list[Path]:
        if self.channels == "mono_split":
            left = self._split_path(requested_path, "Left")
            right = self._split_path(requested_path, "Right")
            if left.exists() and right.exists():
                return [left, right]

        return [requested_path] if requested_path.exists() else []

    def _ffmpeg_available(self) -> bool:
        location = self.ydl_options.get("ffmpeg_location")
        if location:
            path = Path(str(location)).expanduser()
            if path.is_dir():
                return any((path / name).exists() for name in ("ffmpeg", "ffmpeg.exe"))
            return path.exists()

        return shutil.which("ffmpeg") is not None

    def _pause(self) -> None:
        delay = self.sleep_seconds
        if self.random_pause is not None:
            delay += self._rng.uniform(*self.random_pause)
        if delay > 0:
            if self.verbose:
                print(f"Pausing for {delay:.2f} s between requests.")
            self._sleep(delay)

    @staticmethod
    def _matches(message: str | None, patterns: Iterable[str]) -> bool:
        text = _normalize_message(message or "")

        return any(_normalize_message(pattern) in text for pattern in patterns)

    def _report(self,
                job: DownloadJob,
                requested_path: Path,
                final_path: Path | None,
                status: str,
                success: bool,
                error_category: str | None = None,
                error_message: str | None = None,
                retry_count: int = 0) -> DownloadReportEntry:

        return DownloadReportEntry(dataset=job.dataset,
                                   source_id=job.source_id,
                                   youtube_id=job.youtube_id,
                                   start_seconds=job.start_seconds,
                                   end_seconds=job.end_seconds,
                                   requested_output_path=str(requested_path),
                                   final_output_path=str(final_path) if final_path else None,
                                   status=status,
                                   success=success,
                                   error_category=error_category,
                                   error_message=error_message,
                                   retry_count=retry_count,
                                   processing_timestamp=datetime.now(timezone.utc).isoformat(),
                                   original_label_ids=job.original_label_ids,
                                   original_labels=job.original_labels,
                                   salt_labels=job.salt_labels)


def write_download_report(entries: Iterable[DownloadReportEntry],
                          path: str | Path) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([entry.to_dict() for entry in entries], handle, indent=2)


def read_download_report(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _import_download_modules():
    """Import yt_dlp, soundfile and resampy (the ``download`` extra) or raise a precise ImportError."""

    try:
        yt_dlp = importlib.import_module("yt_dlp")
        sf = importlib.import_module("soundfile")
        resampy = importlib.import_module("resampy")
    except ImportError as exc:
        raise ImportError("Downloading requires yt-dlp, soundfile and resampy: "
                          "pip install 'audioset-tools[download]' (and the ffmpeg executable).") from exc

    return yt_dlp, sf, resampy


def _normalize_message(text: str) -> str:
    return text.replace("’", "'").lower()


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate an output filename for {path}.")
