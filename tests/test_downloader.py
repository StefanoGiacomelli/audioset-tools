"""YouTubeDatasetDownloader with a fake ``yt_dlp``: DSP path, report, resume, pauses, halts and cookies.

Nothing here touches the network. The fake module replaces ``yt_dlp.YoutubeDL`` and writes synthetic WAV files, so
these tests prove the behaviour of the code around yt-dlp (resampling, trimming, channel policy, naming, reports,
retry/halt/pause/cookie logic) and NOT the behaviour of YouTube or of the real yt-dlp (see docs/downloader.md).
"""

from __future__ import annotations

import json
import random
import sys
import types

import numpy as np
import pytest

from audioset_tools import (
    AudioDataset,
    ConsecutiveFailuresError,
    DownloaderSetupError,
    DownloadHaltedError,
    DownloadJob,
    ShadowBanError,
    YouTubeDatasetDownloader,
    is_valid_youtube_id,
    read_download_report,
    write_download_report,
)
from audioset_tools.models import safe_output_stem

from .conftest import tone

SHADOW_BAN = "ERROR: [youtube] {}: This content isn’t available, try again later."
AUTH = "ERROR: [youtube] {}: Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies"


def job(index: int = 0, start: float | None = 1.0, end: float | None = 6.0, **kwargs) -> DownloadJob:
    video = f"vid{index:08d}"
    return DownloadJob(dataset="VggSound", source_id=f"{video}_1", youtube_id=video, start_seconds=start,
                       end_seconds=end, original_labels=("wind noise",), salt_labels=("wind",), **kwargs)


def vid(index: int) -> str:
    return f"vid{index:08d}"


def make_downloader(tmp_path, **kwargs) -> YouTubeDatasetDownloader:
    kwargs.setdefault("target_sr", 32000)
    kwargs.setdefault("retries", 0)
    return YouTubeDatasetDownloader(tmp_path / "out", **kwargs)


def marked_stereo(sample_rate: int, seconds: float = 12.0) -> np.ndarray:
    """Left 440 Hz, right 1000 Hz, plus a 0.5 s 2 kHz marker (both channels) starting at t = 3.0 s."""

    left, right = tone(440.0, 1, seconds, sample_rate), tone(1000.0, 1, seconds, sample_rate)
    t = np.arange(int(0.5 * sample_rate)) / sample_rate
    burst = 0.5 * np.sin(2 * np.pi * 2000.0 * t)
    start = int(3.0 * sample_rate)
    left[start:start + len(burst)] += burst
    right[start:start + len(burst)] += burst
    return np.stack([left, right], axis=1)


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return float(np.fft.rfftfreq(len(signal), 1 / sample_rate)[np.argmax(spectrum)])


def marker(signal: np.ndarray, sample_rate: int, frame: float = 0.1) -> tuple[float, float]:
    """(onset, duration) in seconds of the 2 kHz burst, from the 2 kHz spectral bin of consecutive frames."""

    size = int(frame * sample_rate)
    window = np.hanning(size)
    bin_index = int(round(2000.0 * size / sample_rate))
    levels = np.array([abs(np.fft.rfft(signal[i:i + size] * window)[bin_index])
                       for i in range(0, len(signal) - size + 1, size)])
    active = np.flatnonzero(levels > 0.25 * levels.max())
    return float(active[0] * frame), float(len(active) * frame)


# ------------------------------------------------------------------------------ setup, dry run, fail-fast

def test_dry_run_and_invalid_youtube_id_report(tmp_path):
    downloader = YouTubeDatasetDownloader(tmp_path / "out", dry_run=True)  # no prerequisite is checked
    valid = downloader.download_job(DownloadJob(dataset="VggSound", source_id="---g-f_I2yQ_1",
                                                youtube_id="---g-f_I2yQ", start_seconds=1.0, end_seconds=11.0))
    assert valid.success and valid.status == "skipped_dry_run" and valid.final_output_path is None
    assert valid.requested_output_path.endswith("---g-f_I2yQ_1.wav")
    invalid = downloader.download_job(DownloadJob(dataset="VggSound", source_id="bad", youtube_id="not-valid"))
    assert not invalid.success and invalid.error_category == "InvalidYouTubeId" and invalid.status == "invalid_metadata"
    assert list((tmp_path / "out").iterdir()) == []  # nothing is written in a dry run


def test_helpers_validate_ids_and_stems():
    assert is_valid_youtube_id("---g-f_I2yQ") and is_valid_youtube_id("abcdefghijk")
    assert not is_valid_youtube_id("short") and not is_valid_youtube_id("has space!") and not is_valid_youtube_id("")
    assert safe_output_stem("a b/c:d") == "a_b_c_d" and safe_output_stem("///") == "audio"


def test_constructor_validates_arguments(tmp_path):
    for kwargs, message in [({"channels": "surround"}, "channels must be"),
                            ({"on_exists": "delete"}, "on_exists must be"),
                            ({"random_pause": (5, 1)}, "random_pause"),
                            ({"random_pause": (-1, 1)}, "random_pause"),
                            ({"max_consecutive_failures": 0}, "max_consecutive_failures"),
                            ({"ydl_options": {"outtmpl": "x"}}, "managed by the downloader"),
                            ({"ydl_options": {"postprocessors": []}}, "managed by the downloader")]:
        with pytest.raises(ValueError, match=message):
            YouTubeDatasetDownloader(tmp_path / "x", dry_run=True, **kwargs)


def test_missing_ffmpeg_fails_fast_unless_dry_run(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    pytest.importorskip("resampy")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.ModuleType("yt_dlp"))
    monkeypatch.setattr("audioset_tools.downloader.shutil.which", lambda name: None)
    with pytest.raises(DownloaderSetupError, match="ffmpeg"):
        YouTubeDatasetDownloader(tmp_path / "a")
    YouTubeDatasetDownloader(tmp_path / "b", dry_run=True)
    YouTubeDatasetDownloader(tmp_path / "c", check_prerequisites=False)
    folder = tmp_path / "bin"
    folder.mkdir()
    with pytest.raises(DownloaderSetupError):
        YouTubeDatasetDownloader(tmp_path / "d", ydl_options={"ffmpeg_location": str(folder)})  # no ffmpeg inside
    (folder / "ffmpeg").write_text("", encoding="utf-8")
    YouTubeDatasetDownloader(tmp_path / "e", ydl_options={"ffmpeg_location": str(folder)})
    YouTubeDatasetDownloader(tmp_path / "f", ydl_options={"ffmpeg_location": str(folder / "ffmpeg")})


def test_missing_python_packages_fail_fast_with_the_remedy(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # an import of yt_dlp raises ImportError
    with pytest.raises(ImportError, match=r"audioset-tools\[download\]"):
        YouTubeDatasetDownloader(tmp_path / "a")
    YouTubeDatasetDownloader(tmp_path / "b", dry_run=True)


def test_ffmpeg_available_through_path_lookup(tmp_path, fake_yt_dlp):
    assert make_downloader(tmp_path).output_dir.exists()  # the fixture's fake ffmpeg is found through which()


# ------------------------------------------------------------------------------ DSP path (axis=0 regression)

@pytest.mark.parametrize("source_rate", [48000, 44100, 32000])
def test_stereo_clip_is_resampled_trimmed_and_keeps_both_channels(tmp_path, fake_yt_dlp, source_rate):
    # Regression of the AudioSet-Tools v1 defect: resampy's default axis=-1 would resample the 2-element channel
    # axis of a (frames, 2) array, returning (frames, 1) at the ORIGINAL rate; the wrong window would then be cut and
    # the file stamped with the target rate. With axis=0 the time axis is resampled and both channels survive.
    import soundfile as sf

    fake_yt_dlp.behaviours[vid(1)] = (marked_stereo(source_rate), source_rate)
    entry = make_downloader(tmp_path, channels="stereo").download_job(job(1, start=1.0, end=6.0))
    assert entry.status == "downloaded" and entry.success
    data, rate = sf.read(entry.final_output_path)
    assert rate == 32000 and data.shape == (5 * 32000, 2)  # exactly the 5-s window at the target rate
    assert abs(dominant_frequency(data[:int(1.5 * rate), 0], rate) - 440.0) < 2.0  # not slowed, not mixed
    assert abs(dominant_frequency(data[:int(1.5 * rate), 1], rate) - 1000.0) < 2.0
    onset, duration = marker(data[:, 0], rate)
    assert abs(onset - 2.0) <= 0.1 and abs(duration - 0.5) <= 0.1  # 3.0 s in the source minus the 1.0 s start


def test_the_default_resampy_axis_would_not_give_the_right_result():
    # Documents the third-party behaviour behind axis=0 (it is not a statement about this package's code).
    resampy = pytest.importorskip("resampy")
    stereo = np.zeros((48000, 2))
    try:
        wrong = resampy.resample(stereo, 48000, 32000)
    except ValueError:
        return  # a future resampy that refuses the call is just as good
    assert wrong.shape != (32000, 2)
    assert resampy.resample(stereo, 48000, 32000, axis=0).shape == (32000, 2)


def test_mono_policy_averages_channels(tmp_path, fake_yt_dlp):
    import soundfile as sf

    fake_yt_dlp.behaviours[vid(1)] = (marked_stereo(48000), 48000)
    entry = make_downloader(tmp_path, channels="mono").download_job(job(1))
    data, rate = sf.read(entry.final_output_path)
    assert rate == 32000 and data.shape == (5 * 32000,)
    spectrum = np.abs(np.fft.rfft(data[:int(1.5 * rate)] * np.hanning(int(1.5 * rate))))
    freqs = np.fft.rfftfreq(int(1.5 * rate), 1 / rate)
    top_two = sorted(freqs[np.argsort(spectrum)[-6:]])
    assert any(abs(f - 440.0) < 3 for f in top_two) and any(abs(f - 1000.0) < 3 for f in top_two)


def test_mono_split_writes_left_and_right_files(tmp_path, fake_yt_dlp):
    import soundfile as sf

    fake_yt_dlp.behaviours[vid(1)] = (marked_stereo(44100), 44100)
    entry = make_downloader(tmp_path, channels="mono_split").download_job(job(1))
    left_path, right_path = (tmp_path / "out" / f"{vid(1)}_1_{side}.wav" for side in ("Left", "Right"))
    assert entry.final_output_path == str(left_path) and left_path.exists() and right_path.exists()
    left, rate = sf.read(left_path)
    right, _ = sf.read(right_path)
    assert left.shape == right.shape == (5 * 32000,)
    assert abs(dominant_frequency(left[:int(1.5 * rate)], rate) - 440.0) < 2.0
    assert abs(dominant_frequency(right[:int(1.5 * rate)], rate) - 1000.0) < 2.0


def test_mono_source_follows_the_same_path(tmp_path, fake_yt_dlp):
    import soundfile as sf

    source = tone(440.0, 1, 12.0, 44100)
    fake_yt_dlp.behaviours[vid(1)] = (source, 44100)
    fake_yt_dlp.behaviours[vid(2)] = (source, 44100)
    plain = make_downloader(tmp_path, channels="stereo").download_job(job(1))
    split = make_downloader(tmp_path, channels="mono_split").download_job(job(2))
    for entry in (plain, split):  # a mono source never produces _Left/_Right files
        data, rate = sf.read(entry.final_output_path)
        assert rate == 32000 and data.shape == (5 * 32000,)
        assert entry.final_output_path.endswith(f"{entry.youtube_id}_1.wav")
        assert abs(dominant_frequency(data[:int(1.5 * rate)], rate) - 440.0) < 2.0


def test_trimming_edge_cases_and_normalization(tmp_path, fake_yt_dlp):
    import soundfile as sf

    fake_yt_dlp.behaviours[vid(1)] = (tone(300.0, 2, 3.0, 32000), 32000)  # shorter than the requested window
    fake_yt_dlp.behaviours[vid(2)] = (tone(300.0, 1, 4.0, 32000), 32000)
    fake_yt_dlp.behaviours[vid(3)] = (np.zeros((32000, 2)), 32000)  # silence: normalization must not divide by zero
    downloader = make_downloader(tmp_path, channels="stereo")
    short = downloader.download_job(job(1, start=1.0, end=6.0))
    assert sf.read(short.final_output_path)[0].shape == (2 * 32000, 2)  # slice stops at the end of the signal
    whole = downloader.download_job(job(2, start=None, end=None))
    assert sf.read(whole.final_output_path)[0].shape == (4 * 32000,)  # no interval: the whole stream
    normalized = make_downloader(tmp_path, channels="stereo", normalize=True)
    loud = normalized.download_job(job(2, start=0.0, end=2.0))
    assert abs(np.max(np.abs(sf.read(loud.final_output_path)[0])) - 1.0) < 1e-3
    silent = normalized.download_job(job(3, start=0.0, end=1.0))
    assert not np.isnan(sf.read(silent.final_output_path)[0]).any()


# ------------------------------------------------------------------------------ reports and failures

def test_report_file_round_trip_and_fields(tmp_path, fake_yt_dlp):
    downloader = make_downloader(tmp_path)
    entries = downloader.download_jobs([job(1), job(2)], report_path=tmp_path / "report.json")
    loaded = read_download_report(tmp_path / "report.json")
    assert [row["youtube_id"] for row in loaded] == [vid(1), vid(2)]
    assert loaded == json.loads(json.dumps([entry.to_dict() for entry in entries]))
    first = loaded[0]
    assert first["status"] == "downloaded" and first["success"] is True and first["retry_count"] == 0
    assert first["salt_labels"] == ["wind"] and first["original_labels"] == ["wind noise"]
    assert first["requested_output_path"] == first["final_output_path"]
    assert first["processing_timestamp"].endswith("+00:00")
    write_download_report(entries, tmp_path / "again.json")
    assert read_download_report(tmp_path / "again.json") == loaded


def test_failures_are_categorized_and_retried(tmp_path, fake_yt_dlp):
    fake_yt_dlp.behaviours[vid(1)] = RuntimeError("boom")
    fake_yt_dlp.behaviours[vid(2)] = lambda calls: (ValueError("first") if len(calls) == 1
                                                    else (tone(300.0, 1, 8.0, 32000), 32000))
    fake_yt_dlp.behaviours[vid(3)] = None  # yt-dlp "succeeds" without producing a WAV
    downloader = make_downloader(tmp_path, retries=2)
    failed = downloader.download_job(job(1))
    assert (failed.status, failed.success, failed.error_category, failed.error_message, failed.retry_count) == \
           ("failed", False, "RuntimeError", "boom", 2)
    assert fake_yt_dlp.ids().count(vid(1)) == 3  # 1 attempt + 2 retries
    recovered = downloader.download_job(job(2))
    assert recovered.status == "downloaded" and recovered.retry_count == 1
    empty = downloader.download_job(job(3))
    assert empty.status == "failed" and "without producing a WAV" in empty.error_message


def test_download_dataset_uses_source_ids_as_stems_and_honours_limit(tmp_path, vgg):
    downloader = YouTubeDatasetDownloader(tmp_path / "out", dry_run=True)
    entries = downloader.download_dataset(vgg, limit=3, report_path=tmp_path / "dry.json")
    assert len(entries) == 3 and len(read_download_report(tmp_path / "dry.json")) == 3
    assert [e.requested_output_path.rsplit("/", 1)[-1] for e in entries] == [f"{r.source_id}.wav" for r in vgg[:3]]
    assert all(e.status == "skipped_dry_run" for e in entries)
    assert len(downloader.download_jobs(AudioDataset(()).to_download_jobs())) == 0


# ------------------------------------------------------------------------------ on_exists: skip, rename, overwrite

def test_default_rename_never_overwrites(tmp_path, fake_yt_dlp):
    downloader = make_downloader(tmp_path)
    first = downloader.download_job(job(1))
    second = downloader.download_job(job(1))
    third = downloader.download_job(job(1))
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == [f"{vid(1)}_1.wav", f"{vid(1)}_1_1.wav", f"{vid(1)}_1_2.wav"]
    assert [first.status, second.status, third.status] == ["downloaded"] * 3
    assert second.requested_output_path == first.requested_output_path != second.final_output_path


def test_skip_resumes_by_file_without_network_access(tmp_path, fake_yt_dlp):
    downloader = make_downloader(tmp_path, on_exists="skip")
    jobs = [job(1), job(2), job(3)]
    first = downloader.download_jobs(jobs)
    assert [e.status for e in first] == ["downloaded"] * 3 and fake_yt_dlp.ids() == [vid(1), vid(2), vid(3)]
    (tmp_path / "out" / f"{vid(2)}_1.wav").unlink()
    again = downloader.download_jobs(jobs)
    assert [e.status for e in again] == ["skipped_exists", "downloaded", "skipped_exists"]
    assert all(e.success for e in again)
    assert fake_yt_dlp.ids() == [vid(1), vid(2), vid(3), vid(2)]  # only the missing clip was requested again
    assert again[0].final_output_path == again[0].requested_output_path
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == [f"{vid(i)}_1.wav" for i in (1, 2, 3)]  # no renamed copies


def test_skip_with_mono_split_needs_both_channel_files(tmp_path, fake_yt_dlp):
    fake_yt_dlp.default = (marked_stereo(32000), 32000)
    downloader = make_downloader(tmp_path, on_exists="skip", channels="mono_split")
    assert downloader.download_job(job(1)).status == "downloaded"
    assert downloader.download_job(job(1)).status == "skipped_exists"
    (tmp_path / "out" / f"{vid(1)}_1_Right.wav").unlink()
    assert downloader.download_job(job(1)).status == "downloaded"  # incomplete pair: fetched again
    assert (tmp_path / "out" / f"{vid(1)}_1_Right.wav").exists()


def test_overwrite_replaces_the_file(tmp_path, fake_yt_dlp):
    import soundfile as sf

    downloader = make_downloader(tmp_path, on_exists="overwrite", channels="stereo")
    fake_yt_dlp.behaviours[vid(1)] = (tone(300.0, 2, 8.0, 32000), 32000)
    downloader.download_job(job(1))
    fake_yt_dlp.behaviours[vid(1)] = (tone(700.0, 2, 8.0, 32000), 32000)
    entry = downloader.download_job(job(1))
    assert [p.name for p in (tmp_path / "out").iterdir()] == [f"{vid(1)}_1.wav"]
    data, rate = sf.read(entry.final_output_path)
    assert abs(dominant_frequency(data[:, 0], rate) - 700.0) < 2.0


def test_skip_is_not_applied_in_dry_run_or_to_invalid_ids(tmp_path, fake_yt_dlp):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / f"{vid(1)}_1.wav").write_bytes(b"x")
    dry = YouTubeDatasetDownloader(tmp_path / "out", on_exists="skip", dry_run=True).download_job(job(1))
    assert dry.status == "skipped_dry_run"
    assert make_downloader(tmp_path, on_exists="skip").download_job(
        DownloadJob(dataset="X", source_id="bad", youtube_id="bad")).status == "invalid_metadata"


# ------------------------------------------------------------------------------ pauses

def patched(downloader: YouTubeDatasetDownloader, recorded: list[float]) -> YouTubeDatasetDownloader:
    downloader._sleep = recorded.append  # instance attribute set in __init__ takes precedence over the class
    return downloader


def test_random_pauses_are_opt_in_reproducible_and_skip_the_last_job(tmp_path, fake_yt_dlp):
    recorded: list[float] = []
    off = patched(make_downloader(tmp_path / "off"), recorded)
    off.download_jobs([job(1), job(2), job(3)])
    assert recorded == []  # default: no pause at all

    on = patched(make_downloader(tmp_path / "on", random_pause=(5, 20), pause_seed=42), recorded)
    on.download_jobs([job(1), job(2), job(3), job(4)])
    rng = random.Random(42)
    assert recorded == [rng.uniform(5, 20) for _ in range(3)]  # between consecutive jobs only, never after the last
    assert all(5 <= delay <= 20 for delay in recorded)


def test_fixed_and_random_pauses_add_up(tmp_path, fake_yt_dlp):
    recorded: list[float] = []
    both = patched(make_downloader(tmp_path, sleep_seconds=1.5, random_pause=(2, 2)), recorded)
    both.download_jobs([job(1), job(2)])
    assert recorded == [3.5]
    fixed: list[float] = []
    patched(make_downloader(tmp_path / "f", sleep_seconds=0.25), fixed).download_jobs([job(1), job(2), job(3)])
    assert fixed == [0.25, 0.25]


def test_only_jobs_that_contacted_the_network_are_followed_by_a_pause(tmp_path, fake_yt_dlp):
    recorded: list[float] = []
    downloader = patched(make_downloader(tmp_path, on_exists="skip", random_pause=(1, 1)), recorded)
    downloader.download_job(job(2))  # pre-existing file for the skip below
    fake_yt_dlp.behaviours[vid(3)] = RuntimeError("nope")
    downloader.download_jobs([job(1), job(2), DownloadJob(dataset="X", source_id="bad", youtube_id="bad"), job(3),
                              job(4)])
    # job 1 downloaded (pause), job 2 skipped (none), invalid (none), job 3 failed (pause), job 4 last (none)
    assert recorded == [1.0, 1.0]
    dry: list[float] = []
    dry_downloader = YouTubeDatasetDownloader(tmp_path / "d", dry_run=True, random_pause=(1, 1))
    patched(dry_downloader, dry).download_jobs([job(1), job(2)])
    assert dry == []


# ------------------------------------------------------------------------------ halting

def test_shadow_ban_halts_with_a_clear_exception_and_keeps_the_report(tmp_path, fake_yt_dlp):
    fake_yt_dlp.behaviours[vid(2)] = RuntimeError(SHADOW_BAN.format(vid(2)))
    downloader = make_downloader(tmp_path, retries=3, halt_on_shadow_ban=True)
    with pytest.raises(ShadowBanError, match=vid(2)) as caught:
        downloader.download_jobs([job(1), job(2), job(3)], report_path=tmp_path / "partial.json")
    error = caught.value
    assert isinstance(error, DownloadHaltedError) and error.reason == "shadow_ban" and error.job.youtube_id == vid(2)
    assert [e.status for e in error.entries] == ["downloaded", "failed"] and error.consecutive_failures == 1
    assert fake_yt_dlp.ids() == [vid(1)] + [vid(2)]  # the banned job is not retried, the third is never requested
    assert [row["status"] for row in read_download_report(tmp_path / "partial.json")] == ["downloaded", "failed"]


@pytest.mark.parametrize("message", ["This content isn't available, try again later.",
                                     "ERROR: Video unavailable. This content isn’t available.",
                                     "The following content is not available on this app.. Watch on the latest "
                                     "version of YouTube."])
def test_every_v1_shadow_ban_message_is_recognized(tmp_path, fake_yt_dlp, message):
    fake_yt_dlp.behaviours[vid(1)] = RuntimeError(message)
    with pytest.raises(ShadowBanError):
        make_downloader(tmp_path, halt_on_shadow_ban=True).download_jobs([job(1), job(2)])


def test_without_the_flag_a_shadow_ban_message_is_an_ordinary_failure(tmp_path, fake_yt_dlp):
    fake_yt_dlp.behaviours[vid(2)] = RuntimeError(SHADOW_BAN.format(vid(2)))
    entries = make_downloader(tmp_path, retries=1).download_jobs([job(1), job(2), job(3)])
    assert [e.status for e in entries] == ["downloaded", "failed", "downloaded"]
    assert fake_yt_dlp.ids().count(vid(2)) == 2  # retried as usual
    custom = make_downloader(tmp_path / "c", halt_on_shadow_ban=True, shadow_ban_messages=("banned!",))
    fake_yt_dlp.behaviours[vid(5)] = RuntimeError("IP BANNED!")
    with pytest.raises(ShadowBanError):
        custom.download_jobs([job(5), job(6)])


def test_consecutive_failures_halt_the_run(tmp_path, fake_yt_dlp):
    for index in (1, 3, 4):
        fake_yt_dlp.behaviours[vid(index)] = RuntimeError("down")
    downloader = make_downloader(tmp_path, max_consecutive_failures=2)
    with pytest.raises(ConsecutiveFailuresError) as caught:
        downloader.download_jobs([job(1), job(2), job(3), job(4), job(5)], report_path=tmp_path / "r.json")
    assert caught.value.reason == "consecutive_failures" and caught.value.consecutive_failures == 2
    assert [e.status for e in caught.value.entries] == ["failed", "downloaded", "failed", "failed"]
    assert fake_yt_dlp.ids() == [vid(1), vid(2), vid(3), vid(4)]
    assert len(read_download_report(tmp_path / "r.json")) == 4


def test_invalid_and_skipped_jobs_neither_count_nor_reset_the_failure_streak(tmp_path, fake_yt_dlp):
    fake_yt_dlp.default = RuntimeError("down")
    bad = DownloadJob(dataset="X", source_id="bad", youtube_id="bad")
    with pytest.raises(ConsecutiveFailuresError):
        make_downloader(tmp_path, max_consecutive_failures=2).download_jobs([job(1), bad, job(2), job(3)])
    entries = make_downloader(tmp_path / "off").download_jobs([job(1), job(2), job(3)])  # no limit configured
    assert [e.status for e in entries] == ["failed"] * 3


# ------------------------------------------------------------------------------ cookies, options, refresh hook

def options_of(fake, index: int) -> dict:
    return fake.calls_for(vid(index))[-1]["options"]


def test_cookie_and_yt_dlp_options_are_passed_through(tmp_path, fake_yt_dlp):
    plain = make_downloader(tmp_path / "a")
    plain.download_job(job(1))
    options = options_of(fake_yt_dlp, 1)
    assert "cookiefile" not in options and "cookiesfrombrowser" not in options
    assert options["format"] == "bestaudio/best" and options["quiet"] is True
    assert options["postprocessors"] == [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]
    assert options["outtmpl"].endswith(f"{vid(1)}.%(ext)s")

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    make_downloader(tmp_path / "b", cookies_file=cookies, cookies_from_browser="firefox").download_job(job(2))
    assert options_of(fake_yt_dlp, 2)["cookiefile"] == str(cookies)
    assert options_of(fake_yt_dlp, 2)["cookiesfrombrowser"] == ("firefox",)

    make_downloader(tmp_path / "c", cookies_from_browser=("chrome", "Profile 1")).download_job(job(3))
    assert options_of(fake_yt_dlp, 3)["cookiesfrombrowser"] == ("chrome", "Profile 1")

    extra = make_downloader(tmp_path / "d", ydl_options={"sleep_interval_requests": 1.25, "proxy": "socks5://x"},
                            verbose=True)
    extra.download_job(job(4))
    assert options_of(fake_yt_dlp, 4)["sleep_interval_requests"] == 1.25 and options_of(fake_yt_dlp, 4)["proxy"]
    assert options_of(fake_yt_dlp, 4)["quiet"] is False


def test_cookie_refresh_hook_retries_once_after_an_authentication_error(tmp_path, fake_yt_dlp):
    refreshed = tmp_path / "fresh_cookies.txt"
    refreshed.write_text("fresh", encoding="utf-8")
    seen = []

    def hook(failed_job, error):
        seen.append((failed_job.youtube_id, str(error)))
        downloader.cookies_file = refreshed  # e.g. re-exported from the browser; re-read on every attempt
        return True

    fake_yt_dlp.behaviours[vid(1)] = lambda calls: RuntimeError(AUTH.format(vid(1))) if len(calls) == 1 else \
        (tone(300.0, 1, 8.0, 32000), 32000)
    downloader = make_downloader(tmp_path, retries=0, cookie_refresh_hook=hook)  # the refresh attempt is extra
    entry = downloader.download_job(job(1))
    assert entry.status == "downloaded" and entry.retry_count == 1
    assert [call["options"].get("cookiefile") for call in fake_yt_dlp.calls_for(vid(1))] == [None, str(refreshed)]
    assert len(seen) == 1 and seen[0][0] == vid(1) and "Sign in to confirm" in seen[0][1]


def test_cookie_refresh_hook_is_called_once_per_job_and_only_for_authentication_errors(tmp_path, fake_yt_dlp):
    calls = []
    fake_yt_dlp.behaviours[vid(1)] = RuntimeError(AUTH.format(vid(1)))  # keeps failing after the refresh
    fake_yt_dlp.behaviours[vid(2)] = RuntimeError("HTTP Error 429")
    downloader = make_downloader(tmp_path, retries=3,
                                 cookie_refresh_hook=lambda j, e: calls.append(j.youtube_id) or True)
    still_failing = downloader.download_job(job(1))
    assert still_failing.status == "failed" and calls == [vid(1)]
    assert fake_yt_dlp.ids().count(vid(1)) == 5  # 1 + 1 (after the refresh) + 3 retries
    other = downloader.download_job(job(2))
    assert other.status == "failed" and calls == [vid(1)]  # not an authentication error: no hook


def test_cookie_refresh_hook_declining_or_failing_is_reported(tmp_path, fake_yt_dlp):
    fake_yt_dlp.default = RuntimeError(AUTH.format("x"))
    declined = make_downloader(tmp_path / "a", cookie_refresh_hook=lambda j, e: False).download_job(job(1))
    assert declined.status == "failed" and "Sign in to confirm" in declined.error_message
    assert fake_yt_dlp.ids() == [vid(1)]

    def boom(failed_job, error):
        raise OSError("browser not found")

    broken = make_downloader(tmp_path / "b", cookie_refresh_hook=boom).download_job(job(2))
    assert broken.status == "failed" and broken.error_message == "cookie_refresh_hook failed: browser not found"
    assert fake_yt_dlp.ids() == [vid(1), vid(2)]
    nohook = make_downloader(tmp_path / "c").download_job(job(3))
    assert nohook.status == "failed" and fake_yt_dlp.ids().count(vid(3)) == 1


def test_authentication_errors_with_a_straight_apostrophe_are_recognized_too(tmp_path, fake_yt_dlp):
    fake_yt_dlp.behaviours[vid(1)] = lambda calls: RuntimeError("Sign in to confirm you're not a bot") \
        if len(calls) == 1 else (tone(300.0, 1, 8.0, 32000), 32000)
    entry = make_downloader(tmp_path, cookie_refresh_hook=lambda j, e: True).download_job(job(1))
    assert entry.status == "downloaded"


def test_prerequisite_check_can_be_deferred_and_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr("audioset_tools.downloader.shutil.which", lambda name: None)
    pytest.importorskip("soundfile")
    pytest.importorskip("resampy")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.ModuleType("yt_dlp"))
    downloader = YouTubeDatasetDownloader(tmp_path / "x", check_prerequisites=False)
    with pytest.raises(DownloaderSetupError):
        downloader.check_prerequisites()
