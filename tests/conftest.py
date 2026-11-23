"""Shared fixtures: excerpt files in ``tests/data`` and a fake ``yt_dlp`` module (no network is ever used)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from audioset_tools import AudioDataset, SaltResolver

DATA = Path(__file__).resolve().parent / "data"
AUDIOSET_CSV = DATA / "audioset" / "eval_segments_excerpt.csv"
AUDIOSET_LABELS = DATA / "audioset" / "class_labels_indices_excerpt.csv"
STRONG_TSV = DATA / "audioset_strong" / "audioset_eval_strong_excerpt.tsv"
STRONG_LABELS = DATA / "audioset_strong" / "mid_to_display_name_excerpt.tsv"
VGGSOUND_CSV = DATA / "vggsound" / "vggsound_excerpt.csv"
SALT_MAPPING = DATA / "salt" / "salt_event_mapping.tsv"
SALT_ROOTS = DATA / "salt" / "salt_event_roots.json"


@pytest.fixture(scope="session")
def salt() -> SaltResolver:
    return SaltResolver.from_mapping_file(SALT_MAPPING, SALT_ROOTS)


@pytest.fixture(scope="session")
def audioset(salt) -> AudioDataset:
    return AudioDataset.from_file(AUDIOSET_CSV, labels_file=AUDIOSET_LABELS, salt_resolver=salt)


@pytest.fixture(scope="session")
def strong(salt) -> AudioDataset:
    return AudioDataset.from_file(STRONG_TSV, labels_file=STRONG_LABELS, salt_resolver=salt)


@pytest.fixture(scope="session")
def vgg(salt) -> AudioDataset:
    return AudioDataset.from_file(VGGSOUND_CSV, salt_resolver=salt)


class FakeYoutubeDL:
    """Stands in for ``yt_dlp.YoutubeDL``: writes a synthetic WAV (or raises) instead of contacting YouTube."""

    controller: FakeYtDlp

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def download(self, urls):
        import soundfile as sf

        controller = type(self).controller
        (url,) = urls
        youtube_id = url.rsplit("v=", 1)[-1]
        controller.calls.append({"youtube_id": youtube_id, "url": url, "options": dict(self.options)})
        behaviour = controller.behaviours.get(youtube_id, controller.default)
        if callable(behaviour):
            behaviour = behaviour(controller.calls_for(youtube_id))
        if isinstance(behaviour, Exception):
            raise behaviour
        if behaviour is None:  # yt-dlp "succeeds" but leaves no WAV file behind
            return
        data, sample_rate = behaviour
        sf.write(self.options["outtmpl"].replace("%(ext)s", "wav"), data, sample_rate)


class FakeYtDlp:
    """Controller of the fake ``yt_dlp`` module installed by the ``fake_yt_dlp`` fixture.

    ``behaviours[youtube_id]`` is a ``(samples, sample_rate)`` pair, an exception instance, ``None`` (no file is
    produced), or a callable taking the list of calls so far for that id (the current one included) and returning one
    of those. ``default`` applies to unlisted ids.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.behaviours: dict = {}
        self.default = (tone(1000.0, 1, 12.0, 48000), 48000)

    def calls_for(self, youtube_id):
        return [call for call in self.calls if call["youtube_id"] == youtube_id]

    def ids(self):
        return [call["youtube_id"] for call in self.calls]


def tone(freq: float, channels: int, seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wave = 0.4 * np.sin(2 * np.pi * freq * t)
    return wave if channels == 1 else np.stack([wave] * channels, axis=1)


@pytest.fixture
def fake_yt_dlp(monkeypatch, tmp_path):
    """Install a fake ``yt_dlp`` module and a fake ``ffmpeg`` so that real-download code paths run offline."""

    pytest.importorskip("soundfile")
    pytest.importorskip("resampy")
    controller = FakeYtDlp()
    FakeYoutubeDL.controller = controller
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    ffmpeg_dir = tmp_path / "fake_bin"
    ffmpeg_dir.mkdir()
    (ffmpeg_dir / "ffmpeg").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("audioset_tools.downloader.shutil.which",
                        lambda name: str(ffmpeg_dir / name) if name == "ffmpeg" else None)
    return controller
