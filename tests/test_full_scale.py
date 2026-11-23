"""Optional full-scale smoke test on the complete official metadata (skipped unless ``AUDIOSET_TOOLS_DATA`` is set).

``AUDIOSET_TOOLS_DATA`` must point to a folder with the layout of ``gpat_framework/datasets``::

    AudioSet_meta/{eval_segments,balanced_train_segments,unbalanced_train_segments}.csv, class_labels_indices.csv
    AudioSet_Strong_meta/{audioset_eval_strong,audioset_train_strong}.tsv, mid_to_display_name.tsv
    VGGsound_meta/vggsound.csv

The SALT coverage tests additionally need the SALT assets (``AUDIOSET_TOOLS_SALT_DIR`` or the default lookup of
``fetch-salt``). Run with ``AUDIOSET_TOOLS_DATA=/path/to/datasets pytest -m smoke -s`` to see the timings.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from audioset_tools import AudioDataset, SaltResolver, default_salt_dir

DATA_ROOT = os.environ.get("AUDIOSET_TOOLS_DATA")

pytestmark = [pytest.mark.smoke,
              pytest.mark.skipif(not DATA_ROOT, reason="AUDIOSET_TOOLS_DATA is not set (full-scale smoke test)")]

# name -> (metadata file, label file, records, event rows or None)
SOURCES = {
    "audioset_eval": ("AudioSet_meta/eval_segments.csv", "AudioSet_meta/class_labels_indices.csv", 20371, None),
    "audioset_balanced": ("AudioSet_meta/balanced_train_segments.csv", "AudioSet_meta/class_labels_indices.csv",
                          22160, None),
    "audioset_unbalanced": ("AudioSet_meta/unbalanced_train_segments.csv", "AudioSet_meta/class_labels_indices.csv",
                            2041789, None),
    "strong_eval": ("AudioSet_Strong_meta/audioset_eval_strong.tsv", "AudioSet_Strong_meta/mid_to_display_name.tsv",
                    16996, 139538),
    "strong_train": ("AudioSet_Strong_meta/audioset_train_strong.tsv", "AudioSet_Strong_meta/mid_to_display_name.tsv",
                     103463, 934821),
    "vggsound": ("VGGsound_meta/vggsound.csv", None, 199467, None),
}
# name -> records with at least one SALT label (the SALT mapping at the pinned upstream commit)
SALT_COVERAGE = {"audioset_eval": 20367, "strong_eval": 16996, "vggsound": 198775}
_loaded: dict[str, AudioDataset] = {}


def load(name: str, salt: SaltResolver | None = None) -> tuple[AudioDataset, float]:
    path, labels, _, _ = SOURCES[name]
    root = Path(DATA_ROOT)
    started = time.perf_counter()
    dataset = AudioDataset.from_file(root / path, labels_file=root / labels if labels else None, salt_resolver=salt)

    return dataset, time.perf_counter() - started


@pytest.mark.parametrize("name", list(SOURCES))
def test_dataset_sizes(name):
    dataset, seconds = load(name)
    _, _, records, events = SOURCES[name]
    print(f"\n[smoke] {name}: {len(dataset)} records in {seconds:.1f} s", end="")
    assert len(dataset) == records
    if events is not None:
        assert sum(len(record.events) for record in dataset) == events
        print(f", {events} event rows", end="")
    _loaded[name] = dataset


def _salt_resolver() -> SaltResolver:
    directory = default_salt_dir()
    if not (directory / "salt_event_mapping.tsv").exists():
        pytest.skip(f"SALT assets not found in {directory} (audioset-tools fetch-salt, or AUDIOSET_TOOLS_SALT_DIR)")

    return SaltResolver.from_mapping_file()


@pytest.mark.parametrize("name", list(SALT_COVERAGE))
def test_salt_coverage(name):
    dataset, seconds = load(name, _salt_resolver())
    covered = sum(1 for record in dataset if record.salt_labels)
    print(f"\n[smoke] {name}: {covered}/{len(dataset)} records with >= 1 SALT label "
          f"({100 * covered / len(dataset):.2f} %) in {seconds:.1f} s", end="")
    assert covered == SALT_COVERAGE[name]


def test_operations_on_full_metadata():
    dataset, _ = load("audioset_eval")
    started = time.perf_counter()
    speech = dataset.filter_labels(["Speech"], label_space="original")
    balanced = dataset.filter_labels(["Speech", "Music", "Siren"], label_space="original").rebalance(
        focus_labels=["Speech", "Music", "Siren"], random_state=0)
    again = dataset.filter_labels(["Speech", "Music", "Siren"], label_space="original").rebalance(
        focus_labels=["Speech", "Music", "Siren"], random_state=0)
    print(f"\n[smoke] filter + rebalance on the eval split: {len(speech)} Speech clips, {len(balanced)} rebalanced "
          f"in {time.perf_counter() - started:.2f} s", end="")
    assert balanced.records == again.records and len(speech) > 0
