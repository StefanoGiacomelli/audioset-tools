"""Utility functions for normalized AudioSet-Tools datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .dataset import AudioDataset, _ensure_dataset, merge_datasets
from .salt import SaltResolver


def compute_stats(dataset_or_path,
                  labels_file: str | Path | None = None,
                  salt_resolver: SaltResolver | None = None) -> dict:
    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)

    return dataset.summary()


def find_samples_by_youtube_ids(dataset_or_path,
                                youtube_ids: Iterable[str],
                                labels_file: str | Path | None = None,
                                salt_resolver: SaltResolver | None = None) -> AudioDataset:
    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)

    return dataset.find_by_youtube_ids(youtube_ids)


def find_samples_by_samples(targets,
                            dataset_or_path,
                            labels_file: str | Path | None = None,
                            salt_resolver: SaltResolver | None = None) -> AudioDataset:
    """Find samples whose YouTube IDs appear in another dataset or iterable."""

    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)
    if isinstance(targets, AudioDataset):
        youtube_ids = [record.youtube_id for record in targets]
    elif isinstance(targets, (str, Path)):
        youtube_ids = [record.youtube_id for record in AudioDataset.from_file(targets)]
    else:
        youtube_ids = list(targets)

    return dataset.find_by_youtube_ids(youtube_ids)


def merge_files(metadata_files: Sequence[str | Path],
                labels_file: str | Path | None = None,
                salt_resolver: SaltResolver | None = None,
                deduplicate: bool = True) -> AudioDataset:
    datasets = [AudioDataset.from_file(metadata_file,
                                       labels_file=labels_file,
                                       salt_resolver=salt_resolver) for metadata_file in metadata_files]

    return merge_datasets(*datasets, deduplicate=deduplicate)
