"""Functional filtering helpers built on the normalized AudioDataset API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .dataset import AudioDataset, _ensure_dataset
from .salt import SaltResolver


def filter_labels(dataset_or_path,
                  labels: Sequence[str],
                  labels_file: str | Path | None = None,
                  salt_resolver: SaltResolver | None = None,
                  label_space: str = "auto",
                  match: str = "any",
                  include_descendants: bool = False) -> AudioDataset:
    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)

    return dataset.filter_labels(labels,
                                 label_space=label_space,
                                 match=match,
                                 include_descendants=include_descendants)


def blacklist_labels(dataset_or_path,
                     labels: Sequence[str],
                     labels_file: str | Path | None = None,
                     salt_resolver: SaltResolver | None = None,
                     label_space: str = "auto",
                     include_descendants: bool = False) -> AudioDataset:

    return filter_labels(dataset_or_path,
                         labels,
                         labels_file=labels_file,
                         salt_resolver=salt_resolver,
                         label_space=label_space,
                         match="none",
                         include_descendants=include_descendants)


def select_range(dataset_or_path, start_idx: int, end_idx: int, **load_kwargs) -> AudioDataset:
    dataset = _ensure_dataset(dataset_or_path, **load_kwargs)

    return dataset.select_range(start_idx, end_idx)


def rebalance(dataset_or_path,
              focus_labels: Sequence[str] | None = None,
              label_space: str = "original",
              random_state: int | None = None,
              **load_kwargs) -> AudioDataset:
    dataset = _ensure_dataset(dataset_or_path, **load_kwargs)

    return dataset.rebalance(focus_labels=focus_labels,
                             label_space=label_space,
                             random_state=random_state)
