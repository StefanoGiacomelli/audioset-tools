"""Temporal-event utilities for normalized AudioSet Strong records."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .dataset import _ensure_dataset

#: Backward-compatible public alias of the shared helper (the first v2 prototype defined it here).
ensure_dataset = _ensure_dataset


def iter_events(dataset_or_path,
                labels_file=None,
                salt_resolver=None,
                label_space: str = "original"):
    """Yield ``(record, event, label)`` for every temporal event of records that contain strong labels.

    ``label`` is the event's MID (``"id"``), display label (``"original"``) or tuple of SALT labels (``"salt"``).
    """

    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)
    for record in dataset:
        for event in record.events:
            if label_space == "id":
                label = event.original_label_id
            elif label_space == "salt":
                label = event.salt_labels
            elif label_space == "original":
                label = event.original_label
            else:
                raise ValueError("label_space must be one of 'id', 'original', or 'salt'.")
            yield record, event, label


def generate_event_tracks(dataset_or_path,
                          labels_file=None,
                          salt_resolver=None,
                          bin_size: float = 0.1,
                          label_space: str = "original",
                          out_type: str = "np"):
    """Generate binary event tracks for strong labels.

    Returns a list of ``(record_id_label, track)`` tuples where the key is ``<source_id>_<label>``. Every track has
    ``int(10 / bin_size)`` bins covering the fixed 10-s clip (so 20 bins for ``bin_size=0.5``, 33 for ``0.3``);
    a bin is active when the event overlaps it (onset floored, offset ceiled). The implementation mirrors the
    AudioSet Strong utility of AudioSet-Tools v1 but operates on normalized records and supports SALT labels.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    length = int(10 / bin_size)
    tracks = {}
    for record, event, label_value in iter_events(dataset_or_path,
                                                  labels_file=labels_file,
                                                  salt_resolver=salt_resolver,
                                                  label_space=label_space):
        labels = label_value if isinstance(label_value, tuple) else (label_value,)
        for label in labels:
            if not label:
                continue
            start = max(0.0, min(event.start_seconds or 0.0, 10.0))
            end = max(0.0, min(event.end_seconds or 0.0, 10.0))
            start_idx = int(start / bin_size)
            end_idx = min(int(math.ceil(end / bin_size)), length)
            key = f"{record.source_id}_{label}"
            event_track = np.zeros(length, dtype=int)
            event_track[start_idx:end_idx] = 1
            tracks[key] = np.maximum(tracks.get(key, event_track), event_track)

    if out_type == "np":
        return list(tracks.items())
    if out_type == "pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Torch is required for out_type='pt' (pip install 'audioset-tools[torch]').") from exc
        return [(key, torch.from_numpy(track)) for key, track in tracks.items()]
    raise ValueError("out_type must be 'np' or 'pt'.")


def plot_events_pianoroll(dataset_or_path,
                          segment_ids: str | Iterable[str] | None = None,
                          labels_file=None,
                          salt_resolver=None,
                          label_space: str = "original",
                          save_plots: bool = False,
                          output_dir: str | Path | None = None,
                          verbose: bool = False):
    """Create piano-roll plots for strong-label records (needs the ``plot`` extra: matplotlib >= 3.6)."""

    try:
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for piano-roll plots "
                          "(pip install 'audioset-tools[plot]').") from exc

    dataset = _ensure_dataset(dataset_or_path, labels_file, salt_resolver)
    if isinstance(segment_ids, str):
        wanted = {segment_ids}
    elif segment_ids is None:
        wanted = {record.source_id for record in dataset if record.events}
    else:
        wanted = set(segment_ids)

    if save_plots:
        out_dir = Path(output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    figures = []
    for record in dataset:
        if record.source_id not in wanted or not record.events:
            continue
        events_by_label = defaultdict(list)
        for event in record.events:
            if label_space == "id":
                labels = (event.original_label_id,)
            elif label_space == "salt":
                labels = event.salt_labels
            else:
                labels = (event.original_label,)
            for label in labels:
                if label:
                    events_by_label[label].append(event)

        if not events_by_label:
            if verbose:
                print(f"No events found for {record.source_id}.")
            continue

        unique_labels = sorted(events_by_label)
        label_to_y = {label: idx for idx, label in enumerate(unique_labels)}
        cmap = matplotlib.colormaps["tab20"].resampled(len(unique_labels))
        label_to_color = {label: cmap(index) for index, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(10, 2 + len(unique_labels)))
        for label, events in events_by_label.items():
            y = label_to_y[label]
            for event in events:
                start = event.start_seconds or 0.0
                end = event.end_seconds or start
                ax.broken_barh([(start, max(0.0, end - start))],
                               (y - 0.4, 0.8),
                               facecolors=label_to_color[label])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Labels")
        ax.set_yticks(list(label_to_y.values()))
        ax.set_yticklabels(list(label_to_y.keys()))
        ax.set_title(f"Event Piano Roll for {record.source_id}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 10)

        legend_patches = [mpatches.Patch(color=label_to_color[label], label=label) for label in unique_labels]
        ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.15, 1))
        figures.append(fig)

        if save_plots and out_dir is not None:
            fig.savefig(out_dir / f"{record.source_id}_pianoroll.png", bbox_inches="tight")

    return figures
