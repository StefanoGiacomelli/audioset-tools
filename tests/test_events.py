"""Temporal events: tracks, piano roll (F-3 colormap API) and the shared dataset helper (F-4)."""

from __future__ import annotations

import numpy as np
import pytest

from audioset_tools import (
    AudioDataset,
    events,
    filters,
    generate_event_tracks,
    iter_events,
    plot_events_pianoroll,
    utils,
)
from audioset_tools import dataset as dataset_module

from .conftest import STRONG_LABELS, STRONG_TSV


def test_event_tracks_have_the_documented_shape(strong):
    tracks = generate_event_tracks(strong.select_range(0, 1), bin_size=0.5)
    assert tracks
    assert all(track.shape == (20,) for _, track in tracks)  # 10 s / 0.5 s
    assert {track.shape for _, track in generate_event_tracks(strong, bin_size=0.3)} == {(33,)}  # int(10 / 0.3)
    assert {track.shape for _, track in generate_event_tracks(strong, bin_size=0.1)} == {(100,)}


def test_event_track_values_follow_the_overlap_rule(strong):
    tracks = dict(generate_event_tracks(strong.select_range(0, 1), bin_size=0.5, label_space="original"))
    # Cheering: 2.627 -> 7.237 s: first bin floor(2.627 / 0.5) = 5, last bin ceil(7.237 / 0.5) - 1 = 14.
    cheering = tracks["s9d-2nhuJCQ_30000_Cheering"]
    assert cheering.tolist() == [0] * 5 + [1] * 10 + [0] * 5
    music = tracks["s9d-2nhuJCQ_30000_Music"]
    assert music.tolist() == [1] * 20  # 0 -> 10 s
    # Whistling occurs twice (5.634-6.649 and 8.089-9.230): the two events are merged into one track.
    whistling = tracks["s9d-2nhuJCQ_30000_Whistling"]
    assert whistling.tolist() == [0] * 11 + [1] * 3 + [0] * 2 + [1] * 3 + [0] * 1  # bins 11-13 and 16-18


def test_event_tracks_in_other_label_spaces_and_types(strong):
    by_id = dict(generate_event_tracks(strong.select_range(0, 1), bin_size=0.5, label_space="id"))
    assert "s9d-2nhuJCQ_30000_/m/04rlf" in by_id
    by_salt = dict(generate_event_tracks(strong.select_range(0, 1), bin_size=0.5, label_space="salt"))
    assert "s9d-2nhuJCQ_30000_music" in by_salt
    assert generate_event_tracks(strong, bin_size=1.0, out_type="np")[0][1].dtype.kind == "i"
    with pytest.raises(ValueError, match="bin_size"):
        generate_event_tracks(strong, bin_size=0)
    with pytest.raises(ValueError, match="out_type"):
        generate_event_tracks(strong, out_type="tf")
    with pytest.raises(ValueError, match="label_space"):
        generate_event_tracks(strong, label_space="auto")


def test_event_track_torch_output_if_torch_is_installed(strong):
    torch = pytest.importorskip("torch")
    key, track = generate_event_tracks(strong.select_range(0, 1), bin_size=0.5, out_type="pt")[0]
    assert isinstance(track, torch.Tensor) and tuple(track.shape) == (20,) and key.startswith("s9d-2nhuJCQ_30000_")


def test_iter_events_yields_record_event_label(strong):
    rows = list(iter_events(strong, label_space="original"))
    assert len(rows) == 43
    record, event, label = rows[0]
    assert record.source_id == "s9d-2nhuJCQ_30000" and event.original_label == label == "Music"
    assert next(iter(iter_events(strong, label_space="salt")))[2] == ("music",)
    assert next(iter(iter_events(STRONG_TSV, labels_file=STRONG_LABELS)))[2] == "Music"  # loads from a path


def test_records_without_events_yield_no_tracks(vgg):
    assert generate_event_tracks(vgg) == []


def test_piano_roll_uses_the_current_matplotlib_colormap_api(strong, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the deprecated matplotlib.cm.get_cmap would raise here
        figures = plot_events_pianoroll(strong, segment_ids="s9d-2nhuJCQ_30000", save_plots=True, output_dir=tmp_path)
    assert len(figures) == 1
    assert (tmp_path / "s9d-2nhuJCQ_30000_pianoroll.png").stat().st_size > 1000
    axis = figures[0].axes[0]
    assert [t.get_text() for t in axis.get_yticklabels()] == sorted({e.original_label for e in strong[0].events})
    import matplotlib.pyplot as plt

    plt.close("all")


def test_piano_roll_all_segments_and_label_spaces(strong):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert len(plot_events_pianoroll(strong)) == len(strong)
    assert len(plot_events_pianoroll(strong, segment_ids=["s9d-2nhuJCQ_30000", "unknown"], label_space="salt")) == 1
    assert plot_events_pianoroll(AudioDataset(())) == []
    plt.close("all")


def test_ensure_dataset_has_a_single_definition():
    # F-4: filters, utils and events used to carry three copies.
    assert filters._ensure_dataset is dataset_module._ensure_dataset
    assert utils._ensure_dataset is dataset_module._ensure_dataset
    assert events._ensure_dataset is dataset_module._ensure_dataset
    assert events.ensure_dataset is dataset_module._ensure_dataset  # backward-compatible public alias
    dataset = AudioDataset(())
    assert dataset_module._ensure_dataset(dataset) is dataset
    assert len(dataset_module._ensure_dataset(STRONG_TSV, labels_file=STRONG_LABELS)) == 7


def test_numpy_tracks_are_independent_arrays(strong):
    tracks = generate_event_tracks(strong.select_range(0, 1), bin_size=0.5)
    first = tracks[0][1]
    first[:] = 0
    assert any(track.any() for _, track in generate_event_tracks(strong.select_range(0, 1), bin_size=0.5))
    assert isinstance(first, np.ndarray)
