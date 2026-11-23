"""Format detection and parsers (the six tests of the in-tree prototype, ported to the shipped excerpts)."""

from __future__ import annotations

import pytest

from audioset_tools import (
    AUDIOSET,
    AUDIOSET_STRONG,
    VGGSOUND,
    AudioDataset,
    detect_format,
    load_label_map,
    parse_metadata,
)
from audioset_tools.models import NORMALIZED

from .conftest import (
    AUDIOSET_CSV,
    AUDIOSET_LABELS,
    STRONG_LABELS,
    STRONG_TSV,
    VGGSOUND_CSV,
)


def test_detects_supported_metadata_formats():
    assert detect_format(AUDIOSET_CSV) == AUDIOSET
    assert detect_format(STRONG_TSV) == AUDIOSET_STRONG
    assert detect_format(VGGSOUND_CSV) == VGGSOUND


def test_detect_format_rejects_unknown_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_format(tmp_path / "missing.csv")
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported metadata format"):
        detect_format(unknown)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="normalized records"):
        detect_format(bad_json)


def test_label_maps_are_loaded_from_both_layouts():
    audioset_map = load_label_map(AUDIOSET_LABELS)
    assert audioset_map["/m/09x0r"] == "Speech"
    assert audioset_map["/m/02zsn"] == "Female speech, woman speaking"
    strong_map = load_label_map(STRONG_LABELS)
    assert strong_map["/m/04rlf"] == "Music"
    assert load_label_map(None) == {}


def test_audioset_parser_preserves_multiple_labels_and_resolves_salt(audioset):
    assert len(audioset) == 30
    first = audioset[0]
    assert first.dataset == AUDIOSET
    assert first.youtube_id == "--4gqARaEJE"
    assert first.source_id == "--4gqARaEJE_0_10"
    assert (first.start_seconds, first.end_seconds) == (0.0, 10.0)
    assert first.original_label_ids == ("/m/068hy", "/m/07q6cd_", "/m/0bt9lr", "/m/0jbk")
    assert "Domestic animals, pets" in first.original_labels
    assert first.salt_labels == ("animal", "dog")
    assert first.raw["positive_labels"] == "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
    assert first.source_path == str(AUDIOSET_CSV)


def test_audioset_without_label_map_keeps_ids_as_labels():
    dataset = AudioDataset.from_file(AUDIOSET_CSV)
    assert dataset[0].original_labels == dataset[0].original_label_ids
    assert dataset[0].salt_labels == ()


def test_strong_parser_groups_events_by_segment_id(strong):
    assert len(strong) == 7
    first = strong[0]
    assert first.dataset == AUDIOSET_STRONG
    assert first.source_id == "s9d-2nhuJCQ_30000"
    assert first.youtube_id == "s9d-2nhuJCQ"
    assert first.start_seconds == 30.0
    assert first.end_seconds == 40.0
    assert len(first.events) == 6
    assert first.original_label_ids == ("/m/04rlf", "/m/053hz1", "/m/03qtwd", "/m/01w250", "/m/0l15bq")
    assert first.events[1].original_label == "Cheering"
    assert first.events[0].salt_labels == ("music",)
    assert sum(len(record.events) for record in strong) == 43


def test_strong_segment_ids_with_underscores_keep_the_full_youtube_id(strong):
    # rpartition('_'): YouTube ids may contain '_' (v1 split on the first one and mis-parsed ~14.5 % of the ids).
    record = next(record for record in strong if record.youtube_id == "rxhnnJ_hJyk")
    assert record.source_id == "rxhnnJ_hJyk_30000"
    assert record.start_seconds == 30.0


def test_vggsound_parser_and_salt_filtering(vgg):
    assert len(vgg) == 40
    first = vgg[0]
    assert first.dataset == VGGSOUND
    assert first.youtube_id == "---g-f_I2yQ"
    assert first.start_seconds == 1.0
    assert first.end_seconds == 11.0
    assert first.split == "test"
    assert first.original_labels == ("people marching",)
    filtered = vgg.filter_labels(["vehicle"], label_space="salt", include_descendants=True)
    assert len(filtered) == 5
    assert {record.original_labels[0] for record in filtered} == {"ambulance siren", "car engine starting", "airplane"}


def test_curated_v1_files_with_a_downloaded_column_parse_correctly(tmp_path):
    # The published v1 files (EV_*_downloaded.csv) carry a trailing True/False flag column: it must not become a label.
    path = tmp_path / "ev_downloaded.csv"
    path.write_text("yt_id,start_seconds,end_seconds,positive_labels,downloaded\n"
                    "-9GUUhB3QV0, 30.000, 40.000,\"['/m/03j1ly', '/m/03kmc9']\",False\n"
                    "-RBs9pPhHY8, 30.000, 40.000,\"['/m/012ndj']\",True\n", encoding="utf-8")
    dataset = AudioDataset.from_file(path)
    assert dataset.source_format == AUDIOSET
    assert [record.original_label_ids for record in dataset] == [("/m/03j1ly", "/m/03kmc9"), ("/m/012ndj",)]
    assert [record.raw["downloaded"] for record in dataset] == ["False", "True"]
    assert set(dataset.label_counts("id")) == {"/m/03j1ly", "/m/03kmc9", "/m/012ndj"}


def test_curated_v1_files_without_the_flag_and_raw_files_are_unchanged(tmp_path):
    path = tmp_path / "ev.csv"
    path.write_text("yt_id,start_seconds,end_seconds,positive_labels\n"
                    "-4RWqM0UCCY, 0.000, 10.000,\"['/m/012n7d', '/m/02mfyn']\"\n", encoding="utf-8")
    dataset = AudioDataset.from_file(path)
    assert dataset[0].original_label_ids == ("/m/012n7d", "/m/02mfyn")
    assert "downloaded" not in dataset[0].raw


def test_parse_metadata_returns_records_and_accepts_an_explicit_dataset(salt):
    records = parse_metadata(AUDIOSET_CSV, labels_file=AUDIOSET_LABELS, salt_resolver=salt, dataset=AUDIOSET)
    assert len(records) == 30
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        parse_metadata(AUDIOSET_CSV, dataset="Nope")


def test_strong_files_with_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "broken.tsv"
    path.write_text("segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n", encoding="utf-8")
    assert detect_format(path) == AUDIOSET_STRONG
    path.write_text("segment_id\tstart_time_seconds\tend_time_seconds\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported metadata format"):
        detect_format(path)
    forced = tmp_path / "forced.tsv"
    forced.write_text("segment_id\tstart\tend\tlabel\nabc_0\t0\t1\t/m/x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        parse_metadata(forced, dataset=AUDIOSET_STRONG)


def test_normalized_constant_is_exported():
    assert NORMALIZED == "Normalized"
