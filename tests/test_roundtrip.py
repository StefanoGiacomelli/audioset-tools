"""Normalized JSON/CSV export and import (F-1: the CSV round trip)."""

from __future__ import annotations

import csv
import json

import pytest

from audioset_tools import AudioDataset, AudioRecord, TemporalEvent, detect_format, load_dataset
from audioset_tools.models import NORMALIZED, NORMALIZED_CSV_FIELDS


@pytest.fixture(params=["audioset", "strong", "vgg"])
def dataset(request, audioset, strong, vgg):
    return {"audioset": audioset, "strong": strong, "vgg": vgg}[request.param]


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_round_trip_is_lossless_for_every_source(dataset, tmp_path, suffix):
    path = tmp_path / f"normalized{suffix}"
    (dataset.to_json if suffix == ".json" else dataset.to_csv)(path)
    assert detect_format(path) == NORMALIZED
    loaded = load_dataset(path)
    assert loaded.source_format == NORMALIZED
    assert loaded.records == dataset.records  # every field: labels, events, raw, split, source_path
    assert [record.record_id for record in loaded] == [record.record_id for record in dataset]


def test_csv_round_trip_keeps_types(strong, tmp_path):
    path = tmp_path / "strong.csv"
    strong.to_csv(path)
    loaded = load_dataset(path)
    record = loaded[0]
    assert isinstance(record.start_seconds, float) and record.start_seconds == 30.0
    assert isinstance(record.original_label_ids, tuple) and record.original_label_ids[0] == "/m/04rlf"
    assert isinstance(record.events[1], TemporalEvent)
    assert record.events[1].start_seconds == 2.627 and record.events[1].original_label == "Cheering"
    assert isinstance(record.events[1].salt_labels, tuple)
    assert record.events[1].raw["label"] == "/m/053hz1"
    assert record.raw == {"segment_id": "s9d-2nhuJCQ_30000", "row_count": 6}


def test_csv_header_has_the_documented_columns_and_the_identifier(vgg, tmp_path):
    path = tmp_path / "vgg.csv"
    vgg.to_csv(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert list(rows[0]) == list(NORMALIZED_CSV_FIELDS)
    assert rows[0]["record_id"] == vgg[0].record_id
    assert json.loads(rows[0]["original_labels"]) == ["people marching"]


def test_json_export_carries_the_record_id_and_ignores_it_on_load(vgg, tmp_path):
    path = tmp_path / "vgg.json"
    vgg.to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["record_id"] == vgg[0].record_id
    data[0]["record_id"] = "tampered"
    data[0]["original_labels"] = ["edited label"]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_dataset(path)
    assert loaded[0].original_labels == ("edited label",)
    assert loaded[0].record_id != "tampered"  # recomputed from the content


def test_empty_dataset_round_trips(tmp_path):
    empty = AudioDataset(())
    for suffix in (".json", ".csv"):
        path = tmp_path / f"empty{suffix}"
        (empty.to_json if suffix == ".json" else empty.to_csv)(path)
        assert detect_format(path) == NORMALIZED
        assert len(load_dataset(path)) == 0


def test_none_values_and_missing_source_path_survive(tmp_path):
    record = AudioRecord(dataset="X", source_id="a", youtube_id="abcdefghijk")  # no times, split, labels, path
    for suffix in (".json", ".csv"):
        path = tmp_path / f"none{suffix}"
        dataset = AudioDataset((record,))
        (dataset.to_json if suffix == ".json" else dataset.to_csv)(path)
        loaded = load_dataset(path)
        assert loaded[0] == record
        assert loaded[0].source_path is None and loaded[0].start_seconds is None and loaded[0].split is None


def test_unicode_and_commas_in_labels_survive_csv(tmp_path):
    record = AudioRecord(dataset="X", source_id="a", youtube_id="abcdefghijk", start_seconds=1.5, end_seconds=11.5,
                         original_label_ids=("/m/1",), original_labels=('He said "hi", then left', "Café, naïve"),
                         salt_labels=("a,b",), split="train", raw={"note": "line1\nline2, with comma"})
    path = tmp_path / "unicode.csv"
    AudioDataset((record,)).to_csv(path)
    assert load_dataset(path)[0] == record


def test_legacy_nine_column_csv_still_loads(tmp_path):
    # The first v2 prototype wrote only these nine columns (and could not read them back).
    path = tmp_path / "legacy.csv"
    path.write_text("dataset,source_id,youtube_id,start_seconds,end_seconds,original_label_ids,original_labels,"
                    "salt_labels,split\n"
                    'VggSound,abcdefghijk_1,abcdefghijk,1.0,11.0,"[]","[""wind noise""]","[""wind""]",train\n',
                    encoding="utf-8")
    assert detect_format(path) == NORMALIZED
    record = load_dataset(path)[0]
    assert record.original_labels == ("wind noise",) and record.salt_labels == ("wind",)
    assert record.events == () and record.raw == {} and record.source_path == str(path)  # path of the file read


def test_invalid_normalized_files_raise_clear_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("dataset,source_id,youtube_id,original_label_ids\nX,a,abcdefghijk,not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset(path)
    missing = tmp_path / "missing_columns.csv"
    missing.write_text("dataset,source_id,original_label_ids\nX,a,[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported metadata format"):  # the header lacks youtube_id
        detect_format(missing)
    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(missing, dataset=NORMALIZED)
