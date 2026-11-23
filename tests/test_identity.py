"""AudioRecord.record_id: the stable hash identifier (additive property)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from audioset_tools import AudioRecord, TemporalEvent
from audioset_tools.dataset import _record_key

RECORD = AudioRecord(dataset="AudioSet",
                     source_id="--4gqARaEJE_0_10",
                     youtube_id="--4gqARaEJE",
                     start_seconds=0.0,
                     end_seconds=10.0,
                     original_label_ids=("/m/068hy", "/m/0jbk"),
                     original_labels=("Domestic animals, pets", "Animal"))

#: Golden value of the definition in the AudioRecord.record_id docstring. If this test fails the identifier changed:
#: that is a breaking change of the published definition (and of every stored identifier).
GOLDEN = "dea74d57eb2c3dc144e69949ba60f3895ea5cede"


def test_record_id_follows_the_documented_definition():
    payload = ["AudioSet", "--4gqARaEJE_0_10", "--4gqARaEJE", 0.0, 10.0, ["/m/068hy", "/m/0jbk"],
               ["Domestic animals, pets", "Animal"]]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert canonical == ('["AudioSet","--4gqARaEJE_0_10","--4gqARaEJE",0.0,10.0,["/m/068hy","/m/0jbk"],'
                         '["Domestic animals, pets","Animal"]]')
    expected = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    assert RECORD.record_id == expected
    assert len(RECORD.record_id) == 40 and RECORD.record_id == RECORD.record_id.lower()


def test_record_id_golden_value():
    assert RECORD.record_id == GOLDEN


def test_record_id_ignores_non_identity_fields():
    other = replace(RECORD,
                    salt_labels=("animal",),
                    split="eval",
                    events=(TemporalEvent(0.0, 1.0, "/m/0jbk", "Animal"),),
                    raw={"anything": 1},
                    source_path="/somewhere/else.csv")
    assert other.record_id == RECORD.record_id


def test_record_id_changes_with_every_identity_field():
    variants = [replace(RECORD, dataset="VggSound"),
                replace(RECORD, source_id="other"),
                replace(RECORD, youtube_id="abcdefghijk"),
                replace(RECORD, start_seconds=1.0),
                replace(RECORD, end_seconds=11.0),
                replace(RECORD, original_label_ids=("/m/068hy",)),
                replace(RECORD, original_labels=("Animal", "Domestic animals, pets")),  # order matters
                replace(RECORD, start_seconds=None)]
    ids = {RECORD.record_id} | {variant.record_id for variant in variants}
    assert len(ids) == len(variants) + 1


def test_record_id_is_independent_of_int_float_and_negative_zero():
    assert replace(RECORD, start_seconds=0).record_id == RECORD.record_id
    assert replace(RECORD, start_seconds=-0.0).record_id == RECORD.record_id
    assert replace(RECORD, end_seconds=10).record_id == RECORD.record_id


def test_record_id_equality_matches_the_deduplication_key(audioset, strong, vgg):
    records = list(audioset) + list(strong) + list(vgg)
    by_id = {}
    for record in records:
        by_id.setdefault(record.record_id, []).append(record)
    assert len(by_id) == len(records)  # all distinct in the excerpts
    for record in records:
        twin = replace(record, salt_labels=(), split="x", raw={})
        assert twin.record_id == record.record_id and _record_key(twin) == _record_key(record)


def test_record_id_is_stable_across_processes(tmp_path):
    # PYTHONHASHSEED only affects hash(); the identifier must not depend on it.
    import os
    import subprocess
    import sys

    code = ("from audioset_tools import AudioRecord;"
            "r=AudioRecord('AudioSet','--4gqARaEJE_0_10','--4gqARaEJE',0.0,10.0,('/m/068hy','/m/0jbk'),"
            "('Domestic animals, pets','Animal'));print(r.record_id)")
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip()
               for seed in ("0", "1", "12345")}
    assert outputs == {RECORD.record_id}
