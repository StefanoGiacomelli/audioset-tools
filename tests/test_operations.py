"""AudioDataset operations: filtering and query spaces, ranges, grouping, lookup, merge, rebalance, statistics."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from audioset_tools import (
    AudioDataset,
    AudioRecord,
    SaltResolver,
    blacklist_labels,
    compute_stats,
    filter_labels,
    find_samples_by_samples,
    find_samples_by_youtube_ids,
    merge_datasets,
    merge_files,
    rebalance,
    select_range,
)

from .conftest import AUDIOSET_CSV, AUDIOSET_LABELS, SALT_MAPPING, SALT_ROOTS, VGGSOUND_CSV


def synthetic(counts_by_labelset: dict[tuple[str, ...], int]) -> AudioDataset:
    """Records whose original labels are the given label sets (one record per count)."""

    records = []
    for labels, count in counts_by_labelset.items():
        for index in range(count):
            name = "+".join(labels)
            records.append(AudioRecord(dataset="X", source_id=f"{name}_{index}", youtube_id=f"{name}_{index}"[:11],
                                       original_labels=labels))
    return AudioDataset(tuple(records))


# ---------------------------------------------------------------- filtering and query spaces

def test_filter_original_id_and_salt_spaces(audioset):
    by_name = audioset.filter_labels(["Speech"], label_space="original")
    by_mid = audioset.filter_labels(["/m/09x0r"], label_space="id")
    by_salt = audioset.filter_labels(["speech"], label_space="salt")
    assert len(by_name) == len(by_mid) == 10
    assert [r.source_id for r in by_name] == [r.source_id for r in by_mid]
    assert {r.source_id for r in by_salt} >= {r.source_id for r in by_name}  # SALT "speech" also covers other labels
    hammer = audioset.filter_labels(["Hammer"], label_space="original")
    assert audioset.filter_labels(["Hammer"]).records == hammer.records  # "auto" finds the same name


def test_matching_modes_any_all_none(audioset):
    any_ = audioset.filter_labels(["Music", "Speech"], label_space="original", match="any")
    all_ = audioset.filter_labels(["Music", "Speech"], label_space="original", match="all")
    none = audioset.filter_labels(["Music", "Speech"], label_space="original", match="none")
    assert len(all_) > 0
    assert len(any_) == len(audioset) - len(none)
    assert len(any_) == 11 + 10 - len(all_)
    assert all(set(r.original_labels) >= {"Music", "Speech"} for r in all_)
    assert not any({"Music", "Speech"} & set(r.original_labels) for r in none)
    assert audioset.filter_labels([], label_space="original").records == ()


def test_matching_is_case_and_whitespace_insensitive_unless_asked(audioset):
    assert len(audioset.filter_labels(["  SPEECH "], label_space="original")) == 10
    assert len(audioset.filter_labels(["SPEECH"], label_space="original", case_sensitive=True)) == 0
    assert len(audioset.filter_labels(["Domestic   animals,  pets"], label_space="original")) == 2


def test_invalid_arguments_are_rejected(audioset):
    with pytest.raises(ValueError, match="match must be"):
        audioset.filter_labels(["Speech"], match="some")
    with pytest.raises(ValueError, match="label_space must be"):
        audioset.filter_labels(["Speech"], label_space="wrong")


def test_salt_descendant_expansion_on_the_synthetic_roots(vgg, salt):
    assert salt.descendants("vehicle") == ("road_vehicle", "rail_vehicle", "aircraft", "siren_ringing", "car",
                                           "engine", "train", "ambulance_siren", "idling")
    assert salt.descendants("unknown") == ()
    assert salt.expand_salt_labels(["siren_ringing"], include_descendants=True) == ("siren_ringing", "ambulance_siren")
    with_descendants = vgg.filter_labels(["vehicle"], label_space="salt", include_descendants=True)
    without = vgg.filter_labels(["vehicle"], label_space="salt")
    assert len(with_descendants) == 5 and len(without) == 0  # no VggSound label maps to "vehicle" itself
    only_engine = vgg.filter_labels(["engine"], label_space="salt", include_descendants=True)
    assert [r.original_labels for r in only_engine] == [("car engine starting",)]


def test_descendants_are_ignored_without_a_resolver_or_outside_the_salt_space(vgg):
    bare = AudioDataset.from_file(VGGSOUND_CSV)  # no resolver: no salt labels at all
    assert len(bare.filter_labels(["vehicle"], label_space="salt", include_descendants=True)) == 0
    assert len(vgg.filter_labels(["people marching"], label_space="original", include_descendants=True)) == 1


def test_functional_helpers_match_the_methods(vgg, salt):
    args = dict(label_space="salt", include_descendants=True)
    assert filter_labels(vgg, ["vehicle"], **args).records == vgg.filter_labels(["vehicle"], **args).records
    from_path = filter_labels(VGGSOUND_CSV, ["vehicle"], salt_resolver=salt, **args)
    assert from_path.records == vgg.filter_labels(["vehicle"], **args).records
    kept = filter_labels(vgg, ["vehicle"], **args)
    rest = blacklist_labels(vgg, ["vehicle"], **args)
    assert len(kept) + len(rest) == len(vgg)
    assert not {r.source_id for r in kept} & {r.source_id for r in rest}
    assert select_range(vgg, 2, 5).records == vgg.records[2:5]


# ---------------------------------------------------------------- ranges, grouping, lookup, statistics

def test_select_range_is_zero_based_half_open(vgg):
    assert [r.source_id for r in vgg.select_range(0, 3)] == [r.source_id for r in vgg.records[:3]]
    assert len(vgg.select_range(5, 5)) == 0
    assert vgg.select_range(1, 4).records == vgg.records[1:4]  # v1 select_by_samp_idx(2, 5): header counted as row 0


def test_group_by_and_label_counts(vgg, strong):
    groups = vgg.group_by("split")
    assert sum(len(group) for group in groups.values()) == len(vgg)
    assert set(groups) == {r.split for r in vgg}
    assert Counter(r.dataset for r in strong.group_by("dataset")[strong[0].dataset]) == {strong[0].dataset: 7}
    assert vgg.label_counts("original")["cricket chirping"] == 3
    assert vgg.group_by("youtube_id")["--CC5pH97q4"].records == tuple(r for r in vgg if r.youtube_id == "--CC5pH97q4")


def test_find_samples_by_youtube_ids_and_by_samples(vgg):
    found = find_samples_by_youtube_ids(vgg, ["--CC5pH97q4"])
    assert len(found) == 2 and {r.start_seconds for r in found} == {4.0, 33.0}  # two segments of one video
    source = vgg.select_range(0, 5)
    assert find_samples_by_samples(source, vgg).records == tuple(r for r in vgg if r.youtube_id in
                                                                 {s.youtube_id for s in source})
    assert find_samples_by_samples(["---g-f_I2yQ"], vgg).records == vgg.select_range(0, 1).records
    assert len(find_samples_by_samples(VGGSOUND_CSV, vgg)) == len(vgg)  # a path as targets


def test_summary_and_compute_stats(audioset, vgg):
    summary = audioset.summary()
    assert summary["records"] == 30 and summary["datasets"] == {"AudioSet": 30}
    assert summary["unique_youtube_ids"] == 30 and summary["splits"] == {}
    assert summary["label_counts_original"]["Music"] == 11
    assert summary["label_counts_salt"]["music"] == 11
    assert compute_stats(vgg)["splits"] == dict(Counter(r.split for r in vgg))
    assert compute_stats(AUDIOSET_CSV, labels_file=AUDIOSET_LABELS)["label_counts_original"]["Speech"] == 10


# ---------------------------------------------------------------- merge and deduplicate

def test_merge_deduplicates_by_identity_and_keeps_first(vgg):
    merged = merge_datasets(vgg.select_range(0, 5), vgg.select_range(3, 8))
    assert [r.source_id for r in merged] == [r.source_id for r in vgg.records[:8]]
    assert len(merge_datasets(vgg.select_range(0, 5), vgg.select_range(3, 8), deduplicate=False)) == 10
    assert len(vgg.merge(vgg, vgg)) == len(vgg)
    assert len(merge_datasets()) == 0
    assert merged.salt_resolver is vgg.salt_resolver and merged.source_format == vgg.source_format


def test_deduplicate_uses_the_identity_fields_only(vgg):
    record = vgg[0]
    twin = AudioRecord(**{**record.__dict__, "split": "other", "raw": {}, "salt_labels": ()})
    assert len(AudioDataset((record, twin)).deduplicate()) == 1
    different = AudioRecord(**{**record.__dict__, "end_seconds": 99.0})
    assert len(AudioDataset((record, different)).deduplicate()) == 2


def test_merge_files_reads_and_merges_paths(salt):
    merged = merge_files([VGGSOUND_CSV, VGGSOUND_CSV], salt_resolver=salt)
    assert len(merged) == 40
    assert len(merge_files([VGGSOUND_CSV, VGGSOUND_CSV], deduplicate=False)) == 80
    assert merged.salt_resolver is salt


# ---------------------------------------------------------------- rebalance (F-7 semantics)

def test_rebalance_is_deterministic_for_a_seed_and_differs_across_seeds():
    data = synthetic({("A",): 12, ("B",): 9, ("C",): 15, ("A", "B"): 4, ("B", "C"): 3})
    one = data.rebalance(label_space="original", random_state=7)
    two = data.rebalance(label_space="original", random_state=7)
    other = data.rebalance(label_space="original", random_state=8)
    assert one.records == two.records  # same selection AND same order
    assert [r.source_id for r in one] != [r.source_id for r in other]
    assert len({r.source_id for r in one}) == len(one)


def test_rebalance_result_does_not_depend_on_python_hash_seed():
    import os
    import subprocess
    import sys

    code = ("import random;from audioset_tools import AudioDataset, AudioRecord;"
            "recs=[AudioRecord('X',f's{i}',f'y{i}',original_labels=tuple(sorted({'A','B','C'} - {'ABC'[i%3]})))"
            " for i in range(30)];"
            "print([r.source_id for r in AudioDataset(tuple(recs)).rebalance(random_state=3)])")
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("0", "7", "99")}
    assert len(outputs) == 1


def test_rebalance_multi_label_semantics():
    data = synthetic({("A",): 10, ("B",): 6, ("C",): 20, ("A", "C"): 5, ("B", "C"): 4})
    buckets = {label: sum(label in r.original_labels for r in data) for label in "ABC"}
    m = min(buckets.values())  # B: 6 + 4 = 10  ->  A: 15, B: 10, C: 29  ->  m = 10
    assert (buckets, m) == ({"A": 15, "B": 10, "C": 29}, 10)
    balanced = data.rebalance(label_space="original", random_state=0)
    counts = {label: sum(label in r.original_labels for r in balanced) for label in "ABC"}
    assert all(count >= m for count in counts.values())  # every label ends with at least m records
    assert len(balanced) <= 3 * m  # at most (number of buckets) * m records
    assert len(balanced) == len({r.source_id for r in balanced})  # unique records
    assert counts != {"A": m, "B": m, "C": m}  # ... and the final histogram is NOT exactly equal (co-occurrence)


def test_rebalance_single_label_buckets_are_exactly_balanced():
    data = synthetic({("A",): 10, ("B",): 6, ("C",): 20})
    balanced = data.rebalance(label_space="original", random_state=1)
    assert Counter(label for r in balanced for label in r.original_labels) == {"A": 6, "B": 6, "C": 6}


def test_rebalance_focus_labels_are_exact_and_drop_other_records():
    data = synthetic({("A",): 10, ("B",): 6, ("C",): 20, ("A", "D"): 3, ("D",): 50})
    balanced = data.rebalance(focus_labels=["A", "B"], label_space="original", random_state=0)
    assert {label for r in balanced for label in r.original_labels} <= {"A", "B", "D"}
    assert all({"A", "B"} & set(r.original_labels) for r in balanced)  # records without a focus label are dropped
    assert not any(r.original_labels == ("D",) or r.original_labels == ("C",) for r in balanced)
    assert len(data.rebalance(focus_labels=["a"], label_space="original", random_state=0)) == 0  # exact, case-sensitive


def test_rebalance_edge_cases(vgg):
    assert len(AudioDataset(()).rebalance()) == 0
    assert len(vgg.rebalance(focus_labels=["no such label"], label_space="original")) == 0
    seeded = vgg.rebalance(focus_labels=["foghorn", "wind noise"], label_space="original", random_state=0)
    assert Counter(r.original_labels[0] for r in seeded) == {"foghorn": 1, "wind noise": 1}


def test_functional_rebalance_equals_method(audioset):
    focus = ["Music", "Speech"]
    kwargs = dict(focus_labels=focus, label_space="original", random_state=5)
    assert rebalance(audioset, **kwargs).records == audioset.rebalance(**kwargs).records
    from_path = rebalance(AUDIOSET_CSV, labels_file=AUDIOSET_LABELS, **kwargs)  # no SALT labels, same selection
    assert [r.source_id for r in from_path] == [r.source_id for r in audioset.rebalance(**kwargs)]


def test_rebalance_order_follows_sorted_labels_then_seeded_shuffle():
    # The documented procedure, re-implemented independently, must give the same result.
    data = synthetic({("b",): 3, ("a",): 4, ("c",): 5, ("a", "c"): 2})
    rng = random.Random(11)
    buckets = {label: [r for r in data if label in r.original_labels] for label in ("a", "b", "c")}
    m = min(len(bucket) for bucket in buckets.values())
    expected, seen = [], set()
    for label in sorted(buckets):
        bucket = list(buckets[label])
        rng.shuffle(bucket)
        added = 0
        for record in bucket:
            if record.source_id not in seen:
                seen.add(record.source_id)
                expected.append(record.source_id)
                added += 1
            if added >= m:
                break
    assert [r.source_id for r in data.rebalance(label_space="original", random_state=11)] == expected


def test_dataset_repr_is_short(audioset):
    assert repr(audioset) == "AudioDataset(records=30, source_format='AudioSet', salt_resolver=yes)"
    assert "no" in repr(AudioDataset(()))


def test_resolver_loaded_from_fixture_files_is_usable_standalone():
    resolver = SaltResolver.from_mapping_file(SALT_MAPPING, SALT_ROOTS)
    assert resolver.resolve("AudioSet", ["Animal"]) == ("animal",)
