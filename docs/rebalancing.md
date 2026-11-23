# Rebalancing (`AudioDataset.rebalance`)

```python
balanced = dataset.rebalance(focus_labels=None, label_space="original", random_state=None)
```

`rebalance` undersamples a (possibly multi-label) dataset towards its smallest label bucket. The semantics below are
the behaviour of the implementation (and of its tests); they are the same procedure that AudioSet-Tools v1 used for the
AudioSet-EV negatives, with a fixed label order and a private seeded generator.

## Procedure

1. **Buckets.** Every record is added to one bucket per label of `label_space` (`id`, `original`, `salt`, `auto`;
   each label once per record). With `focus_labels`, only those labels form buckets and records that carry none of them
   belong to no bucket and are dropped. Focus labels are compared **exactly** (case-sensitive), unlike `filter_labels`.
2. **Target.** `m` is the size of the smallest bucket. No bucket at all gives an empty dataset.
3. **Selection.** Buckets are visited in **sorted label order**. Each bucket (records in input order) is shuffled
   with one `random.Random(random_state)` shared by the whole call, then records are taken until `m` *new* records have been added
   from it; a record already selected through an earlier label is skipped and does not count.
4. **Result.** The selected records in selection order (label order, then shuffled order). Identity for "already selected" is
   the record identity of `deduplicate()` (`record_id`).

## Multi-label clips (what to expect)

A clip is in the bucket of every label it carries, but it is selected once. Hence:

* the result has **at most `len(buckets) × m` records** (usually fewer when clips are shared between buckets);
* **every label ends with at least `m` records** (`m` new ones from its own bucket, or its whole bucket if earlier labels
  already took many of its clips);
* **final per-label counts are not equal**: a label can end above `m` because its clips co-occur with labels visited earlier or
  later. `rebalance` bounds the contribution of each bucket; it does not equalize the final label histogram.

Worked examples (executed with this package; commands in [`validation.md`](validation.md)):

| Data | Buckets | `m` | Selected | Per-label counts after `random_state=0` |
|---|---|---|---|---|
| synthetic: `A`×10, `B`×6, `C`×20, `A+C`×5, `B+C`×4 | A 15, B 10, C 29 | 10 | 30 | A 11, B 10, C 16 |
| AudioSet eval (20,371 clips), focus `Speech`, `Music`, `Siren`, `Dog`, `Engine` | 5,324 / 5,695 / 212 / 230 / 189 | 189 | 945 | Speech 363, Music 269, Siren 189, Dog 190, Engine 189 |

If you need exact per-label equality, post-process the result (for example filter each label's records to `m`) or select
single-label subsets before rebalancing.

## Determinism

* The same `random_state`, the same input record **order**, the same labels and focus give the same selection and the
  same order, on any machine and for any `PYTHONHASHSEED` (tests: `test_rebalance_*`).
* `random_state=None` draws from the operating system: not reproducible.
* The shuffle starts from the input order, so a reproducible input order (sorted files, deterministic parsing) is part of
  the reproducibility contract. Parsing is deterministic; merging files keeps the order you pass.

### Difference from AudioSet-Tools v1

v1 (`rebalancing_filter`, legacy mode, the code that produced the published AudioSet-EV negatives) visited the labels in the insertion
order of a dictionary filled from `set` intersections and wrote `list(set(...))`: both depend on `PYTHONHASHSEED` and on the
file-listing order of the calling script, so the published draw cannot be regenerated bit for bit (sizes 8,500-8,502 were
observed). v2 sorts the labels and uses a private generator. The cleaned v1 snapshot (tag `v1.0.0`) added
`rebalancing_filter(..., deterministic=True, seed=s)`, which is the same procedure: on the complete AudioSet eval and balanced splits
(three focus sets including all 527 labels, seeds 0, 7, 42) v1's deterministic mode and `rebalance(random_state=s)` returned the
identical selection **and** order in all 18 combinations (script and output in [`validation.md`](validation.md)). The legacy
mode has no v2 equivalent: v2 gives a different, equally valid draw.
