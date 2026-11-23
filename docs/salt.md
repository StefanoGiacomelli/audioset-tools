# SALT label space

[SALT](https://github.com/tpt-adasp/salt) (Standardized Audio event Label Taxonomy; Stamatiadis, Essid et al.,
DCASE 2024, [arXiv:2409.11746](https://arxiv.org/abs/2409.11746)) maps the labels of many audio datasets onto one
hierarchy. AudioSet-Tools uses it as the `salt` query space: `SaltResolver` turns the native labels of a record into
SALT *standard events*, and optionally expands a queried standard event over its descendants.

## The assets are not distributed with this package

`SaltResolver` reads two files of the upstream repository, `py-salt/assets/salt_event_mapping.tsv` (columns
`standard_event`, `dataset_label`, `dataset`) and `py-salt/assets/salt_event_roots.json` (nested dictionary: the
hierarchy). They are **not vendored** here:

* Licence check (2026-10-26, GitHub): `GET https://api.github.com/repos/tpt-adasp/salt` returns `"license": null`;
  `GET https://api.github.com/repos/tpt-adasp/salt/license` returns 404; the tree of the default branch (43 entries, commit
  `e89d13b65f59bf621539faaf4dd3ad7ff81232c6` of 2024-11-28) contains no `LICENSE`, `COPYING` or `NOTICE` file; the
  README of the repository and of `py-salt` and `py-salt/setup.py` state no licence. Without a licence, redistribution is not
  permitted by default, so the files are fetched from upstream instead of being copied into this repository.
* Tests use **synthetic** fixtures (`tests/data/salt/`), not an excerpt of the real files.

### Fetching the assets

```bash
audioset-tools fetch-salt              # -> $AUDIOSET_TOOLS_SALT_DIR, or ~/.cache/audioset-tools/salt/e89d13b65f59/
audioset-tools fetch-salt --dest salt  # a folder of your choice (then pass the paths, or export AUDIOSET_TOOLS_SALT_DIR)
```

or `fetch_salt_assets(dest=None, force=False)` from Python. The download comes from
`raw.githubusercontent.com/tpt-adasp/salt/<commit>/py-salt/assets/...` at the pinned commit above; each file is
checked against a pinned SHA-256 and written atomically (nothing is written on a mismatch):

| File | SHA-256 at the pinned commit | Bytes |
|---|---|---|
| `salt_event_mapping.tsv` | `7b75ce63949afef979aebba5ef9fc4dc850a8a7ac1c32b5e0cccc425e58be165` | 345,709 |
| `salt_event_roots.json` | `89c85198a327d7eed86400d884fa3efe6551b2739dd3e9eee0bee035bd24a4f9` | 84,142 |

The checksums were computed from the upstream git objects of that commit. The test-suite exercises the fetch code against a local HTTP server; a real
download of both files was made once on 2026-10-27 (Python 3.13.3, macOS) and both checksums matched. On macOS with a python.org interpreter,
`CERTIFICATE_VERIFY_FAILED` means that the interpreter has no certificate bundle (run the `Install Certificates.command` of that Python, or set
`SSL_CERT_FILE`); nothing is written in that case.

### Using them

```python
from audioset_tools import SaltResolver, load_dataset

salt = SaltResolver.from_mapping_file()                       # default lookup: $AUDIOSET_TOOLS_SALT_DIR, then the cache folder
salt = SaltResolver.from_mapping_file("salt/salt_event_mapping.tsv", "salt/salt_event_roots.json")   # explicit files
salt = SaltResolver.from_py_salt()                            # through py_salt.EventExplorer (optional extra, see below)

dataset = load_dataset("vggsound.csv", salt_resolver=salt)
vehicles = dataset.filter_labels(["vehicle"], label_space="salt", include_descendants=True)
```

Without the argument and without assets, `from_mapping_file()` raises `FileNotFoundError` that names the remedy. If only the mapping file is
given, the roots file is read from the same folder when it exists there; without it no hierarchy exists and
`include_descendants` expands nothing.

Dataset keys of the mapping used by the parsers: `AudioSet`, `AudioSetStrong`, `VggSound` (other keys of the file, e.g.
`Freesound50k`, are ignored by these parsers). Labels are compared after lower-casing and collapsing whitespace; rows with an
empty field and short rows are skipped.

## Coverage on the official metadata (measured, see `docs/verify_dataset_statistics.md`)

With the assets at the pinned upstream commit: 20,367 of 20,371 AudioSet eval clips (99.98 %), 16,996 of 16,996 AudioSet
Strong eval segments and 198,775 of 199,467 VGGSound clips (99.65 %) have at least one SALT label. The same numbers result from
the author's cleaned copy of the TSV (fork commit `2358c17`, which only removes 1,057 blank rows).

## `from_py_salt()` and the `salt` extra

`SaltResolver.from_py_salt(explorer=None)` builds the resolver from a `py_salt.EventExplorer` (or any object with `map_df`, the
standard-event column name in `std_col`/`_std_col`, and `roots`). The upstream `py-salt` package cannot be installed from its
repository as it stands (`setup.py` looks for a package named `pysalt*` while the package is `py_salt`, and a missing comma merges the
`numpy` and `tqdm` requirements), so the optional extra points to the author's fork **pinned to commit
`2edbadb9fae33e46751a40de1f76b190ecb2b5e4`** (branch `gpat/py-salt-local-dev` of `StefanoGiacomelli/salt_exp`, which fixes the packaging and adds the BST mapping):

```bash
pip install "audioset-tools[salt]"      # direct reference: needs git; Python < 3.12 and numpy < 2 (py-salt requirements)
```

The accessor of the standard-event column is private in the tested py-salt revision (`_std_col`); the code reads a public `std_col` first
(forward compatible) and falls back to `_std_col`, then to `"standard_event"` (the column of `EventExplorer`). The tests use a stub
explorer; py-salt itself is not installed in the test environment.

`BstSaltMapper` (a front end for the Broad Sound Taxonomy) is part of the author's py-salt fork (`py_salt/bst_mapping.py`), not of this package.
