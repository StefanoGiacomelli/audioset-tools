"""Optional SALT label resolution for AudioSet-Tools.

SALT (Standardized Audio event Label Taxonomy, Stamatiadis et al., DCASE 2024) maps the labels of many audio
datasets onto one hierarchy. The two SALT asset files used here (``salt_event_mapping.tsv`` and
``salt_event_roots.json``) are **not distributed with this package**: the upstream repository
``tpt-adasp/salt`` does not declare a licence, so redistribution is not assumed. Use ``fetch-salt``
(``audioset-tools fetch-salt`` or :func:`fetch_salt_assets`) to download them from the upstream repository at a
pinned commit with checksum verification, or point the resolver to your own copies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path

#: Upstream repository of SALT (no licence file at the pinned commit; see NOTICE and docs/salt.md).
SALT_UPSTREAM_REPOSITORY = "https://github.com/tpt-adasp/salt"
#: Upstream ``main`` HEAD (2024-11-28, "Update Readme") on which the checksums below were computed.
SALT_UPSTREAM_COMMIT = "e89d13b65f59bf621539faaf4dd3ad7ff81232c6"
SALT_RAW_BASE_URL = "https://raw.githubusercontent.com/tpt-adasp/salt"
MAPPING_FILENAME = "salt_event_mapping.tsv"
ROOTS_FILENAME = "salt_event_roots.json"
#: Environment variable holding a directory with the two SALT asset files (read by the default lookup).
SALT_DIR_ENV = "AUDIOSET_TOOLS_SALT_DIR"

#: File name -> (path inside the upstream repository, SHA-256 of the file at ``SALT_UPSTREAM_COMMIT``).
SALT_ASSETS: dict[str, tuple[str, str]] = {
    MAPPING_FILENAME: ("py-salt/assets/salt_event_mapping.tsv",
                       "7b75ce63949afef979aebba5ef9fc4dc850a8a7ac1c32b5e0cccc425e58be165"),
    ROOTS_FILENAME: ("py-salt/assets/salt_event_roots.json",
                     "89c85198a327d7eed86400d884fa3efe6551b2739dd3e9eee0bee035bd24a4f9"),
}


class SaltAssetError(RuntimeError):
    """The SALT asset files could not be downloaded or failed checksum verification."""


def _norm_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def default_salt_dir() -> Path:
    """Directory where the SALT assets are looked up and where ``fetch-salt`` stores them by default.

    ``$AUDIOSET_TOOLS_SALT_DIR`` if set, otherwise ``$XDG_CACHE_HOME`` (or ``~/.cache``) followed by
    ``audioset-tools/salt/<first 12 characters of the pinned upstream commit>``.
    """

    override = os.environ.get(SALT_DIR_ENV)
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"

    return base / "audioset-tools" / "salt" / SALT_UPSTREAM_COMMIT[:12]


def fetch_salt_assets(dest: str | Path | None = None,
                      *,
                      force: bool = False,
                      base_url: str | None = None,
                      commit: str | None = None,
                      checksums: Mapping[str, str] | None = None,
                      timeout: float = 60.0) -> tuple[Path, Path]:
    """Download the two SALT asset files from the upstream repository and verify their SHA-256.

    Returns ``(mapping_path, roots_path)``. Files already present with the expected checksum are kept unless
    ``force`` is true. Nothing is written when a checksum does not match. ``base_url`` (default: the raw GitHub
    URL of the upstream repository), ``commit`` (default: the pinned one) and ``checksums`` allow a mirror or another
    revision; a commit other than the pinned one requires explicit ``checksums``.
    This function performs network access; the tests of this package exercise it against a local HTTP server only.
    """

    base_url = base_url or SALT_RAW_BASE_URL
    commit = commit or SALT_UPSTREAM_COMMIT
    expected = {name: spec[1] for name, spec in SALT_ASSETS.items()}
    if checksums:
        expected.update(checksums)
    elif commit != SALT_UPSTREAM_COMMIT:
        raise ValueError("checksums are required for a commit other than the pinned upstream commit.")

    directory = Path(dest).expanduser() if dest is not None else default_salt_dir()
    directory.mkdir(parents=True, exist_ok=True)

    from . import __version__

    paths = []
    for name, (repo_path, _) in SALT_ASSETS.items():
        target = directory / name
        paths.append(target)
        if target.exists() and not force and _sha256(target.read_bytes()) == expected[name]:
            continue
        url = f"{base_url.rstrip('/')}/{commit}/{repo_path}"
        request = urllib.request.Request(url, headers={"User-Agent": f"audioset-tools/{__version__}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https/http only
                payload = response.read()
        except OSError as exc:
            raise SaltAssetError(f"Could not download {url}: {exc}") from exc
        digest = _sha256(payload)
        if digest != expected[name]:
            raise SaltAssetError(f"Checksum mismatch for {name} downloaded from {url}: expected {expected[name]}, "
                                 f"got {digest}. Nothing was written.")
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(payload)
        os.replace(partial, target)

    return paths[0], paths[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class SaltResolver:
    """Resolve dataset labels to SALT standardized labels.

    The resolver is deliberately lightweight: it can read SALT's TSV/JSON assets
    directly, and it can also be constructed from a py_salt explorer object when
    callers need py-salt's richer taxonomy navigation elsewhere.
    """

    def __init__(self,
                 mapping: dict[tuple[str, str], set[str]],
                 roots: dict | None = None) -> None:
        self._mapping = mapping
        self._roots = roots or {}
        self._children = self._build_children_index(self._roots)

    @classmethod
    def from_mapping_file(cls,
                          mapping_file: str | Path | None = None,
                          roots_file: str | Path | None = None) -> SaltResolver:
        """Create a resolver from SALT asset files.

        Without arguments the files are looked up in :func:`default_salt_dir` (populated by ``fetch-salt`` or by
        ``$AUDIOSET_TOOLS_SALT_DIR``); a missing default raises ``FileNotFoundError`` with the remedy. When only
        ``mapping_file`` is given, the roots file is read from the same directory if it exists there (otherwise no
        hierarchy is available and ``include_descendants`` expands nothing).
        """

        if mapping_file is None:
            mapping_path = default_salt_dir() / MAPPING_FILENAME
            if not mapping_path.exists():
                raise FileNotFoundError(f"SALT mapping file not found: {mapping_path}. The SALT assets are not "
                                        "distributed with audioset-tools (the upstream repository declares no "
                                        "licence): run 'audioset-tools fetch-salt', set "
                                        f"${SALT_DIR_ENV}, or pass mapping_file/roots_file explicitly.")
        else:
            mapping_path = Path(mapping_file)
        if roots_file is None:
            roots_path = mapping_path.parent / ROOTS_FILENAME
        else:
            roots_path = Path(roots_file)

        if not mapping_path.exists():
            raise FileNotFoundError(f"SALT mapping file not found: {mapping_path}")

        mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"standard_event", "dataset_label", "dataset"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"SALT mapping file is missing columns: {sorted(missing)}")
            for row in reader:
                dataset = (row.get("dataset") or "").strip()
                dataset_label = (row.get("dataset_label") or "").strip()
                standard_event = (row.get("standard_event") or "").strip()
                if dataset and dataset_label and standard_event:
                    mapping[(dataset, _norm_label(dataset_label))].add(standard_event)

        roots = {}
        if roots_path.exists():
            with roots_path.open(encoding="utf-8") as handle:
                roots = json.load(handle)

        return cls(dict(mapping), roots=roots)

    @classmethod
    def from_py_salt(cls, explorer=None) -> SaltResolver:
        """Create a resolver using py-salt's default EventExplorer assets.

        This keeps ``audioset_tools`` usable without importing py_salt at module
        import time. If py_salt is unavailable to the caller's environment, this
        method raises an ImportError with a precise message.

        Interface used from the explorer object: ``map_df`` (a pandas DataFrame with the columns ``dataset`` and
        ``dataset_label`` plus the standard-event column), the column name (public attribute ``std_col`` if the
        explorer has one, otherwise the private ``_std_col`` of the tested py-salt revision, otherwise
        ``"standard_event"``) and the optional ``roots`` mapping. The tested revision is the ``salt`` extra of this
        package (fork commit ``2edbadb``); any object offering these attributes works.
        """

        if explorer is None:
            try:
                from py_salt import EventExplorer
            except ImportError as exc:
                raise ImportError("py_salt is not importable. Use SaltResolver.from_mapping_file() "
                                  "or install the optional extra: pip install 'audioset-tools[salt]' "
                                  "(Python < 3.12, numpy < 2).") from exc
            explorer = EventExplorer()

        std_col = getattr(explorer, "std_col", None) or getattr(explorer, "_std_col", None) or "standard_event"
        mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
        for _, row in explorer.map_df.iterrows():
            dataset = str(row["dataset"]).strip()
            dataset_label = str(row["dataset_label"]).strip()
            standard_event = str(row[std_col]).strip()
            if dataset and dataset_label and standard_event:
                mapping[(dataset, _norm_label(dataset_label))].add(standard_event)

        return cls(dict(mapping), roots=getattr(explorer, "roots", {}))

    def resolve(self, dataset: str, labels: Iterable[str]) -> tuple[str, ...]:
        """Resolve source labels for one mapped dataset into SALT labels."""

        resolved: list[str] = []
        seen = set()
        for label in labels:
            for salt_label in sorted(self._mapping.get((dataset, _norm_label(label)), ())):
                if salt_label not in seen:
                    seen.add(salt_label)
                    resolved.append(salt_label)

        return tuple(resolved)

    def expand_salt_labels(self,
                           labels: Iterable[str],
                           include_descendants: bool = False) -> tuple[str, ...]:
        """Expand SALT labels with descendants from SALT roots when requested."""

        result: list[str] = []
        seen = set()
        for label in labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
            if include_descendants:
                for descendant in self.descendants(label):
                    if descendant not in seen:
                        seen.add(descendant)
                        result.append(descendant)

        return tuple(result)

    def descendants(self, label: str) -> tuple[str, ...]:
        """Return the descendants of a SALT label from the roots JSON.

        Breadth-first order of first discovery, each label once, never the label itself. The children of every
        occurrence of a label in the nested roots dictionary are merged, and cycles are handled: the real SALT roots
        file repeats each of its ten top-level roots as its own first child (``{"music": {"music": {...}}}``), which
        made a plain breadth-first search of the first prototype run forever for those labels.
        """

        output: list[str] = []
        visited = {label}
        queue = deque(self._children.get(label, ()))
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            output.append(current)
            queue.extend(self._children.get(current, ()))

        return tuple(output)

    @staticmethod
    def _build_children_index(roots: dict) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = defaultdict(list)

        def walk(node: dict | list, parent: str | None = None) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if parent is not None:
                        children[parent].append(key)
                    walk(value, key)

        walk(roots)

        return {key: tuple(dict.fromkeys(values)) for key, values in children.items()}
