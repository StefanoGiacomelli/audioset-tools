"""SaltResolver, the SALT asset lookup and the fetch step (no real network: a local HTTP server stands in)."""

from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from audioset_tools import SaltAssetError, SaltResolver, default_salt_dir, fetch_salt_assets
from audioset_tools import salt as salt_module
from audioset_tools.cli import main as cli_main

from .conftest import SALT_MAPPING, SALT_ROOTS

FIXTURE_FILES = {"salt_event_mapping.tsv": SALT_MAPPING, "salt_event_roots.json": SALT_ROOTS}
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_resolve_normalizes_labels_and_keeps_datasets_apart(salt):
    assert salt.resolve("AudioSet", ["Engine"]) == ("engine",)
    assert salt.resolve("Freesound50k", ["Engine"]) == ("vehicle",)  # same label, another dataset
    assert salt.resolve("AudioSet", ["  ENGINE ", "engine knocking", "Engine"]) == ("engine",)  # unique, ordered
    assert salt.resolve("AudioSet", ["Domestic animals, pets", "Dog"]) == ("animal", "dog")
    assert salt.resolve("AudioSet", ["not a label"]) == ()
    assert salt.resolve("NoSuchDataset", ["Engine"]) == ()


def test_loader_skips_malformed_rows():
    resolver = SaltResolver.from_mapping_file(SALT_MAPPING, SALT_ROOTS)
    # The fixture ends with a row without standard event ("Horse" kept from its good row), a tab-only row and a
    # one-column row: none may raise or create an entry.
    assert resolver.resolve("AudioSet", ["Horse"]) == ("animal",)
    assert resolver.resolve("AudioSet", ["orphan_event"]) == ()


def test_loader_reports_missing_files_and_columns(tmp_path):
    with pytest.raises(FileNotFoundError, match="SALT mapping file not found"):
        SaltResolver.from_mapping_file(tmp_path / "missing.tsv")
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        SaltResolver.from_mapping_file(bad)


def test_roots_default_to_the_file_next_to_the_mapping(tmp_path):
    for name, source in FIXTURE_FILES.items():
        (tmp_path / name).write_bytes(source.read_bytes())
    with_roots = SaltResolver.from_mapping_file(tmp_path / "salt_event_mapping.tsv")
    assert "ambulance_siren" in with_roots.descendants("vehicle")
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / "salt_event_mapping.tsv").write_bytes(SALT_MAPPING.read_bytes())
    assert SaltResolver.from_mapping_file(lonely / "salt_event_mapping.tsv").descendants("vehicle") == ()


def test_default_lookup_uses_the_environment_and_explains_the_remedy(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIOSET_TOOLS_SALT_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    expected = tmp_path / "cache" / "audioset-tools" / "salt" / salt_module.SALT_UPSTREAM_COMMIT[:12]
    assert default_salt_dir() == expected
    with pytest.raises(FileNotFoundError, match="fetch-salt"):
        SaltResolver.from_mapping_file()
    monkeypatch.setenv("AUDIOSET_TOOLS_SALT_DIR", str(tmp_path / "mine"))
    (tmp_path / "mine").mkdir()
    for name, source in FIXTURE_FILES.items():
        (tmp_path / "mine" / name).write_bytes(source.read_bytes())
    assert default_salt_dir() == tmp_path / "mine"
    assert SaltResolver.from_mapping_file().resolve("AudioSet", ["Animal"]) == ("animal",)


def test_pinned_upstream_constants_are_well_formed():
    commit = salt_module.SALT_UPSTREAM_COMMIT
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    assert set(salt_module.SALT_ASSETS) == {"salt_event_mapping.tsv", "salt_event_roots.json"}
    for repo_path, digest in salt_module.SALT_ASSETS.values():
        assert repo_path.startswith("py-salt/assets/")
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert salt_module.SALT_UPSTREAM_REPOSITORY == "https://github.com/tpt-adasp/salt"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence the test output
        pass


@pytest.fixture
def local_mirror(tmp_path):
    """HTTP server on 127.0.0.1 serving ``<commit>/py-salt/assets/<file>`` built from the synthetic fixtures."""

    root = tmp_path / "mirror"
    assets = root / FAKE_COMMIT / "py-salt" / "assets"
    assets.mkdir(parents=True)
    for name, source in FIXTURE_FILES.items():
        (assets / name).write_bytes(source.read_bytes())
    handler = lambda *a, **k: _Handler(*a, directory=str(root), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checksums = {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in FIXTURE_FILES.items()}
    yield f"http://127.0.0.1:{server.server_address[1]}", checksums, assets
    server.shutdown()
    server.server_close()


def test_fetch_downloads_verifies_and_is_idempotent(tmp_path, local_mirror):
    base_url, checksums, _ = local_mirror
    dest = tmp_path / "salt"
    mapping, roots = fetch_salt_assets(dest, base_url=base_url, commit=FAKE_COMMIT, checksums=checksums)
    assert mapping == dest / "salt_event_mapping.tsv" and roots == dest / "salt_event_roots.json"
    assert mapping.read_bytes() == SALT_MAPPING.read_bytes() and roots.read_bytes() == SALT_ROOTS.read_bytes()
    assert not list(dest.glob("*.part"))
    stamp = mapping.stat().st_mtime_ns
    fetch_salt_assets(dest, base_url=base_url, commit=FAKE_COMMIT, checksums=checksums)
    assert mapping.stat().st_mtime_ns == stamp  # valid files are kept
    mapping.write_text("corrupted", encoding="utf-8")
    fetch_salt_assets(dest, base_url=base_url, commit=FAKE_COMMIT, checksums=checksums)
    assert mapping.read_bytes() == SALT_MAPPING.read_bytes()  # an invalid file is replaced
    fetch_salt_assets(dest, base_url=base_url, commit=FAKE_COMMIT, checksums=checksums, force=True)
    resolver = SaltResolver.from_mapping_file(mapping)
    assert resolver.resolve("AudioSet", ["Animal"]) == ("animal",) and "dog" in resolver.descendants("animal")


def test_fetch_rejects_a_checksum_mismatch_and_writes_nothing(tmp_path, local_mirror):
    base_url, checksums, _ = local_mirror
    wrong = dict(checksums, **{"salt_event_roots.json": "0" * 64})
    dest = tmp_path / "salt"
    with pytest.raises(SaltAssetError, match="Checksum mismatch for salt_event_roots.json"):
        fetch_salt_assets(dest, base_url=base_url, commit=FAKE_COMMIT, checksums=wrong)
    assert not (dest / "salt_event_roots.json").exists() and not list(dest.glob("*.part"))


def test_fetch_reports_download_errors_and_requires_checksums_for_other_commits(tmp_path, local_mirror):
    base_url, checksums, _ = local_mirror
    with pytest.raises(SaltAssetError, match="Could not download"):
        fetch_salt_assets(tmp_path / "a", base_url=base_url, commit="f" * 40, checksums=checksums)
    with pytest.raises(ValueError, match="checksums are required"):
        fetch_salt_assets(tmp_path / "b", base_url=base_url, commit=FAKE_COMMIT)


def test_cli_fetch_salt_and_version(tmp_path, local_mirror, capsys, monkeypatch):
    base_url, checksums, _ = local_mirror
    monkeypatch.setattr(salt_module, "SALT_RAW_BASE_URL", base_url)
    monkeypatch.setattr(salt_module, "SALT_UPSTREAM_COMMIT", FAKE_COMMIT)
    monkeypatch.setattr(salt_module, "SALT_ASSETS",
                        {name: (path, checksums[name]) for name, (path, _) in salt_module.SALT_ASSETS.items()})
    assert cli_main(["fetch-salt", "--dest", str(tmp_path / "cli")]) == 0
    out = capsys.readouterr().out
    assert "salt_event_mapping.tsv" in out and "AUDIOSET_TOOLS_SALT_DIR" in out
    assert "declares no licence" in out
    assert (tmp_path / "cli" / "salt_event_roots.json").exists()
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    assert "audioset-tools 2.0.0" in capsys.readouterr().out


class _StubExplorer:
    """Minimal stand-in for py_salt.EventExplorer (``map_df`` with ``iterrows`` and a private ``_std_col``)."""

    class _Frame:
        def __init__(self, rows):
            self._rows = rows

        def iterrows(self):
            return enumerate(self._rows)

    def __init__(self, std_col="standard_event", public=False):
        rows = [{"dataset": "AudioSet", "dataset_label": " Animal ", std_col: "animal"},
                {"dataset": "VggSound", "dataset_label": "coyote howling", std_col: "coyote"},
                {"dataset": "VggSound", "dataset_label": "", std_col: "ignored"}]
        self.map_df = self._Frame(rows)
        self.roots = {"animal": {"wild_animal": {"coyote": {}}}}
        if public:
            self.std_col = std_col
        else:
            self._std_col = std_col


@pytest.mark.parametrize("explorer", [_StubExplorer(), _StubExplorer("event", public=True), _StubExplorer("event")])
def test_from_py_salt_with_an_explorer_object(explorer):
    resolver = SaltResolver.from_py_salt(explorer)
    assert resolver.resolve("AudioSet", ["animal"]) == ("animal",)
    assert resolver.resolve("VggSound", ["Coyote Howling"]) == ("coyote",)
    assert resolver.descendants("animal") == ("wild_animal", "coyote")


def test_from_py_salt_without_the_package_explains_the_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "py_salt", None)  # makes "from py_salt import ..." raise ImportError
    with pytest.raises(ImportError, match=r"audioset-tools\[salt\]"):
        SaltResolver.from_py_salt()


def test_the_package_does_not_ship_salt_assets():
    package = Path(salt_module.__file__).parent
    shipped = {path.name for path in package.rglob("*") if path.is_file()}
    assert "salt_event_mapping.tsv" not in shipped and "salt_event_roots.json" not in shipped


def test_descendants_terminate_on_self_nested_roots_and_cycles():
    # F-9: the real SALT roots file nests each top-level root inside itself ({"music": {"music": {...}}}); a plain
    # breadth-first search never ended for those labels.
    resolver = SaltResolver({}, roots={"music": {"music": {"happy_music": {}, "singing": {}}},
                                       "a": {"b": {"c": {"a": {}}}}})
    assert resolver.descendants("music") == ("happy_music", "singing")
    assert resolver.descendants("a") == ("b", "c")  # cycle a -> b -> c -> a: terminates, never lists the label itself
    assert resolver.descendants("c") == ("a", "b")
    assert resolver.expand_salt_labels(["music"], include_descendants=True) == ("music", "happy_music", "singing")


def test_descendants_list_each_label_once_and_merge_repeated_nodes():
    resolver = SaltResolver({}, roots={"x": {"y": {"z": {}}, "z": {}}, "w": {"x": {"q": {}}}})
    assert resolver.descendants("x") == ("y", "z", "q")  # z is reachable twice, q comes from the second occurrence of x
    assert resolver.descendants("nothing") == ()


def test_the_synthetic_roots_mirror_the_self_nesting_of_the_real_file(salt):
    assert salt.descendants("music") == ("singing",)
    assert salt.expand_salt_labels(["music"], include_descendants=True) == ("music", "singing")
