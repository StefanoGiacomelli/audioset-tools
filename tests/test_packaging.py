"""Packaging and repository-metadata consistency (runs against an editable install and against the built wheel)."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import re
from pathlib import Path

import pytest

import audioset_tools

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_consistent():
    assert audioset_tools.__version__ == "2.0.0"
    assert importlib.metadata.version("audioset-tools") == audioset_tools.__version__
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "2\.0\.0"$', pyproject, re.M)
    meta = importlib.metadata.metadata("audioset-tools")          # the licence travels with the package metadata
    assert meta.get("License-Expression") == "GPL-3.0-or-later"
    assert meta.get_all("License-File") == ["LICENSE"]


def test_package_is_typed_and_ships_no_data_files():
    package = importlib.resources.files("audioset_tools")
    assert (package / "py.typed").is_file()
    names = {item.name for item in package.iterdir()}
    assert {"models.py", "parsers.py", "dataset.py", "filters.py", "utils.py", "events.py", "salt.py",
            "downloader.py", "cli.py"} <= names
    assert not any(name.endswith((".tsv", ".json", ".csv", ".wav")) for name in names)


def test_public_names_exist_and_are_documented():
    for name in audioset_tools.__all__:
        assert hasattr(audioset_tools, name), name
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = [name for name in audioset_tools.__all__ if not name.startswith("__") and name not in api]
    assert missing == [], f"public names missing from docs/api.md: {missing}"


def test_console_script_is_declared():
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    scripts = {entry.name: entry.value for entry in entry_points if entry.name == "audioset-tools"}
    assert scripts == {"audioset-tools": "audioset_tools.cli:main"}


def test_declared_extras_match_the_documentation():
    extras = set(importlib.metadata.metadata("audioset-tools").get_all("Provides-Extra"))
    assert extras == {"download", "plot", "torch", "salt", "dev"}
    requires = importlib.metadata.requires("audioset-tools")
    core = {re.split(r"[ ;<>=!~\[]", item)[0] for item in requires if "extra ==" not in item}
    assert core == {"numpy", "tqdm"}


@pytest.mark.skipif(not (ROOT / "CITATION.cff").exists(), reason="repository root files are not available")
def test_citation_and_notice_files_are_consistent():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in citation
    assert "https://orcid.org/0009-0009-0438-1748" in citation
    assert 'doi: "10.1186/s13636-025-00436-z"' in citation and "10.5281/zenodo.14882314" in citation
    assert 'version: "2.0.0"' in citation
    assert "license: GPL-3.0-or-later" in citation
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence.lstrip().startswith("GNU GENERAL PUBLIC LICENSE")  # the official GNU GPL v3 text (gnu.org)
    assert "Version 3, 29 June 2007" in licence[:200]
    notice = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
    assert (
        "GNU General Public License, version 3 or (at your option) any later version "
        "(SPDX identifier GPL-3.0-or-later)"
    ) in notice
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "[2.0.0]" in changelog
    assert not (ROOT / "salt_event_mapping.tsv").exists()  # SALT assets are never committed (no upstream licence)
