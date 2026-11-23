"""Command-line entry point: ``audioset-tools`` (also ``python -m audioset_tools``)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .salt import SALT_UPSTREAM_COMMIT, SALT_UPSTREAM_REPOSITORY, SaltAssetError, default_salt_dir, fetch_salt_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audioset-tools",
                                     description="AudioSet-Tools: taxonomy-aware curation of AudioSet, "
                                                 "AudioSet Strong and VGGSound metadata.")
    parser.add_argument("--version", action="version", version=f"audioset-tools {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-salt",
                                  help="download the SALT asset files from the upstream repository",
                                  description=f"Download salt_event_mapping.tsv and salt_event_roots.json from "
                                              f"{SALT_UPSTREAM_REPOSITORY} (commit {SALT_UPSTREAM_COMMIT[:12]}) and "
                                              "verify their SHA-256. The upstream repository declares no licence: the "
                                              "files are fetched for your own use and are never redistributed by "
                                              "audioset-tools.")
    fetch.add_argument("--dest", type=Path, default=None,
                       help=f"target directory (default: {default_salt_dir()})")
    fetch.add_argument("--force", action="store_true", help="download again even if the files are present and valid")
    fetch.add_argument("--timeout", type=float, default=60.0, help="network timeout in seconds (default: 60)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fetch-salt":
        try:
            mapping, roots = fetch_salt_assets(args.dest, force=args.force, timeout=args.timeout)
        except SaltAssetError as exc:
            print(f"error: {exc}")
            return 1
        print(f"SALT mapping: {mapping}")
        print(f"SALT roots:   {roots}")
        print("Note: the upstream SALT repository declares no licence; use the files for your own work and do not "
              "redistribute them.")
        if args.dest is not None:
            print("Use them with SaltResolver.from_mapping_file(mapping, roots), or export "
                  "AUDIOSET_TOOLS_SALT_DIR to that directory for the default lookup.")
        return 0

    return 2
