"""Regenerate the small metadata excerpts of ``tests/data`` from the full official files.

Usage (the folder must contain ``AudioSet_meta``, ``AudioSet_Strong_meta`` and ``VGGsound_meta``)::

    python tests/data/make_fixtures.py /path/to/datasets

The excerpts are plain prefixes of the official files (AudioSet eval: the three comment lines and the first 30
segments; AudioSet Strong eval: the header and the first whole segments covering at least 40 event rows; VGGSound:
the first 40 rows) plus the rows of the two label-name files that those excerpts use. The SALT fixtures in
``tests/data/salt`` are synthetic and are NOT produced by this script. See ``tests/data/README.md``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
AUDIOSET_ROWS = 30
STRONG_MIN_ROWS = 40
VGGSOUND_ROWS = 40


def main(source: Path) -> None:
    # AudioSet Standard: comment header + first 30 segments, and the matching label names.
    lines = (source / "AudioSet_meta" / "eval_segments.csv").read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    segments = [line for line in lines if not line.startswith("#")][:AUDIOSET_ROWS]
    (OUT / "audioset" / "eval_segments_excerpt.csv").write_text("\n".join(comments + segments) + "\n", encoding="utf-8")
    mids = {mid for row in csv.reader(segments, skipinitialspace=True) for mid in row[3].split(",")}
    with (source / "AudioSet_meta" / "class_labels_indices.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        kept = [row for row in reader if row[1] in mids]
    with (OUT / "audioset" / "class_labels_indices_excerpt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(kept)

    # AudioSet Strong: header + whole segments until at least 40 event rows, and the matching label names.
    strong_path = source / "AudioSet_Strong_meta" / "audioset_eval_strong.tsv"
    strong_lines = strong_path.read_text(encoding="utf-8").splitlines()
    kept_lines, segment_ids = [strong_lines[0]], []
    for line in strong_lines[1:]:
        segment = line.split("\t")[0]
        if segment not in segment_ids:
            if len(kept_lines) - 1 >= STRONG_MIN_ROWS:
                break
            segment_ids.append(segment)
        kept_lines.append(line)
    (OUT / "audioset_strong" / "audioset_eval_strong_excerpt.tsv").write_text("\n".join(kept_lines) + "\n",
                                                                               encoding="utf-8")
    strong_mids = {line.split("\t")[3] for line in kept_lines[1:]}
    names = [line for line in (source / "AudioSet_Strong_meta" / "mid_to_display_name.tsv")
             .read_text(encoding="utf-8").splitlines() if line.split("\t")[0] in strong_mids]
    (OUT / "audioset_strong" / "mid_to_display_name_excerpt.tsv").write_text("\n".join(names) + "\n", encoding="utf-8")

    # VGGSound: first 40 rows.
    vgg = (source / "VGGsound_meta" / "vggsound.csv").read_text(encoding="utf-8").splitlines()[:VGGSOUND_ROWS]
    (OUT / "vggsound" / "vggsound_excerpt.csv").write_text("\n".join(vgg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
