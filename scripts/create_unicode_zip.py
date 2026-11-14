#!/usr/bin/env python3
"""Create a ZIP64 archive while preserving Unicode path names."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    walk_root = source
    if os.name == "nt" and not str(source).startswith("\\\\?\\"):
        walk_root = Path("\\\\?\\" + str(source))

    files = sorted(path for path in walk_root.rglob("*") if path.is_file())
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as output:
        for path in files:
            relative = PurePosixPath(path.relative_to(walk_root).as_posix())
            output.write(path, str(PurePosixPath(args.root_name) / relative))

    print(f"UNICODE_ZIP_PASS files={len(files)} archive={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

