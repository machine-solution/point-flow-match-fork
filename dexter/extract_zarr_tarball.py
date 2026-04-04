#!/usr/bin/env python3
"""Extract train_*.tar.gz so the zarr tree (directory with data/ + meta/) lands at DEST."""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


def find_zarr_roots(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if "data" in dirnames and "meta" in dirnames:
            out.append(p)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: extract_zarr_tarball.py ARCHIVE.tar.gz DEST_DIR", file=sys.stderr)
        raise SystemExit(2)
    archive = Path(sys.argv[1]).resolve()
    dest = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp)

        roots = find_zarr_roots(tmp)
        if not roots:
            print(
                "ERROR: archive has no directory with both data/ and meta/",
                file=sys.stderr,
            )
            raise SystemExit(1)

        zr = min(roots, key=lambda p: (len(p.parts), str(p)))

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if zr == tmp:
            dest.mkdir(parents=True, exist_ok=True)
            for sub in ("data", "meta"):
                shutil.move(str(zr / sub), str(dest / sub))
        else:
            shutil.move(str(zr), str(dest))

    print(f"OK: zarr -> {dest}")


if __name__ == "__main__":
    main()
