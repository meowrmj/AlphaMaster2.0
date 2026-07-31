"""Probe and optionally build the AlphaMaster native CUDA extension."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.native_backend import load_native_extension, probe_native_build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Try to build and load the extension")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    status = probe_native_build()
    print(status)
    if not args.build:
        return
    ext = load_native_extension(verbose=args.verbose)
    print("loaded", ext)


if __name__ == "__main__":
    main()
