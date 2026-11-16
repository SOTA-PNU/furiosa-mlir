"""furiosa-compile: Triton model file -> per-kernel TTIR and raised linalg, plus graph.json, and with
--crate the furiosa-opt crate that runs the model's graphs.

    compiler/.venv/bin/python -m furiosa_compiler <model.py> -o <out dir> [--crate <crate dir>]
"""

import argparse
import sys
from pathlib import Path

from furiosa_compiler.crate import CrateError, build_crate, failed_crate
from furiosa_compiler.driver import compile_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--keep-going", action="store_true", help="record a failing kernel in graph.json and continue")
    parser.add_argument("--crate", type=Path, help="also write the furiosa-opt crate here")
    args = parser.parse_args()

    result = compile_model(args.model.resolve(), args.out, keep_going=args.keep_going)
    failed = 0
    for symbol, kernel in result["kernels"].items():
        status = f"FAILED {kernel['error']}" if "error" in kernel else f"written {kernel['written']}"
        failed += "error" in kernel
        print(f"{symbol}: {status}")
    if failed:
        sys.exit(1)
    if args.crate:
        try:
            build_crate(result, args.out, args.crate)
            print(f"crate: {args.crate}")
        except CrateError as e:
            # The raised kernels stay valid; the crate carries the diagnostic as a compile_error!.
            failed_crate(args.crate, result["model"], str(e))
            print(f"crate: FAILED {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
