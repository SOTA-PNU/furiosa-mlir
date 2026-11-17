"""Runs one operator case in the compiler venv and prints one JSON line {status, detail}.
Stage `npu` (the verdict) launches kernel[grid](...) on npu3 through the furiosa driver; stage `raise`
runs the raised linalg on the CPU JIT for quick triage.

    compiler/.venv/bin/python -m harness.ops.case unary.exp.float32.1x64x64 --stage npu
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from harness.ops.cases import Case, build, check


def run_raised(kernel, grid, args):
    import numpy as np
    from furiosa_compiler import driver, l1, npu, tensors

    bufs = {n: npu.Buf(n, npu.DTYPES[t.dtype], tuple(t.shape)) for n, t in args.items() if hasattr(t, "dtype")}
    launch = npu.Launch(kernel, tuple(grid), {n: bufs.get(n, v) for n, v in args.items()})
    graph = npu.Graph(kernel.__name__, {b.name: b for b in bufs.values()}, [launch])
    with tempfile.TemporaryDirectory(prefix="ops-", dir="/mnt/nvme1n1p2/chacha") as out:
        result = driver.compile_graphs([graph], kernel.__name__, Path(out))
        symbol = next(iter(result["kernels"]))
        _, address, written = l1.Kernels(Path(out), result["kernels"]).get(symbol)
        names = result["graphs"][0]["launches"][0]["args"]
        arrays = [npu.host(args[n]) for n in names]
        outs = [np.empty_like(arrays[w]) for w in written]
        tensors.call(address, arrays + outs)
    for w, o in zip(written, outs):
        npu.store(args[names[w]], o)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("--stage", choices=["npu", "raise"], default="npu")
    a = parser.parse_args()
    case = Case.parse(a.key)
    try:
        kernel, grid, args, refs, kind = build(case)
        if a.stage == "npu":
            kernel[grid](**args)
        else:
            run_raised(kernel, grid, args)
    except Exception as e:  # noqa: BLE001 - every failure is this case's ERROR
        detail = f"{type(e).__name__}: {e}".strip().splitlines()
        print(json.dumps({"key": a.key, "status": "ERROR", "detail": " | ".join(detail[:3])[:400]}))
        sys.exit(0)
    oks, details = zip(*(check(args[name], ref, kind) for name, ref in refs.items()))
    print(json.dumps({"key": a.key, "status": "PASS" if all(oks) else "FAIL", "detail": "; ".join(details)}))


if __name__ == "__main__":
    main()
