"""L1 per launch (teacher forcing): every launch runs on the golden state just before it, and the buffers
it writes must match the golden state just after it within one ulp of their dtype; the next launch
starts again from golden. Errors cannot compound, so a failure names the kernel that is wrong.

    python -m furiosa_compiler.l1_launch harness/out/<case>/compiler harness/golden/<case> --rtol R --atol A
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from furiosa_compiler import tensors
from furiosa_compiler.l1 import Kernels, compare


def judge(out_dir: Path, golden_dir: Path, rtol: float, atol: float) -> tuple[bool, str]:
    meta = json.loads((out_dir / "graph.json").read_text())
    kernels = Kernels(out_dir, meta["kernels"])
    count = 0
    worst, worst_at = 0.0, ""
    for gi, graph in enumerate(meta["graphs"]):
        launches = golden_dir / str(gi) / "launches"
        if not launches.exists():
            return False, f"graph {gi}: no per-launch golden (rerun python -m harness.golden)"
        buffers = dict(tensors.load(golden_dir / str(gi) / "inputs.safetensors"))
        for name, info in graph["buffers"].items():
            buffers.setdefault(name, np.zeros(info["shape"], dtype=tensors.DTYPES[info["dtype"]]))
        for j, launch in enumerate(graph["launches"]):
            try:
                _, address, written = kernels.get(launch["kernel"])
                args = [buffers[n] for n in launch["args"]]
                outs = [np.empty_like(args[w]) for w in written]
                tensors.call(address, args + outs)
            except Exception as e:  # noqa: BLE001 - reported as this launch's failure
                return False, f"graph {gi} launch {j} ({launch['kernel']}): {e}"
            golden = tensors.load(launches / f"{j}.safetensors")
            got = {launch["args"][w]: out for w, out in zip(written, outs)}
            ok, detail = compare(got, {n: golden[n] for n in got}, rtol, atol)
            if not ok:
                return False, f"graph {gi} launch {j} ({launch['kernel']}): {detail}"
            for n, out in got.items():
                ref = golden[n].astype(np.float64)
                scale = float(np.abs(ref).max()) if ref.size else 0.0
                err = float(np.abs(out.astype(np.float64) - ref).max()) if ref.size else 0.0
                if scale and err / scale > worst:
                    worst, worst_at = err / scale, f"{launch['kernel']}"
            buffers.update(golden)
            count += 1
    return True, f"{count} launches, worst rel {worst:.2g}" + (f" @ {worst_at}" if worst_at else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("golden_dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        ok, detail = judge(args.out_dir, args.golden_dir, args.rtol, args.atol)
    except Exception as e:  # noqa: BLE001 - the harness reads the last line
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(json.dumps({"status": "PASS" if ok else "FAIL", "detail": detail}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
