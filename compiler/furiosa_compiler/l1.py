"""Harness level L1: runs a compiled model's raised linalg kernels on the CPU, graph by graph, and
compares every output and state buffer with golden.

    python -m furiosa_compiler.l1 harness/out/<case>/compiler harness/golden/<case> --rtol R --atol A

Graph i starts from golden inputs i (state included); each launch calls its kernel with every
pointer-argument buffer and writes the kernel's results back into the buffers it names. Prints one
JSON line: {"status": "PASS" | "FAIL", "detail": "..."}.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from furiosa_compiler import tensors

_FUNC = re.compile(r"func\.func (?:public )?@([\w$.]+)\(")


class Kernels:
    """JIT-compiles each kernel of graph.json once, on first use."""

    def __init__(self, out_dir: Path, entries: dict, stage: str = "linalg"):
        self.out_dir, self.entries, self.stage, self.compiled = out_dir, entries, stage, {}

    def get(self, symbol: str):
        if symbol not in self.compiled:
            from triton._C.libtriton import furiosa

            entry = self.entries[symbol]
            if "error" in entry:
                raise RuntimeError(f"{symbol} failed to compile: {entry['error']}")
            path = self.out_dir / f"kernels/{symbol}/{self.stage}.mlir"
            if not path.exists():
                raise RuntimeError(f"{symbol}: no {self.stage} stage at {path}")
            text = path.read_text()
            if text.startswith("// furiosa codegen failed:"):
                reason = " ".join(line[3:] for line in text.splitlines()[1:4])
                raise RuntimeError(f"{symbol}: {self.stage} stage failed: {reason[:400]}")
            names = _FUNC.findall(text)
            name = entry["kernel"] if entry["kernel"] in names else (names[0] if names else None)
            if name is None:
                raise RuntimeError(f"{symbol}: no func.func in {path}")
            try:
                engine = furiosa.runtime.compile(text)
            except RuntimeError as e:
                raise RuntimeError(f"{symbol}: JIT failed: {str(e).strip().splitlines()[0][:300]}") from None
            self.compiled[symbol] = (engine, engine.lookup(name), entry.get("written", []))
        return self.compiled[symbol]


def run_graph(graph: dict, buffers: dict, kernels: Kernels) -> None:
    for name, info in graph["buffers"].items():
        if name not in buffers:
            buffers[name] = np.zeros(info["shape"], dtype=tensors.DTYPES[info["dtype"]])
    for index, launch in enumerate(graph["launches"]):
        _, address, written = kernels.get(launch["kernel"])
        args = [buffers[name] for name in launch["args"]]
        outs = [np.empty_like(args[w]) for w in written]
        try:
            tensors.call(address, args + outs)
        except Exception as e:  # noqa: BLE001 - reported as the launch's failure
            raise RuntimeError(f"launch {index} ({launch['kernel']}) raised {e}") from None
        for w, out in zip(written, outs):
            buffers[launch["args"][w]] = out


# One unit in the last place, relative: Triton leaves fp32 accumulation order unspecified, so a raised
# kernel may round the final value one ulp differently (harness/README.md, L1).
ULP = {"bfloat16": 2.0**-7, "float16": 2.0**-10}


def compare(got: dict, want: dict, rtol: float, atol: float) -> tuple[bool, str]:
    """harness/run.py's rule: integers exact; floats max|err| <= atol + rtol * max|golden|, NaN fails;
    rtol is at least one ulp of the output dtype."""
    worst, worst_name = 0.0, ""
    for name, ref in want.items():
        if name not in got:
            return False, f"{name}: never produced"
        out = got[name]
        if out.shape != ref.shape or out.dtype != ref.dtype:
            return False, f"{name}: got {out.dtype}{list(out.shape)}, want {ref.dtype}{list(ref.shape)}"
        if not np.issubdtype(ref.dtype, np.floating) and ref.dtype.name != "bfloat16":
            mismatches = int((out != ref).sum())
            if mismatches:
                return False, f"{name}: {mismatches} integer mismatches"
            continue
        o, r = out.astype(np.float64), ref.astype(np.float64)
        err = float(np.abs(o - r).max()) if r.size else 0.0
        scale = float(np.abs(r).max()) if r.size else 0.0
        tol = max(rtol, ULP.get(ref.dtype.name, 0.0))
        if not err <= atol + tol * scale:
            return False, f"{name}: max err {err:.3g} > {atol} + {tol:.3g}*{scale:.3g}"
        if scale and err / scale >= worst:
            worst, worst_name = err / scale, name
    return True, f"rel {worst:.2g} @ {worst_name}" if worst_name else ""


# The raised form, the structured form, the structured form split into steps, and the steps tiled into the
# loops codegen maps.
STAGES = ("linalg", "structured", "steps", "tiled")


def judge(out_dir: Path, golden_dir: Path, rtol: float, atol: float) -> tuple[bool, str]:
    """Both stages must reproduce the golden outputs; the detail reports the raised stage."""
    details = []
    for stage in STAGES:
        ok, detail = judge_stage(out_dir, golden_dir, rtol, atol, stage)
        if not ok:
            return False, f"{stage}: {detail}"
        details.append(detail)
    return True, details[0]


def judge_stage(out_dir: Path, golden_dir: Path, rtol: float, atol: float, stage: str) -> tuple[bool, str]:
    graph_file = out_dir / "graph.json"
    if not graph_file.exists():
        return False, f"no {graph_file}"
    compiled = json.loads(graph_file.read_text())
    meta = json.loads((golden_dir / "meta.json").read_text())
    if len(meta["graphs"]) != len(compiled["graphs"]):
        return False, f"golden has {len(meta['graphs'])} graphs, the compiler {len(compiled['graphs'])}"
    kernels = Kernels(out_dir, compiled["kernels"], stage)
    details = []
    for i, graph in enumerate(compiled["graphs"]):
        buffers = tensors.load(golden_dir / str(i) / "inputs.safetensors")
        try:
            run_graph(graph, buffers, kernels)
        except RuntimeError as e:
            return False, f"graph {i}: {e}"
        ok, detail = compare(buffers, tensors.load(golden_dir / str(i) / "outputs.safetensors"), rtol, atol)
        if not ok:
            return False, f"graph {i}: {detail}"
        if detail:
            details.append(detail if len(compiled["graphs"]) == 1 else f"graph {i}: {detail}")
    return True, "; ".join(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("golden_dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        ok, detail = judge(args.out_dir, args.golden_dir, args.rtol, args.atol)
    except Exception as e:  # noqa: BLE001 - L1 always ends with a verdict line
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(json.dumps({"status": "PASS" if ok else "FAIL", "detail": detail}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
