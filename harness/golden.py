"""Runs a case's graphs on the GPU through the Triton JIT and stores their inputs and outputs as the
golden data every later level is judged against (see harness/README.md).

    uv run python -m harness.golden vector_add rmsnorm      # or --all
"""

import argparse
import json
import subprocess
import sys

import torch
import triton
from safetensors.torch import save_file

from harness.registry import CASES
from harness import weights as checkpoints
from tests.harness import ROOT, TORCH_DTYPES, load_model, run

GOLDEN = ROOT / "harness" / "golden"


def make(buf, case, generator, checkpoint):
    """A fresh tensor for an input, weight or state buffer."""
    dtype = TORCH_DTYPES[buf.dtype]
    if buf.kind == "state":
        return torch.zeros(buf.shape, dtype=dtype, device="cuda")
    if checkpoint is not None and buf.kind == "weight":
        if buf.name not in checkpoint:
            raise KeyError(f"{buf.name} is not in the checkpoint")
        return checkpoint[buf.name].to(device="cuda", dtype=dtype).contiguous()
    if not dtype.is_floating_point:
        if buf.name not in case.int_ranges:
            raise ValueError(f"integer buffer {buf.name} needs a range in harness/registry.py")
        low, high = case.int_ranges[buf.name]
        return torch.randint(low, high, buf.shape, generator=generator, device="cuda", dtype=dtype)
    x = torch.randn(buf.shape, generator=generator, device="cuda", dtype=torch.float32)
    if buf.kind == "weight" and len(buf.shape) >= 2:
        fan_in = 1
        for d in buf.shape[1:]:
            fan_in *= d
        x = x / fan_in**0.5
    elif buf.kind == "weight":
        x = 0.5 + torch.rand(buf.shape, generator=generator, device="cuda")  # positive: also valid as a variance
    return x.to(dtype)


def generate(name: str):
    case = CASES[name]
    model = load_model(case.path)
    generator = torch.Generator(device="cuda").manual_seed(case.seed)
    checkpoint = checkpoints.load(case.weights, name) if case.weights else None
    out_dir = GOLDEN / name
    tensors, graphs = {}, []
    for i, graph in enumerate(model.graphs()):
        for buf in graph.buffers.values():
            if buf.kind in ("input", "weight", "state") and buf.name not in tensors:
                tensors[buf.name] = make(buf, case, generator, checkpoint)
            elif buf.kind == "input":
                # A later graph's input that an earlier graph already had gets fresh values.
                tensors[buf.name] = make(buf, case, generator, checkpoint)
        ins = {b.name: tensors[b.name] for b in graph.buffers.values() if b.kind in ("input", "weight", "state")}
        (out_dir / str(i)).mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in ins.items()}, out_dir / str(i) / "inputs.safetensors")
        run(graph, tensors)
        torch.cuda.synchronize()
        outs = {b.name: tensors[b.name] for b in graph.buffers.values() if b.kind in ("output", "state")}
        for key, value in outs.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} graph {i}: golden {key} is not finite")
        save_file({k: v.contiguous().cpu() for k, v in outs.items()}, out_dir / str(i) / "outputs.safetensors")
        graphs.append({
            "index": i,
            "name": graph.name,
            "buffers": {b.name: {"kind": b.kind, "dtype": b.dtype, "shape": list(b.shape)} for b in graph.buffers.values()},
        })
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"case": name, "path": case.path, "seed": case.seed, "weights": case.weights,
            "triton": triton.__version__, "commit": commit, "graphs": graphs}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {len(graphs)} graph(s) -> {out_dir.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = list(CASES) if args.all else args.cases
    unknown = [n for n in names if n not in CASES]
    if unknown or not names:
        sys.exit(f"unknown or no cases {unknown}; known: {', '.join(CASES)}")
    for name in names:
        if not (ROOT / CASES[name].path).exists():
            print(f"{name}: {CASES[name].path} does not exist yet, skipped")
            continue
        generate(name)


if __name__ == "__main__":
    main()
