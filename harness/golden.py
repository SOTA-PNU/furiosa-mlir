"""Runs a case's graphs on the GPU through the Triton JIT and stores their inputs and outputs as the
golden data every later level is judged against (see harness/README.md).

    uv run python -m harness.golden vector_add rmsnorm      # or --all
    uv run python -m harness.golden --inputs qwen3_0_6b     # prints inputs_dir(): every graph's inputs with weights
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from harness.registry import CASES
from harness import weights as checkpoints
from tests.harness import ROOT, TORCH_DTYPES, load_model, run

GOLDEN = ROOT / "harness" / "golden"
# Golden inputs with their checkpoint weights restored (inputs_dir): derived data, so on the cache disk when it exists.
CACHE = Path("/mnt/nvme1n1p2/chacha/golden-cache") if Path("/mnt/nvme1n1p2/chacha").exists() else ROOT / "harness" / "out" / "golden-cache"
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))  # torchvision's checkpoint downloads, off the root disk


def weight(checkpoint, name, dtype):
    """A checkpoint weight in the graph's dtype, on the CPU: the golden run's and inputs_dir's alike."""
    if name not in checkpoint:
        raise KeyError(f"{name} is not in the checkpoint")
    return checkpoint[name].to(dtype=dtype).contiguous()


def make(buf, case, generator, checkpoint):
    """A fresh tensor for an input, weight or state buffer."""
    dtype = TORCH_DTYPES[buf.dtype]
    if buf.kind == "state":
        return torch.zeros(buf.shape, dtype=dtype, device="cuda")
    if checkpoint is not None and buf.kind == "weight":
        return weight(checkpoint, buf.name, dtype).to("cuda")
    if not dtype.is_floating_point:
        if buf.name not in case.int_ranges:
            raise ValueError(f"integer buffer {buf.name} needs a range in harness/registry.py")
        low, high = case.int_ranges[buf.name]
        return torch.randint(low, high, buf.shape, generator=generator, device="cuda", dtype=dtype)
    if buf.name in case.float_ranges:
        low, high = case.float_ranges[buf.name]
        return (low + (high - low) * torch.rand(buf.shape, generator=generator, device="cuda")).to(dtype)
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
        # Checkpoint weights are stored by reference (meta.json weights_version): inputs_dir() restores them.
        ins = {b.name: tensors[b.name] for b in graph.buffers.values()
               if b.kind in ("input", "state") or (b.kind == "weight" and checkpoint is None)}
        (out_dir / str(i)).mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in ins.items()}, out_dir / str(i) / "inputs.safetensors")
        # Per launch, the buffers it writes (pointer args the kernel stores to are not known here, so
        # every non-input buffer the launch binds), for L1's per-launch check.
        launch_dir = out_dir / str(i) / "launches"
        launch_dir.mkdir(parents=True, exist_ok=True)

        def snapshot(j, launch, tensors_):
            names = {v.name for v in launch.args.values() if hasattr(v, "kind") and v.kind not in ("input", "weight")}
            torch.cuda.synchronize()
            save_file({n: tensors_[n].contiguous().cpu() for n in sorted(names)}, launch_dir / f"{j}.safetensors")

        run(graph, tensors, after_launch=snapshot)
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
            "weights_version": checkpoints.version(case.weights, name) if case.weights else None,
            "triton": triton.__version__, "commit": commit, "graphs": graphs}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {len(graphs)} graph(s) -> {out_dir.relative_to(ROOT)}")


def inputs_dir(name: str) -> Path:
    """A dir whose <i>/inputs.safetensors holds every input, weight and state buffer of graph i: the golden
    itself, or, when it stores its checkpoint weights by reference, a copy with them restored in CACHE (built
    from the pinned checkpoint on first use, rebuilt when the golden file it came from changes)."""
    golden = GOLDEN / name
    meta = json.loads((golden / "meta.json").read_text())
    graphs = {g["index"]: [n for n, b in g["buffers"].items() if b["kind"] == "weight"] for g in meta["graphs"]}
    stored = True
    for i, weights in graphs.items():
        with safe_open(golden / str(i) / "inputs.safetensors", "pt") as f:
            stored &= set(weights) <= set(f.keys())
    if stored:
        return golden
    if not meta.get("weights_version"):
        raise ValueError(f"{golden}: inputs lack the weights and meta.json records no weights_version")
    if checkpoints.version(meta["weights"], name) != meta["weights_version"]:
        raise ValueError(f"{golden} was recorded with weights {meta['weights_version']}, harness/weights.py pins "
                         f"{checkpoints.version(meta['weights'], name)}: regenerate the golden")
    cache = CACHE / name / meta["weights_version"]
    cache.mkdir(parents=True, exist_ok=True)
    sources = {}  # each golden file's identity: a regenerated golden, or another tree's, rebuilds the copy
    for i in graphs:
        src = (golden / str(i) / "inputs.safetensors").resolve()
        sources[i] = f"{src} {src.stat().st_size} {src.stat().st_mtime_ns}"
    with open(cache / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one build per case at a time; later callers find it done
        todo = [i for i in graphs
                if not (cache / str(i) / "source").exists() or (cache / str(i) / "source").read_text() != sources[i]]
        if todo:
            checkpoint = checkpoints.load(meta["weights"], name)
            dtypes = {n: b["dtype"] for g in meta["graphs"] for n, b in g["buffers"].items()}
            for i in todo:
                tensors = load_file(golden / str(i) / "inputs.safetensors")
                tensors.update({n: weight(checkpoint, n, TORCH_DTYPES[dtypes[n]]) for n in graphs[i]})
                (cache / str(i)).mkdir(exist_ok=True)
                save_file(tensors, cache / str(i) / "inputs.tmp")
                os.replace(cache / str(i) / "inputs.tmp", cache / str(i) / "inputs.safetensors")
                (cache / str(i) / "source").write_text(sources[i])
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--inputs", action="store_true", help="print each case's inputs_dir() instead of generating")
    args = parser.parse_args()
    names = list(CASES) if args.all else args.cases
    unknown = [n for n in names if n not in CASES]
    if unknown or not names:
        sys.exit(f"unknown or no cases {unknown}; known: {', '.join(CASES)}")
    if args.inputs:
        for name in names:
            print(inputs_dir(name))
        return
    for name in names:
        if not (ROOT / CASES[name].path).exists():
            print(f"{name}: {CASES[name].path} does not exist yet, skipped")
            continue
        generate(name)


if __name__ == "__main__":
    main()
