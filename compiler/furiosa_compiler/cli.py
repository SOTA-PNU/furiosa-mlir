"""furiosa: compile a pure-Triton model for the Furiosa NPU, run it, and trace it launch by launch.

    furiosa compile models/my_model.py -o out/my_model      # every kernel's stages, graph.json and the crate
    furiosa run out/my_model [--graph 0] [--inputs in.safetensors] [--outputs out.safetensors] [--emulation]
    furiosa trace models/my_model.py -o out/my_model [--npu] [--dump out/my_model/trace]

A model file defines graphs(): launch graphs of @triton.jit kernels (models/*.py, harness/cases/*.py). Docs:
docs/usage.md.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from furiosa_compiler import driver, l1, tensors

KINDS_FED = ("input", "weight", "state")  # buffers a graph reads before writing them


def compile_model(model: Path, out: Path, keep_going: bool = True) -> dict:
    """The model's kernels through every stage into out/compiler, and its crate into out/crate."""
    from furiosa_compiler.crate import CrateError, build_crate, failed_crate

    # Named after the output dir as well: crates of one name share cargo's unit hashes in a target slot.
    name = f"{model.stem}-{hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:8]}"
    result = driver.compile_graphs(driver.load_model(model).graphs(), name, out / "compiler", keep_going=keep_going)
    for symbol, kernel in result["kernels"].items():
        stages = out / "compiler" / "kernels" / symbol
        print(f"{symbol}: {'FAILED ' + kernel['error'] if 'error' in kernel else 'ok'}  ({stages})")
    try:
        build_crate(result, out / "compiler", out / "crate")
        print(f"crate: {out / 'crate'}")
    except CrateError as e:
        failed_crate(out / "crate", result["model"], str(e))
        print(f"crate: FAILED {e}", file=sys.stderr)
    return result


def fed_buffers(graph: dict, inputs: Path | None, seed: int = 0) -> dict:
    """The buffers a graph reads first, from `inputs` (safetensors) or random: floats N(0, 1) scaled by
    1/sqrt(fan-in) for weights, integers in [0, 2) (pass real ids and indices with --inputs)."""
    given = tensors.load(inputs) if inputs else {}
    rng = np.random.default_rng(seed)
    out = {}
    for name, info in graph["buffers"].items():
        if info["kind"] not in KINDS_FED:
            continue
        if name in given:
            out[name] = given[name]
            continue
        dtype, shape = tensors.DTYPES[info["dtype"]], info["shape"]
        if np.issubdtype(dtype, np.integer):
            out[name] = rng.integers(0, 2, shape).astype(dtype)
        else:
            x = rng.standard_normal(shape)
            if info["kind"] == "weight" and len(shape) >= 2:
                x /= np.sqrt(np.prod(shape[1:]))
            out[name] = x.astype(dtype)
    return out


def summary(arrays: dict) -> str:
    rows = []
    for name, a in arrays.items():
        f = a.astype(np.float64)
        rows.append(f"  {name:24} {str(a.dtype):9} {str(list(a.shape)):20} max|x| {np.abs(f).max() if f.size else 0:.4g}"
                    f"  mean {f.mean() if f.size else 0:.4g}")
    return "\n".join(rows)


def run(out: Path, graph_index: int, inputs: Path | None, outputs: Path | None, emulation: bool) -> Path:
    """Builds out/crate's run_graph (npu3, or the CPU emulation) and runs one graph of the model."""
    from furiosa_compiler.locks import npu_lock
    from furiosa_compiler.npu import NPU, build_run_graph

    graph = json.loads((out / "compiler" / "graph.json").read_text())["graphs"][graph_index]
    if inputs is None:  # random inputs, written next to the outputs so a run can be repeated
        inputs = out / f"inputs-{graph_index}.safetensors"
        tensors.save(inputs, fed_buffers(graph, None))
    outputs = outputs or out / f"outputs-{graph_index}{'-emulation' if emulation else ''}.safetensors"
    exe = build_run_graph(out / "crate", out / ("run_graph-emulation" if emulation else "run_graph"), npu=not emulation)
    cmd = [str(exe), str(graph_index), str(inputs.resolve()), str(outputs.resolve())]
    env = dict(os.environ, FURIOSA_VISIBLE_CHIPS=NPU)
    if emulation:
        proc = subprocess.run(cmd, cwd=out / "crate", env=env)
    else:
        with npu_lock():
            proc = subprocess.run(cmd, cwd=out / "crate", env=env)
    if proc.returncode:
        sys.exit(f"run_graph exited {proc.returncode}")
    print(f"{graph['name']} on {'the CPU emulation' if emulation else 'npu' + NPU}: {outputs}")
    print(summary(tensors.load(outputs)))
    return outputs


def trace(model: Path, out: Path, graph_index: int, inputs: Path | None, dump: Path | None, npu: bool,
          rtol: float, atol: float) -> bool:
    """Runs one graph launch by launch on the CPU reference (the raised kernels on the JIT) and, with `npu`, each
    launch also on npu3 from the reference's buffers; prints every launch's written buffers and the NPU's error,
    and with `dump` saves each launch's buffers (reference, and NPU) as safetensors."""
    result = compile_model(model, out)
    graph = result["graphs"][graph_index]
    kernels = l1.Kernels(out / "compiler", result["kernels"])
    launches = driver.load_model(model).graphs()[graph_index].launches  # the @triton.jit functions, for npu runs
    buffers = fed_buffers(graph, inputs)
    for name, info in graph["buffers"].items():
        buffers.setdefault(name, np.zeros(info["shape"], dtype=tensors.DTYPES[info["dtype"]]))
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        tensors.save(dump / "inputs.safetensors", {n: buffers[n] for n in buffers if graph["buffers"][n]["kind"] in KINDS_FED})
    ok = True
    print(f"{'#':>3}  {'kernel':28} {'buffer':20} {'max|ref|':>10} {'npu max err':>12} {'rel':>9}")
    for i, (launch, src) in enumerate(zip(graph["launches"], launches)):
        before = dict(buffers)
        _, address, written = kernels.get(launch["kernel"])
        args = [buffers[name] for name in launch["args"]]
        outs = [np.empty_like(args[w]) for w in written]
        tensors.call(address, args + outs)
        names = [launch["args"][w] for w in written]
        for name, a in zip(names, outs):
            buffers[name] = a
        got = run_launch_on_npu(src, before, names) if npu else {}
        for name in names:
            ref = buffers[name].astype(np.float64)
            scale = float(np.abs(ref).max()) if ref.size else 0.0
            cell = ""
            if name in got:
                err = float(np.abs(got[name].astype(np.float64) - ref).max()) if ref.size else 0.0
                good = err <= atol + rtol * scale
                ok &= good
                cell = f"{err:12.4g} {err / scale if scale else 0:9.2g}{'' if good else '  <-- exceeds tolerance'}"
            print(f"{i:3}  {launch['kernel']:28} {name:20} {scale:10.4g} {cell}")
        if dump:
            tensors.save(dump / f"{i:03d}_{launch['kernel']}.safetensors", {n: buffers[n] for n in names})
            if got:
                tensors.save(dump / f"{i:03d}_{launch['kernel']}.npu.safetensors", got)
    return ok


def run_launch_on_npu(launch, buffers: dict, names: list) -> dict:
    """One launch on npu3 through kernel[grid], its inputs the given buffers; the buffers it writes."""
    import torch

    from furiosa_compiler import npu

    torch_dtype = {v: k for k, v in npu.DTYPES.items()}
    args, host = {}, {}
    for param, value in launch.args.items():
        if driver.is_buffer(value):
            a = buffers[value.name]
            t = torch.empty(a.shape, dtype=torch_dtype[value.dtype])
            npu.store(t, a)
            args[param], host[value.name] = t, t
        else:
            args[param] = value
    npu.run(launch.kernel, launch.grid, args)
    return {name: npu.host(host[name]) for name in names}


def main():
    parser = argparse.ArgumentParser(prog="furiosa", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile", help="every kernel through every stage, and the furiosa-opt crate")
    c.add_argument("model", type=Path)
    c.add_argument("-o", "--out", type=Path, required=True)
    r = sub.add_parser("run", help="build the crate and run a graph on npu3 (or the CPU emulation)")
    r.add_argument("out", type=Path, help="a `furiosa compile` output dir")
    r.add_argument("--graph", type=int, default=0)
    r.add_argument("--inputs", type=Path, help="safetensors with the graph's inputs and weights (default: random)")
    r.add_argument("--outputs", type=Path)
    r.add_argument("--emulation", action="store_true", help="the furiosa-opt CPU emulation instead of npu3")
    t = sub.add_parser("trace", help="launch by launch: the CPU reference, and with --npu each launch on npu3")
    t.add_argument("model", type=Path)
    t.add_argument("-o", "--out", type=Path, required=True)
    t.add_argument("--graph", type=int, default=0)
    t.add_argument("--inputs", type=Path)
    t.add_argument("--npu", action="store_true", help="also run every launch on npu3 and compare")
    t.add_argument("--dump", type=Path, help="save every launch's buffers here")
    t.add_argument("--rtol", type=float, default=2e-2)
    t.add_argument("--atol", type=float, default=1e-2)
    a = parser.parse_args()
    if a.command == "compile":
        result = compile_model(a.model.resolve(), a.out)
        sys.exit(1 if any("error" in k for k in result["kernels"].values()) else 0)
    if a.command == "run":
        run(a.out, a.graph, a.inputs, a.outputs, a.emulation)
    else:
        sys.exit(0 if trace(a.model.resolve(), a.out, a.graph, a.inputs, a.dump, a.npu, a.rtol, a.atol) else 1)


if __name__ == "__main__":
    main()
