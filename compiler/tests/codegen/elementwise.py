"""Codegen test for the elementwise op class, independent of the raising pass: hand-written raised kernels
(compiler/README.md's linalg contract, structured form) for several sizes and Vector Engine ops go into one
generated crate, one graph each, which then runs on furiosa-opt's CPU backend and (if npu3 is idle) the
NPU and is compared with numpy.

    PYTHONPATH=compiler compiler/.venv/bin/python compiler/tests/codegen/elementwise.py [--npu] [--out DIR]
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from furiosa_compiler.crate import build_crate

ROOT = Path(__file__).resolve().parents[3]

# (name, size, body over block args %a [%b], numpy reference, number of inputs)
VARIANTS = [
    ("add_2048", 2048, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("add_1000", 1000, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("add_100", 100, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("add_8", 8, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("add_4096", 4096, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("add_98432", 98432, "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("max_2048", 2048, "%r = arith.maxnumf %a, %b : f32", np.maximum, 2),
    ("add_8x1000", (8, 1000), "%r = arith.addf %a, %b : f32", lambda x, y: x + y, 2),
    ("relu_1000", 1000, "%c = arith.constant 0.0 : f32\n      %r = arith.maximumf %a, %c : f32",
     lambda x: np.maximum(x, 0), 1),
    ("exp_2048", 2048, "%r = math.exp %a : f32", np.exp, 1),
    ("scale_4096", 4096, "%c = arith.constant 0.5 : f32\n      %r = arith.mulf %a, %c : f32", lambda x: x * 0.5, 1),
    ("sub_2048", 2048, "%c = arith.constant 1.5 : f32\n      %r = arith.subf %a, %c : f32", lambda x: x - 1.5, 1),
    ("neg_2048", 2048, "%r = arith.negf %a : f32", lambda x: -x, 1),
]


def shape_of(size) -> tuple:
    return size if isinstance(size, tuple) else (size,)


def linalg(size, body: str, inputs: int) -> str:
    shape = shape_of(size)
    t = f"tensor<{'x'.join(map(str, shape))}xf32>"
    dims = ", ".join(f"d{i}" for i in range(len(shape)))
    args = ", ".join(f'%p{i}: {t} loc("in{i}")' for i in range(inputs))
    ins = ", ".join(f"%p{i}" for i in range(inputs))
    blocks = ["%a: f32", "%b: f32"][:inputs]
    maps = ", ".join([f"affine_map<({dims}) -> ({dims})>"] * (inputs + 1))
    return f"""module {{
  func.func @elementwise({args}, %out: {t} loc("out")) -> {t} attributes {{furiosa.written = [{inputs}]}} {{
    %init = tensor.empty() : {t}
    %r = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{", ".join(['"parallel"'] * len(shape))}]}}
        ins({ins} : {", ".join([t] * inputs)}) outs(%init : {t}) {{
    ^bb0({", ".join(blocks)}, %o: f32):
      {body}
      linalg.yield %r : f32
    }} -> {t}
    return %r : {t}
  }}
}}
"""


def save_safetensors(path: Path, tensors: dict):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def load_safetensors(path: Path) -> dict:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(raw[8 + n + start : 8 + n + end], dtype=np.float32).reshape(info["shape"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "harness" / "out" / "codegen_elementwise")
    parser.add_argument("--npu", action="store_true", help="also run on npu3 when it is idle")
    args = parser.parse_args()
    out = args.out
    compiler_out = out / "compiler"
    rng = np.random.default_rng(0)

    kernels, graphs, inputs = {}, [], []
    for g, (name, size, body, ref, n_in) in enumerate(VARIANTS):
        symbol = f"elementwise_{name}"
        kdir = compiler_out / "kernels" / symbol
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "linalg.mlir").write_text(linalg(size, body, n_in))
        args_ = [f"x{g}_{i}" for i in range(n_in)] + [f"y{g}"]
        kernels[symbol] = {"kernel": "elementwise", "pointer_args": [f"in{i}" for i in range(n_in)] + ["out"],
                           "written": [n_in], "linalg": f"kernels/{symbol}/linalg.mlir"}
        buffers = {b: {"kind": "input", "dtype": "fp32", "shape": list(shape_of(size))} for b in args_[:-1]}
        buffers[args_[-1]] = {"kind": "output", "dtype": "fp32", "shape": list(shape_of(size))}
        graphs.append({"name": name, "buffers": buffers, "launches": [{"kernel": symbol, "args": args_, "grid": [1, 1, 1]}]})
        xs = [rng.standard_normal(shape_of(size)).astype(np.float32) for _ in range(n_in)]
        inputs.append((xs, ref(*xs).astype(np.float32)))
        save_safetensors(out / f"in{g}.safetensors", dict(zip(args_[:-1], xs)))
    result = {"model": str(out / "elementwise_test.py"), "kernels": kernels, "graphs": graphs}
    crate = build_crate(result, compiler_out, out / "crate")

    env = dict(os.environ, PATH=f"{ROOT / '.tools' / 'bin'}:{os.environ['PATH']}", LIBCLANG_PATH="/usr/lib/llvm-18/lib")
    modes = ["cpu"]
    occupancy = Path("/sys/class/rngd_mgmt/rngd!npu3mgmt/pe_occupancy")
    if args.npu and occupancy.exists() and set(occupancy.read_text().split()) == {"0"}:
        modes.append("npu")
    failures = 0
    for mode in modes:
        run = ["cargo", "furiosa-opt", "run"] if mode == "npu" else ["cargo", "run"]
        menv = dict(env, CARGO_TARGET_DIR=str(crate / "target" / mode))
        if mode == "npu":
            menv["FURIOSA_VISIBLE_CHIPS"] = "3"
        build = subprocess.run(run[:-1] + ["build", "--release", "--bin", "run_graph"], cwd=crate, env=menv,
                               capture_output=True, text=True)
        if build.returncode:
            print(f"{mode}: build failed\n{build.stderr[-3000:]}")
            failures += 1
            continue
        for g, (name, *_rest) in enumerate(VARIANTS):
            path = out / f"out{g}_{mode}.safetensors"
            proc = subprocess.run(run + ["--release", "--bin", "run_graph", "--", str(g), str(out / f"in{g}.safetensors"),
                                         str(path)], cwd=crate, env=menv, capture_output=True, text=True)
            if proc.returncode:
                print(f"{mode} {name}: run failed: {proc.stderr.strip().splitlines()[-1:]}")
                failures += 1
                continue
            got = load_safetensors(path)[f"y{g}"]
            want = inputs[g][1]
            err = float(np.max(np.abs(got - want)))
            ok = err <= 1e-5 + 1e-5 * float(np.max(np.abs(want)))  # exp is the Vector Engine's, not libm's
            failures += not ok
            print(f"{mode} {name}: {'PASS' if ok else 'FAIL'} max err {err:.3g}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
