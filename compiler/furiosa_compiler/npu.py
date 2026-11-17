"""Runs one Triton launch, ``kernel[grid](...)``, on the NPU: the furiosa Triton driver's launcher calls
``run``. The launch is specialized (scalars as constexprs, the grid, every tensor's dtype and shape),
compiled through the whole pipeline into a crate, built once per specialization, and run on the NPU;
the tensors the kernel writes are copied back. Nothing falls back to the CPU: a kernel the compiler
cannot map raises with the compiler's diagnostic.
"""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

from furiosa_compiler import driver, tensors
from furiosa_compiler.crate import build_crate
from furiosa_compiler.locks import CACHE, npu_lock, target_slot

ROOT = Path(__file__).resolve().parents[2]
NPU = os.environ.get("FURIOSA_VISIBLE_CHIPS", "3")  # the only NPU this host may open
OCCUPANCY = Path(f"/sys/class/rngd_mgmt/rngd!npu{NPU}mgmt/pe_occupancy")
DTYPES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.int32: "i32", torch.int64: "i64",
          torch.int8: "i8", torch.float8_e4m3fn: "fp8e4nv"}
# dtypes numpy has no native form of: the integer view torch hands over, and the ml_dtypes type to read it as.
VIEWS = {torch.bfloat16: (torch.int16, np.int16, ml_dtypes.bfloat16),
         torch.float8_e4m3fn: (torch.uint8, np.uint8, ml_dtypes.float8_e4m3fn)}


@dataclass(frozen=True)
class Buf:
    """A launch's tensor as the compiler sees it; every one is read and written back (kind "state")."""

    name: str
    dtype: str
    shape: tuple
    kind: str = "state"


@dataclass
class Launch:
    kernel: object
    grid: tuple
    args: dict
    options: dict = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    buffers: dict
    launches: list


def host(t: torch.Tensor) -> np.ndarray:
    t = t.detach().contiguous()
    if t.dtype in VIEWS:
        bits, _, kind = VIEWS[t.dtype]
        return t.view(bits).numpy().view(kind)
    return t.numpy()


def store(t: torch.Tensor, a: np.ndarray):
    if t.dtype in VIEWS:
        _, bits, _ = VIEWS[t.dtype]
        a = a.view(bits)
        t.copy_(torch.from_numpy(a).view(t.dtype))
    else:
        t.copy_(torch.from_numpy(a))


def deps_binary(uplifted: Path) -> Path:
    """The hashed deps/ file a cargo-uplifted binary is a hard link to: unique per crate."""
    inode = uplifted.stat().st_ino
    return next(p for p in (uplifted.parent / "deps").glob(f"{uplifted.name}-*")
                if not p.suffix and p.stat().st_ino == inode)


def build_run_graph(crate: Path, exe: Path, npu: bool = True) -> Path:
    """Builds the crate's run_graph for npu3 (cargo furiosa-opt) or the CPU emulation (cargo) in a target slot
    and copies it to `exe`: every crate's binary is uplifted to target/release/run_graph, and a crate with the name
    of another in the slot would share its deps/ binary, so the copy is resolved while the slot is held."""
    # The pinned furiosa-opt tools: .tools/bin in a checkout, bin/ in a binary package.
    env = dict(os.environ, FURIOSA_VISIBLE_CHIPS=NPU,
               PATH=f"{ROOT / '.tools' / 'bin'}:{ROOT / 'bin'}:{os.environ['PATH']}")
    env.setdefault("LIBCLANG_PATH", "/usr/lib/llvm-18/lib")
    cmd = (["cargo", "furiosa-opt", "build"] if npu else ["cargo", "build"])
    cmd += ["--release", "--bin", "run_graph", "--message-format=json-render-diagnostics"]
    with target_slot(CACHE / ("targets" if npu else "targets-cpu"), crate) as target:
        env["CARGO_TARGET_DIR"] = str(target)
        proc = subprocess.run(cmd, cwd=crate, env=env, capture_output=True, text=True)
        uplifted = next((json.loads(line)["executable"] for line in proc.stdout.splitlines()
                         if line.startswith("{") and '"executable":"' in line.replace(" ", "")), None)
        if proc.returncode or not uplifted:
            raise RuntimeError(f"{crate}: {'NPU' if npu else 'emulation'} build failed:\n{proc.stderr[-2000:]}")
        shutil.copy2(deps_binary(Path(uplifted)), exe)
    return exe


def executable(launch: Launch) -> tuple[Path, Path]:
    """The built run_graph for this launch's specialization (built on first use) and its work dir."""
    from triton.runtime.cache import triton_key  # libtriton: raising and codegen

    src, options, _, _ = driver.specialize(launch)
    compiler = triton_key() + "".join(p.read_text() for p in sorted(Path(__file__).parent.glob("*.py")))
    key = repr((compiler, launch.kernel.cache_key, sorted(src.constants.items(), key=str), src.signature,
                options["grid"], options["buffers"]))
    work = CACHE / f"{launch.kernel.__name__}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one build per specialization, whatever process asks
        record = work / "executable"
        if record.exists() and Path(record.read_text()).exists():
            return Path(record.read_text()), work
        graph = Graph(launch.kernel.__name__, {b.name: b for b in launch.args.values() if isinstance(b, Buf)}, [launch])
        # The crate is named after the work dir (kernel and specialization hash): every specialization of a kernel
        # is its own workspace, so with one name they would share cargo's unit hashes, and a build in a target slot
        # could take another specialization's binary as fresh.
        result = driver.compile_graphs([graph], work.name, work / "out")
        build_crate(result, work / "out", work / "crate")
        exe = build_run_graph(work / "crate", work / "run_graph")
        record.write_text(str(exe))
        return exe, work


def run(kernel, grid: tuple, args: dict):
    """kernel[grid](**args) on the NPU; tensor arguments are host tensors and receive the results."""
    bufs = {name: Buf(name, DTYPES[t.dtype], tuple(t.shape)) for name, t in args.items() if isinstance(t, torch.Tensor)}
    launch = Launch(kernel, tuple(grid), {name: bufs.get(name, value) for name, value in args.items()})
    exe, work = executable(launch)
    inputs, outputs = work / f"in-{os.getpid()}.safetensors", work / f"out-{os.getpid()}.safetensors"
    tensors.save(inputs, {name: host(args[name]) for name in bufs})
    with npu_lock():
        for _ in range(180):  # npu3 may be busy with another user's run: wait rather than fail
            if not OCCUPANCY.exists() or set(OCCUPANCY.read_text().split()) <= {"0"}:  # no status file: no wait
                break
            time.sleep(10)
        proc = subprocess.run([str(exe), "0", str(inputs), str(outputs)], cwd=work / "crate",
                              env=dict(os.environ, FURIOSA_VISIBLE_CHIPS=NPU), capture_output=True, text=True)
    if proc.returncode or not outputs.exists():
        raise RuntimeError(f"{kernel.__name__}: NPU run failed:\n{proc.stderr[-2000:]}")
    for name, array in tensors.load(outputs).items():
        store(args[name], array)
    inputs.unlink()
    outputs.unlink()
