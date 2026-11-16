"""rmsnorm: y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, bf16 in and out, fp32 math (Qwen3's eps)."""

from dataclasses import dataclass, field

import triton
import triton.language as tl


# ---------------------------------------------------------------------------------------------------
# Launch graph: the model as kernel launches in execution order. scripts/dump_mlir.py compiles a
# graph to TTIR without torch or a GPU; tests/harness.py runs it on the GPU.
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Buf:
    """A device buffer. ``kind`` is "input", "weight" (a checkpoint tensor, by its checkpoint name),
    "state" (kept across graphs, e.g. a KV cache), "activation", or "output"."""

    name: str
    dtype: str  # Triton element type: "fp32", "fp16", "bf16", "i32", "i64"
    shape: tuple
    kind: str

    def data_ptr(self):
        """Lets Triton specialize a Buf as an aligned device pointer of ``dtype``."""
        return 0


@dataclass(frozen=True)
class Launch:
    kernel: object
    grid: tuple
    args: dict  # every kernel parameter by name: Buf, int, float, or constexpr value
    options: dict  # num_warps, num_stages


@dataclass
class Graph:
    """Kernel launches in execution order and every buffer they touch."""

    name: str
    buffers: dict = field(default_factory=dict)
    launches: list = field(default_factory=list)

    def buffer(self, name, dtype, shape, kind):
        buf = Buf(name, dtype, tuple(shape), kind)
        if self.buffers.setdefault(name, buf) != buf:
            raise ValueError(f"buffer {name} redeclared as {buf}, was {self.buffers[name]}")
        return buf

    def launch(self, kernel, grid, *, num_warps=4, num_stages=3, **args):
        unknown = set(args) - set(kernel.arg_names)
        if unknown:
            raise TypeError(f"{kernel.__name__} has no parameters {sorted(unknown)}")
        self.launches.append(Launch(kernel, tuple(grid), args, {"num_warps": num_warps, "num_stages": num_stages}))


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, H, eps, BLOCK_H: tl.constexpr):
    """One program per row of x [R, H]."""
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x = tl.load(x_ptr + r * H + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * H + offs, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

ROWS = 8
H = 1024
EPS = 1e-6


def forward():
    g = Graph("rmsnorm")
    x = g.buffer("x", "bf16", (ROWS, H), "input")
    w = g.buffer("weight", "bf16", (H,), "weight")
    y = g.buffer("y", "bf16", (ROWS, H), "output")
    g.launch(rmsnorm_kernel, (ROWS,), x_ptr=x, w_ptr=w, y_ptr=y, H=H, eps=EPS, BLOCK_H=H)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
