"""vector_add: out = x + y over 2048 fp32 elements."""

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
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N = 2048
BLOCK = 512


def forward():
    g = Graph("vector_add")
    x = g.buffer("x", "fp32", (N,), "input")
    y = g.buffer("y", "fp32", (N,), "input")
    out = g.buffer("out", "fp32", (N,), "output")
    g.launch(add_kernel, (triton.cdiv(N, BLOCK),), x_ptr=x, y_ptr=y, out_ptr=out, N=N, BLOCK=BLOCK)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
