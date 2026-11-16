"""heldout_transpose: y = x^T for a 300x200 fp32 matrix in 32x64 tiles (pure data movement).

Held-out case: written apart from the corpus the compiler was developed on (harness/README.md).
"""

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
def transpose_kernel(x_ptr, y_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[C, R] = x[R, C]^T, one BLOCK_R x BLOCK_C tile of x per program."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    tile = tl.load(x_ptr + rows[:, None] * C + cols[None, :], mask=mask)
    tl.store(y_ptr + cols[:, None] * R + rows[None, :], tl.trans(tile), mask=tl.trans(mask))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 300, 200


def forward():
    g = Graph("transpose")
    x = g.buffer("x", "fp32", (R, C), "input")
    y = g.buffer("y", "fp32", (C, R), "output")
    g.launch(transpose_kernel, (triton.cdiv(R, 32), triton.cdiv(C, 64)), x_ptr=x, y_ptr=y, R=R, C=C, BLOCK_R=32, BLOCK_C=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
