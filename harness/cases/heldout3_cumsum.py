"""heldout3_cumsum: prefix sums along the rows of a 19x300 int32 matrix with tl.cumsum: inclusive, 4 rows per
program in one masked 4x512 tile; exclusive, one program per row stepping 128-column chunks with a running carry.

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
def cumsum_rows_kernel(x_ptr, y_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[r, c] = x[r, 0] + ... + x[r, c]."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows < R)[:, None] & (cols < C)[None, :]
    x = tl.load(x_ptr + rows[:, None] * C + cols[None, :], mask=mask, other=0)
    tl.store(y_ptr + rows[:, None] * C + cols[None, :], tl.cumsum(x, axis=1), mask=mask)


@triton.jit
def exclusive_cumsum_rows_kernel(x_ptr, y_ptr, C, BLOCK_C: tl.constexpr):
    """y[r, c] = x[r, 0] + ... + x[r, c - 1] (y[r, 0] = 0)."""
    row = tl.program_id(0)
    carry = 0
    for c0 in range(0, C, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < C
        x = tl.load(x_ptr + row * C + cols, mask=mask, other=0)
        tl.store(y_ptr + row * C + cols, carry + tl.cumsum(x, axis=0) - x, mask=mask)
        carry += tl.sum(x, axis=0)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 19, 300


def forward():
    g = Graph("cumsum")
    x = g.buffer("x", "i32", (R, C), "input")
    inc = g.buffer("inclusive", "i32", (R, C), "output")
    exc = g.buffer("exclusive", "i32", (R, C), "output")
    g.launch(cumsum_rows_kernel, (triton.cdiv(R, 4),), x_ptr=x, y_ptr=inc, R=R, C=C, BLOCK_R=4, BLOCK_C=512)
    g.launch(exclusive_cumsum_rows_kernel, (R,), x_ptr=x, y_ptr=exc, C=C, BLOCK_C=128)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
