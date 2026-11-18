"""heldout2_softmax_bwd: softmax backward dx = y * (dy - sum(dy * y)) on 33x500 fp16 rows (fp32 math), four rows
per program.

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
def softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """dx[r, :] = y[r, :] * (dy[r, :] - <dy[r, :], y[r, :]>) for BLOCK_R rows."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    offs = rows[:, None] * C + cols[None, :]
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(y * dy, axis=1)
    tl.store(dx_ptr + offs, (y * (dy - dot[:, None])).to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 33, 500


def backward():
    g = Graph("softmax_bwd")
    y = g.buffer("y", "fp16", (R, C), "input")
    dy = g.buffer("dy", "fp16", (R, C), "input")
    dx = g.buffer("dx", "fp16", (R, C), "output")
    g.launch(softmax_bwd_kernel, (triton.cdiv(R, 4),), y_ptr=y, dy_ptr=dy, dx_ptr=dx, R=R, C=C, BLOCK_R=4,
             BLOCK_C=512)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [backward()]
