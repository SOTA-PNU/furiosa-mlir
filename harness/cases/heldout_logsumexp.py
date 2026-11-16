"""heldout_logsumexp: y[r] = log(sum_c exp(x[r, c])) over 37 rows of 1000 fp32 values, as an online
(running max and rescaled sum) loop over column blocks.

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
def logsumexp_kernel(x_ptr, y_ptr, C, BLOCK: tl.constexpr):
    """One row per program: per lane, a running max m and sum s of exp(v - m), combined at the end."""
    row = tl.program_id(0)
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK,), tl.float32)
    for c0 in range(0, C, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * C + cols, mask=cols < C, other=float("-inf"))
        m_new = tl.maximum(m, v)
        s = s * tl.exp(m - m_new) + tl.exp(v - m_new)
        m = m_new
    top = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - top), axis=0)
    tl.store(y_ptr + row, top + tl.log(total))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 37, 1000


def forward():
    g = Graph("logsumexp")
    x = g.buffer("x", "fp32", (R, C), "input")
    y = g.buffer("y", "fp32", (R,), "output")
    g.launch(logsumexp_kernel, (R,), x_ptr=x, y_ptr=y, C=C, BLOCK=256)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
