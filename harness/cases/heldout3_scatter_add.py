"""heldout3_scatter_add: embedding backward, the rows of a 150x96 fp32 output gradient scatter-added into a 40x96
table by int32 token ids (repeated ids certain): blocks of 32 tokens x 64 columns, one tl.atomic_add per element.

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
def embedding_bwd_kernel(ids_ptr, dy_ptr, dw_ptr, T, D, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """dw[ids[t], d] += dy[t, d] for every token t."""
    toks = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    tmask = toks < T
    mask = tmask[:, None] & (cols[None, :] < D)
    rows = tl.load(ids_ptr + toks, mask=tmask, other=0)
    g = tl.load(dy_ptr + toks[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.atomic_add(dw_ptr + rows[:, None] * D + cols[None, :], g, mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

T, V, D = 150, 40, 96


def forward():
    g = Graph("scatter_add")
    ids = g.buffer("ids", "i32", (T,), "input")
    dy = g.buffer("dy", "fp32", (T, D), "input")
    dw = g.buffer("dw", "fp32", (V, D), "output")
    g.launch(embedding_bwd_kernel, (triton.cdiv(T, 32), triton.cdiv(D, 64)), ids_ptr=ids, dy_ptr=dy, dw_ptr=dw, T=T,
             D=D, BLOCK_T=32, BLOCK_D=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
