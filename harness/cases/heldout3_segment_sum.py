"""heldout3_segment_sum: sums the rows of a 200x48 bf16 matrix into 13 segments given sorted int32 segment offsets
(a CSR row pointer), fp32 sums. Golden inputs are drawn independently, so the offsets are made on the device from
per-segment lengths in [0, 16) (at most 195 rows; the rest belong to no segment) by a first launch; the segment
launch loops over its rows in blocks of 8 between the offsets it loads.

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
def lengths_to_offsets_kernel(len_ptr, off_ptr, S, BLOCK_S: tl.constexpr):
    """off[s] = len[0] + ... + len[s - 1] for s = 0..S."""
    s = tl.arange(0, BLOCK_S)
    lens = tl.load(len_ptr + s, mask=s < S, other=0)
    off = tl.sum(tl.where(s[None, :] < s[:, None], lens[None, :], 0), axis=1)
    tl.store(off_ptr + s, off, mask=s <= S)


@triton.jit
def segment_sum_kernel(x_ptr, off_ptr, y_ptr, D, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    """y[s, :] = x[off[s], :] + ... + x[off[s + 1] - 1, :] (zero for an empty segment)."""
    seg = tl.program_id(0)
    start = tl.load(off_ptr + seg)
    end = tl.load(off_ptr + seg + 1)
    cols = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for r0 in range(start, end, BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)
        mask = (rows < end)[:, None] & (cols < D)[None, :]
        x = tl.load(x_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0)
        acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(y_ptr + seg * D + cols, acc, mask=cols < D)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, D, S = 200, 48, 13


def forward():
    g = Graph("segment_sum")
    x = g.buffer("x", "bf16", (R, D), "input")
    lengths = g.buffer("seg_len", "i32", (S,), "input")
    offsets = g.buffer("seg_off", "i32", (S + 1,), "output")
    y = g.buffer("y", "fp32", (S, D), "output")
    g.launch(lengths_to_offsets_kernel, (1,), len_ptr=lengths, off_ptr=offsets, S=S, BLOCK_S=16)
    g.launch(segment_sum_kernel, (S,), x_ptr=x, off_ptr=offsets, y_ptr=y, D=D, BLOCK_R=8, BLOCK_D=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
