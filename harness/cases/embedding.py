"""embedding: out[t] = table[ids[t]] with i32 token ids and a bf16 table [V, H]."""

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
def embedding_kernel(ids_ptr, table_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    """One program per token: out[t, :] = table[ids[t], :]."""
    t = tl.program_id(0)
    row = tl.load(ids_ptr + t)
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        tl.store(out_ptr + t * H + offs, tl.load(table_ptr + row * H + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

T = 8
V = 1024
H = 1024


def forward():
    g = Graph("embedding")
    ids = g.buffer("ids", "i32", (T,), "input")
    table = g.buffer("table", "bf16", (V, H), "weight")
    out = g.buffer("out", "bf16", (T, H), "output")
    g.launch(embedding_kernel, (T,), ids_ptr=ids, table_ptr=table, out_ptr=out, H=H, BLOCK_H=512)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
