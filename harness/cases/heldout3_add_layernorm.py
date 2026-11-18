"""heldout3_add_layernorm: fused residual add + layer norm forward on 37 rows of 600 fp16 values: s = x + residual
(stored, fp16) and y = layer_norm(s) * gamma + beta (fp16) with fp32 statistics, 4 rows per program in one masked
4x1024 tile.

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
def add_layernorm_kernel(x_ptr, r_ptr, w_ptr, b_ptr, s_ptr, y_ptr, R, C, eps, BLOCK_R: tl.constexpr,
                         BLOCK_C: tl.constexpr):
    """s = x + r; y = (s - mean(s)) / sqrt(var(s) + eps) * w + b, statistics per row."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    mask = (rows < R)[:, None] & cmask[None, :]
    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(s_ptr + offs, s.to(tl.float16), mask=mask)
    mean = tl.sum(s, axis=1) / C
    d = tl.where(mask, s - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    y = d * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 37, 600


def forward():
    g = Graph("add_layernorm")
    x = g.buffer("x", "fp16", (R, C), "input")
    r = g.buffer("residual", "fp16", (R, C), "input")
    w = g.buffer("gamma", "fp16", (C,), "weight")
    b = g.buffer("beta", "fp16", (C,), "weight")
    s = g.buffer("sum", "fp16", (R, C), "output")
    y = g.buffer("y", "fp16", (R, C), "output")
    g.launch(add_layernorm_kernel, (triton.cdiv(R, 4),), x_ptr=x, r_ptr=r, w_ptr=w, b_ptr=b, s_ptr=s, y_ptr=y, R=R,
             C=C, eps=1e-5, BLOCK_R=4, BLOCK_C=1024)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
