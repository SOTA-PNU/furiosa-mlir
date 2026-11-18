"""heldout2_group_norm: group norm forward on NCHW fp32 (2x12x7x7) with 4 groups of 3 channels: program (n, g)
loads its group as a masked channels x pixels tile, takes mean and variance over it, then applies the
per-channel affine.

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
def group_norm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, HW, G, eps, CPG_BLOCK: tl.constexpr,
                          HW_BLOCK: tl.constexpr):
    """y[n, c] = (x[n, c] - mean) * rstd * w[c] + b[c], mean and rstd over group g = c // (C / G) of sample n."""
    n = tl.program_id(0)
    g = tl.program_id(1)
    cpg = C // G
    ci = tl.arange(0, CPG_BLOCK)
    c = g * cpg + ci
    p = tl.arange(0, HW_BLOCK)
    cmask = ci < cpg
    mask = cmask[:, None] & (p[None, :] < HW)
    offs = (n * C + c)[:, None] * HW + p[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    count = cpg * HW
    mean = tl.sum(tl.sum(x, axis=1), axis=0) / count
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(tl.sum(diff * diff, axis=1), axis=0) / count
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + c, mask=cmask, other=0.0)
    b = tl.load(b_ptr + c, mask=cmask, other=0.0)
    tl.store(y_ptr + offs, diff * rstd * w[:, None] + b[:, None], mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C, H, W, G = 2, 12, 7, 7, 4


def forward():
    g = Graph("group_norm")
    x = g.buffer("x", "fp32", (N, C, H, W), "input")
    w = g.buffer("gamma", "fp32", (C,), "weight")
    b = g.buffer("beta", "fp32", (C,), "weight")
    y = g.buffer("y", "fp32", (N, C, H, W), "output")
    g.launch(group_norm_fwd_kernel, (N, G), x_ptr=x, w_ptr=w, b_ptr=b, y_ptr=y, C=C, HW=H * W, G=G, eps=1e-5,
             CPG_BLOCK=4, HW_BLOCK=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
