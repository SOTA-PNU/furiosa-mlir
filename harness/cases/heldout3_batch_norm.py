"""heldout3_batch_norm: batch norm in training mode on an NCHW fp32 tensor (4x6x9x9): one program per channel takes
the mean and the biased variance over N*H*W = 324 elements (two passes over blocks of 128), then a 2-D grid of
(pixel block, n*c) normalizes with the per-channel affine.

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
def bn_stats_kernel(x_ptr, mean_ptr, var_ptr, N, C, HW, BLOCK: tl.constexpr):
    """mean[c] and the biased var[c] of x[:, c, :, :] over its N*H*W elements."""
    c = tl.program_id(0)
    count = N * HW
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, count, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + (idx // HW * C + c) * HW + idx % HW, mask=idx < count, other=0.0)
    mean = tl.sum(acc, axis=0) / count
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, count, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < count
        x = tl.load(x_ptr + (idx // HW * C + c) * HW + idx % HW, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        acc += d * d
    tl.store(mean_ptr + c, mean)
    tl.store(var_ptr + c, tl.sum(acc, axis=0) / count)


@triton.jit
def bn_normalize_kernel(x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, y_ptr, C, HW, eps, BLOCK: tl.constexpr):
    """y[n, c] = (x[n, c] - mean[c]) / sqrt(var[c] + eps) * w[c] + b[c]."""
    nc = tl.program_id(1)
    c = nc % C
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < HW
    x = tl.load(x_ptr + nc * HW + p, mask=mask, other=0.0)
    scale = tl.load(w_ptr + c) * tl.rsqrt(tl.load(var_ptr + c) + eps)
    y = (x - tl.load(mean_ptr + c)) * scale + tl.load(b_ptr + c)
    tl.store(y_ptr + nc * HW + p, y, mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C, H, W = 4, 6, 9, 9


def forward():
    g = Graph("batch_norm")
    x = g.buffer("x", "fp32", (N, C, H, W), "input")
    w = g.buffer("gamma", "fp32", (C,), "weight")
    b = g.buffer("beta", "fp32", (C,), "weight")
    mean = g.buffer("mean", "fp32", (C,), "output")
    var = g.buffer("var", "fp32", (C,), "output")
    y = g.buffer("y", "fp32", (N, C, H, W), "output")
    g.launch(bn_stats_kernel, (C,), x_ptr=x, mean_ptr=mean, var_ptr=var, N=N, C=C, HW=H * W, BLOCK=128)
    g.launch(bn_normalize_kernel, (triton.cdiv(H * W, 64), N * C), x_ptr=x, mean_ptr=mean, var_ptr=var, w_ptr=w,
             b_ptr=b, y_ptr=y, C=C, HW=H * W, eps=1e-5, BLOCK=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
