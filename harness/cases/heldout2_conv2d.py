"""heldout2_conv2d: direct 3x3 convolution, stride 1, padding 1, NCHW fp32 (2x4x12x10 -> 2x6x12x10) with bias, no
im2col: program (n, pixel block) accumulates every output channel as outer products of a weight column and a
shifted input row.

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
def conv2d_3x3_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C_in, H, W, C_out, BLOCK_CO: tl.constexpr,
                      BLOCK_P: tl.constexpr):
    """y[n, co, h, w] = b[co] + sum_{ci, kh, kw} x[n, ci, h + kh - 1, w + kw - 1] * w[co, ci, kh, kw]."""
    n = tl.program_id(0)
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_co = tl.arange(0, BLOCK_CO)
    oh = offs_p // W
    ow = offs_p % W
    pmask = offs_p < H * W
    comask = offs_co < C_out
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    for ci in range(0, C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih = oh + kh - 1
                iw = ow + kw - 1
                ok = pmask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xv = tl.load(x_ptr + ((n * C_in + ci) * H + ih) * W + iw, mask=ok, other=0.0)
                wv = tl.load(w_ptr + ((offs_co * C_in + ci) * 3 + kh) * 3 + kw, mask=comask, other=0.0)
                acc += wv[:, None] * xv[None, :]
    acc += tl.load(b_ptr + offs_co, mask=comask, other=0.0)[:, None]
    y_offs = (n * C_out + offs_co)[:, None] * (H * W) + offs_p[None, :]
    tl.store(y_ptr + y_offs, acc, mask=comask[:, None] & pmask[None, :])


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C_IN, H, W, C_OUT = 2, 4, 12, 10, 6


def forward():
    g = Graph("conv2d")
    x = g.buffer("x", "fp32", (N, C_IN, H, W), "input")
    w = g.buffer("weight", "fp32", (C_OUT, C_IN, 3, 3), "weight")
    b = g.buffer("bias", "fp32", (C_OUT,), "weight")
    y = g.buffer("y", "fp32", (N, C_OUT, H, W), "output")
    g.launch(conv2d_3x3_kernel, (N, triton.cdiv(H * W, 64)), x_ptr=x, w_ptr=w, b_ptr=b, y_ptr=y, C_in=C_IN, H=H, W=W,
             C_out=C_OUT, BLOCK_CO=8, BLOCK_P=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
