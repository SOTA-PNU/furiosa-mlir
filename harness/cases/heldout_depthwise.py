"""heldout_depthwise: depthwise 3x3 convolution, stride 1, padding 1, on NHWC fp32 (2x14x14x40), with
bias and ReLU6; pixel blocks x channel blocks x batch on a 3-D grid.

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
def depthwise_kernel(x_ptr, w_ptr, b_ptr, y_ptr, H, W, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[n, h, w, c] = relu6(b[c] + sum_{kh, kw} x[n, h + kh - 1, w + kw - 1, c] * w[c, kh, kw])."""
    n = tl.program_id(2)
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    h = offs_p // W
    w = offs_p % W
    pm = offs_p < H * W
    cm = offs_c < C
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for kh in range(3):
        for kw in range(3):
            ih = h + kh - 1
            iw = w + kw - 1
            ok = pm & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + ((n * H + ih) * W + iw)[:, None] * C + offs_c[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
            wv = tl.load(w_ptr + offs_c * 9 + kh * 3 + kw, mask=cm, other=0.0)
            acc += xv * wv[None, :]
    acc += tl.load(b_ptr + offs_c, mask=cm, other=0.0)[None, :]
    acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)
    tl.store(y_ptr + (n * H * W + offs_p)[:, None] * C + offs_c[None, :], acc, mask=pm[:, None] & cm[None, :])


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, H, W, C = 2, 14, 14, 40


def forward():
    g = Graph("depthwise")
    x = g.buffer("x", "fp32", (N, H, W, C), "input")
    w = g.buffer("weight", "fp32", (C, 3, 3), "weight")
    b = g.buffer("bias", "fp32", (C,), "weight")
    y = g.buffer("y", "fp32", (N, H, W, C), "output")
    g.launch(depthwise_kernel, (triton.cdiv(H * W, 64), triton.cdiv(C, 32), N), x_ptr=x, w_ptr=w, b_ptr=b, y_ptr=y,
             H=H, W=W, C=C, BLOCK_P=64, BLOCK_C=32)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
