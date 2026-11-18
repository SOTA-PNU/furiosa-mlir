"""heldout3_upsample_bilinear: 2x bilinear upsampling with align_corners=False on an NCHW bf16 tensor
(1x6x13x10 -> 26x20), fp32 interpolation weights, one program per 8 output rows of one (n, c) plane.

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
def upsample_bilinear2x_kernel(x_ptr, y_ptr, H, W, OH, OW, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """y[nc, oh, ow] = x[nc] sampled bilinearly at ((oh + 0.5) / 2 - 0.5, (ow + 0.5) / 2 - 0.5), clamped at 0."""
    nc = tl.program_id(1)
    oh = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = tl.arange(0, BLOCK_W)
    sh = tl.maximum((oh.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    sw = tl.maximum((ow.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    h0 = sh.to(tl.int32)
    w0 = sw.to(tl.int32)
    h1 = tl.minimum(h0 + 1, H - 1)
    w1 = tl.minimum(w0 + 1, W - 1)
    lh = (sh - h0.to(tl.float32))[:, None]
    lw = (sw - w0.to(tl.float32))[None, :]
    mask = (oh < OH)[:, None] & (ow < OW)[None, :]
    base = x_ptr + nc * H * W
    x00 = tl.load(base + h0[:, None] * W + w0[None, :], mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(base + h0[:, None] * W + w1[None, :], mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(base + h1[:, None] * W + w0[None, :], mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(base + h1[:, None] * W + w1[None, :], mask=mask, other=0.0).to(tl.float32)
    y = (1.0 - lh) * ((1.0 - lw) * x00 + lw * x01) + lh * ((1.0 - lw) * x10 + lw * x11)
    tl.store(y_ptr + nc * OH * OW + oh[:, None] * OW + ow[None, :], y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C, H, W = 1, 6, 13, 10


def forward():
    g = Graph("upsample_bilinear")
    x = g.buffer("x", "bf16", (N, C, H, W), "input")
    y = g.buffer("y", "bf16", (N, C, 2 * H, 2 * W), "output")
    g.launch(upsample_bilinear2x_kernel, (triton.cdiv(2 * H, 8), N * C), x_ptr=x, y_ptr=y, H=H, W=W, OH=2 * H,
             OW=2 * W, BLOCK_H=8, BLOCK_W=32)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
