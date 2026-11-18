"""heldout3_avgpool: 3x3 average pooling, stride 2, padding 1, count_include_pad=False, on an NCHW fp16 tensor
(2x8x15x21 -> 8x11), fp32 sums; a 3-D grid of (column block, row block, n*c) with 4x16 output tiles.

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
def avgpool_kernel(x_ptr, y_ptr, H, W, OH, OW, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """y[nc, oh, ow] = mean of x[nc] over the in-image part of the 3x3 window at (2*oh - 1, 2*ow - 1)."""
    ow = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    nc = tl.program_id(2)
    h0 = oh * 2 - 1
    w0 = ow * 2 - 1
    acc = tl.zeros((BLOCK_H, BLOCK_W), tl.float32)
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h0 + kh
            iw = w0 + kw
            ok = ((ih >= 0) & (ih < H))[:, None] & ((iw >= 0) & (iw < W))[None, :]
            acc += tl.load(x_ptr + nc * H * W + ih[:, None] * W + iw[None, :], mask=ok, other=0.0).to(tl.float32)
    # count_include_pad=False: divide by the number of window taps inside the image.
    taps_h = tl.minimum(h0 + 3, H) - tl.maximum(h0, 0)
    taps_w = tl.minimum(w0 + 3, W) - tl.maximum(w0, 0)
    y = acc / (taps_h[:, None] * taps_w[None, :]).to(tl.float32)
    out = (oh < OH)[:, None] & (ow < OW)[None, :]
    tl.store(y_ptr + nc * OH * OW + oh[:, None] * OW + ow[None, :], y.to(tl.float16), mask=out)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C, H, W = 2, 8, 15, 21
OH, OW = (H - 1) // 2 + 1, (W - 1) // 2 + 1


def forward():
    g = Graph("avgpool")
    x = g.buffer("x", "fp16", (N, C, H, W), "input")
    y = g.buffer("y", "fp16", (N, C, OH, OW), "output")
    g.launch(avgpool_kernel, (triton.cdiv(OW, 16), triton.cdiv(OH, 4), N * C), x_ptr=x, y_ptr=y, H=H, W=W, OH=OH,
             OW=OW, BLOCK_H=4, BLOCK_W=16)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
