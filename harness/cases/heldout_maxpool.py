"""heldout_maxpool: 3x3 max pooling, stride 2, padding 1, on an NCHW fp32 tensor (2x24x29x29), one
program per block of output pixels of one (n, c) plane.

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
def maxpool_kernel(x_ptr, y_ptr, H, W, OH, OW, BLOCK: tl.constexpr):
    """y[nc, oh, ow] = max of x[nc] over the 3x3 window at (2*oh - 1, 2*ow - 1); padding never wins."""
    nc = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oh = offs // OW
    ow = offs % OW
    valid = offs < OH * OW
    best = tl.full((BLOCK,), float("-inf"), tl.float32)
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = oh * 2 - 1 + kh
            iw = ow * 2 - 1 + kw
            ok = valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            v = tl.load(x_ptr + nc * H * W + ih * W + iw, mask=ok, other=float("-inf"))
            best = tl.maximum(best, v)
    tl.store(y_ptr + nc * OH * OW + offs, best, mask=valid)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

N, C, H, W = 2, 24, 29, 29
OH, OW = (H - 1) // 2 + 1, (W - 1) // 2 + 1


def forward():
    g = Graph("maxpool")
    x = g.buffer("x", "fp32", (N, C, H, W), "input")
    y = g.buffer("y", "fp32", (N, C, OH, OW), "output")
    g.launch(maxpool_kernel, (triton.cdiv(OH * OW, 64), N * C), x_ptr=x, y_ptr=y, H=H, W=W, OH=OH, OW=OW, BLOCK=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
