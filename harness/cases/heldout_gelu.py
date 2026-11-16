"""heldout_gelu: y = gelu_tanh(x + bias) on a 40x768 bf16 matrix (fp32 math), 2-D tiles.

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
def bias_gelu_kernel(x_ptr, b_ptr, y_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """y = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 z^3))) with z = x + bias[col]."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    z = tl.load(x_ptr + rows[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    z += tl.load(b_ptr + cols, mask=cols < C, other=0.0).to(tl.float32)[None, :]
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    t = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    tl.store(y_ptr + rows[:, None] * C + cols[None, :], (0.5 * z * (1.0 + t)).to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 40, 768


def forward():
    g = Graph("gelu")
    x = g.buffer("x", "bf16", (R, C), "input")
    b = g.buffer("bias", "bf16", (C,), "weight")
    y = g.buffer("y", "bf16", (R, C), "output")
    g.launch(bias_gelu_kernel, (triton.cdiv(R, 16), triton.cdiv(C, 256)), x_ptr=x, b_ptr=b, y_ptr=y, R=R, C=C,
             BLOCK_R=16, BLOCK_C=256)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
