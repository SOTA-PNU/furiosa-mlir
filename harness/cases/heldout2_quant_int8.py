"""heldout2_quant_int8: per-row symmetric int8 quantization of a 24x200 fp32 matrix (scale = max|x| / 127,
round half up, clamp to [-127, 127]) and dequantization on 2-D tiles. q holds int8 values in an i32 buffer:
the harness's buffer dtypes (tests/harness.py) have no i8.

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
def quantize_rows_kernel(x_ptr, q_ptr, scale_ptr, C, BLOCK_C: tl.constexpr):
    """scale[r] = max|x[r, :]| / 127 (1 for an all-zero row); q[r, c] = clamp(floor(x / scale + 0.5), -127, 127)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=mask, other=0.0)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    q = tl.clamp(tl.floor(x / scale + 0.5), -127.0, 127.0).to(tl.int8)
    tl.store(q_ptr + row * C + cols, q.to(tl.int32), mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def dequantize_rows_kernel(q_ptr, scale_ptr, y_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[r, c] = q[r, c] * scale[r]."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    q = tl.load(q_ptr + rows[:, None] * C + cols[None, :], mask=mask, other=0)
    scale = tl.load(scale_ptr + rows, mask=rows < R, other=0.0)
    tl.store(y_ptr + rows[:, None] * C + cols[None, :], q.to(tl.float32) * scale[:, None], mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, C = 24, 200


def forward():
    g = Graph("quant_int8")
    x = g.buffer("x", "fp32", (R, C), "input")
    q = g.buffer("q", "i32", (R, C), "output")
    scale = g.buffer("scale", "fp32", (R,), "output")
    y = g.buffer("x_dq", "fp32", (R, C), "output")
    g.launch(quantize_rows_kernel, (R,), x_ptr=x, q_ptr=q, scale_ptr=scale, C=C, BLOCK_C=256)
    g.launch(dequantize_rows_kernel, (triton.cdiv(R, 8), triton.cdiv(C, 64)), q_ptr=q, scale_ptr=scale, y_ptr=y,
             R=R, C=C, BLOCK_R=8, BLOCK_C=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
