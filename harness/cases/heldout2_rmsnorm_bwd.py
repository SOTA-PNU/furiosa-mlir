"""heldout2_rmsnorm_bwd: RMSNorm backward on a 70x320 fp16 activation (fp32 math): each program walks a band of
rows (the last band is short), writes dx and its partial dw sum; a second kernel sums the partials into dw.

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
def rms_norm_bwd_kernel(x_ptr, dy_ptr, w_ptr, dx_ptr, dw_partial_ptr, M, N, eps,
                        ROWS_PER_PROG: tl.constexpr, BLOCK_N: tl.constexpr):
    """For y = x * rstd * w with rstd = 1/sqrt(mean(x^2) + eps): dx = rstd * (w*dy - x * rstd^2 * mean(w*dy*x)),
    and dw_partial[pid] = sum of dy * x * rstd over this program's rows."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, M)
    for row in range(row_start, row_end):
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        wdy = w * dy
        dx = rstd * (wdy - x * (rstd * rstd * tl.sum(wdy * x, axis=0) / N))
        tl.store(dx_ptr + row * N + cols, dx.to(tl.float16), mask=mask)
        dw += dy * x * rstd
    tl.store(dw_partial_ptr + pid * N + cols, dw, mask=mask)


@triton.jit
def sum_partials_kernel(src_ptr, dst_ptr, R, N, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    """dst[c] = sum_r src[r, c], cast to dst's element type."""
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_R)
    mask = (rows[:, None] < R) & (cols[None, :] < N)
    part = tl.load(src_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    tl.store(dst_ptr + cols, tl.sum(part, axis=0).to(dst_ptr.dtype.element_ty), mask=cols < N)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

M, N, ROWS = 70, 320, 9
PROGS = triton.cdiv(M, ROWS)


def backward():
    g = Graph("rms_norm_bwd")
    x = g.buffer("x", "fp16", (M, N), "input")
    dy = g.buffer("dy", "fp16", (M, N), "input")
    w = g.buffer("weight", "fp16", (N,), "weight")
    dx = g.buffer("dx", "fp16", (M, N), "output")
    part = g.buffer("dw_partial", "fp32", (PROGS, N), "activation")
    dw = g.buffer("dweight", "fp16", (N,), "output")
    g.launch(rms_norm_bwd_kernel, (PROGS,), x_ptr=x, dy_ptr=dy, w_ptr=w, dx_ptr=dx, dw_partial_ptr=part, M=M, N=N,
             eps=1e-6, ROWS_PER_PROG=ROWS, BLOCK_N=512)
    g.launch(sum_partials_kernel, (triton.cdiv(N, 128),), src_ptr=part, dst_ptr=dw, R=PROGS, N=N, BLOCK_R=8,
             BLOCK_N=128)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [backward()]
