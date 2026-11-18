"""heldout2_layernorm_bwd: layer norm backward on a 50x200 bf16 activation (fp32 math, fp32 gamma): dx one row
per program, then dgamma and dbeta as column sums in a second kernel that recomputes x_hat from the saved
mean and rstd.

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
def layer_norm_bwd_dx_kernel(x_ptr, dy_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr, N, BLOCK_N: tl.constexpr):
    """dx = rstd * (w*dy - x_hat * mean(w*dy*x_hat) - mean(w*dy)) with x_hat = (x - mean) * rstd."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(dx_ptr + row * N + cols, dx.to(tl.bfloat16), mask=mask)


@triton.jit
def layer_norm_bwd_dwdb_kernel(x_ptr, dy_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr, M, N,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """dw[c] = sum_r dy[r, c] * x_hat[r, c] and db[c] = sum_r dy[r, c] for BLOCK_N columns per program."""
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    dw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r0 in range(0, M, BLOCK_M):
        rows = r0 + tl.arange(0, BLOCK_M)
        rmask = rows < M
        mask = rmask[:, None] & cmask[None, :]
        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=rmask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        dw += dy * (x - mean[:, None]) * rstd[:, None]
        db += dy
    tl.store(dw_ptr + cols, tl.sum(dw, axis=0), mask=cmask)
    tl.store(db_ptr + cols, tl.sum(db, axis=0), mask=cmask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

M, N = 50, 200


def backward():
    g = Graph("layer_norm_bwd")
    x = g.buffer("x", "bf16", (M, N), "input")
    dy = g.buffer("dy", "bf16", (M, N), "input")
    w = g.buffer("gamma", "fp32", (N,), "weight")
    mean = g.buffer("mean", "fp32", (M,), "input")
    rstd = g.buffer("rstd", "fp32", (M,), "input")
    dx = g.buffer("dx", "bf16", (M, N), "output")
    dw = g.buffer("dgamma", "fp32", (N,), "output")
    db = g.buffer("dbeta", "fp32", (N,), "output")
    g.launch(layer_norm_bwd_dx_kernel, (M,), x_ptr=x, dy_ptr=dy, w_ptr=w, mean_ptr=mean, rstd_ptr=rstd, dx_ptr=dx,
             N=N, BLOCK_N=256)
    g.launch(layer_norm_bwd_dwdb_kernel, (triton.cdiv(N, 64),), x_ptr=x, dy_ptr=dy, mean_ptr=mean, rstd_ptr=rstd,
             dw_ptr=dw, db_ptr=db, M=M, N=N, BLOCK_M=16, BLOCK_N=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [backward()]
