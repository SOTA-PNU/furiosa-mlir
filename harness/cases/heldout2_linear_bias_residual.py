"""heldout2_linear_bias_residual: y = x @ w^T + bias + residual with bf16 x [96, 200], w [80, 200] (nn.Linear
layout), fp32 bias and bf16 residual and output; fp32 accumulation, pointer-advancing K loop with a masked tail.

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
def linear_bias_residual_kernel(x_ptr, w_ptr, b_ptr, r_ptr, y_ptr, M, N, K, BLOCK_M: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One BLOCK_M x BLOCK_N tile of y = x @ w^T + b + r; w's tile is read transposed as [BLOCK_K, BLOCK_N]."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = offs_k + k0 < K
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=kmask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    res = tl.load(r_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    y = acc + bias[None, :] + res
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

M, N, K = 96, 80, 200


def forward():
    g = Graph("linear_bias_residual")
    x = g.buffer("x", "bf16", (M, K), "input")
    w = g.buffer("weight", "bf16", (N, K), "weight")
    b = g.buffer("bias", "fp32", (N,), "weight")
    r = g.buffer("residual", "bf16", (M, N), "input")
    y = g.buffer("y", "bf16", (M, N), "output")
    g.launch(linear_bias_residual_kernel, (triton.cdiv(M, 32), triton.cdiv(N, 32)), x_ptr=x, w_ptr=w, b_ptr=b,
             r_ptr=r, y_ptr=y, M=M, N=N, K=K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=64)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
