"""matmul: y = x @ w^T with x [M, K], w [N, K] (nn.Linear layout), bf16 operands, fp32 accumulation, bf16 out."""

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
def matmul_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=m_mask[:, None] & (k[None, :] < K), other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=(k[:, None] < K) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

M = 8
K = 1024
N = 1024


def forward():
    g = Graph("matmul")
    x = g.buffer("x", "bf16", (M, K), "input")
    w = g.buffer("weight", "bf16", (N, K), "weight")
    y = g.buffer("y", "bf16", (M, N), "output")
    g.launch(
        matmul_kernel, (triton.cdiv(M, 16), triton.cdiv(N, 64)),
        x_ptr=x, w_ptr=w, y_ptr=y, M=M, N=N, K=K, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
