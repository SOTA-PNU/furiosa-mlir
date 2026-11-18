"""heldout3_fp8_matmul: c = (a @ b) * sa[:, None] * sb[None, :] with a and b in fp8 e4m3 (48x144 @ 144x72), per-row
and per-column fp32 scales, fp32 accumulation, bf16 output; 32x32x32 tiles, masked in M, N and K. The harness has
no fp8 buffers, so a and b are stored as bf16 and quantized to e4m3 on load (round to nearest even).

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
def fp8_scaled_matmul_kernel(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """c[m, n] = sa[m] * sb[n] * sum_k e4m3(a[m, k]) * e4m3(b[k, n])."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a.to(tl.float8e4nv), b.to(tl.float8e4nv), acc)
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    c = acc * sa[:, None] * sb[None, :]
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

M, N, K = 48, 72, 144


def forward():
    g = Graph("fp8_matmul")
    a = g.buffer("a", "bf16", (M, K), "input")
    b = g.buffer("b", "bf16", (K, N), "input")
    sa = g.buffer("scale_a", "fp32", (M,), "weight")
    sb = g.buffer("scale_b", "fp32", (N,), "weight")
    c = g.buffer("c", "bf16", (M, N), "output")
    g.launch(fp8_scaled_matmul_kernel, (triton.cdiv(M, 32), triton.cdiv(N, 32)), a_ptr=a, b_ptr=b, sa_ptr=sa,
             sb_ptr=sb, c_ptr=c, M=M, N=N, K=K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
