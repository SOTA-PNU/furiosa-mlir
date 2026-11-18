"""heldout3_sliding_window_attention: causal attention forward with a sliding window on fp16 q, k, v (2 batches x 2
heads, 100 positions, head dim 64): query i attends to keys i - WINDOW < j <= i (WINDOW = 40). One program per block
of 32 queries of one head runs an online softmax over the key blocks its window reaches; fp32 accumulation, fp16 out.

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
def sliding_window_attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, WINDOW, sm_scale, HEAD_DIM: tl.constexpr,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """o[bh, i] = sum_j softmax_j(q[bh, i] . k[bh, j] * sm_scale) v[bh, j] over i - WINDOW < j <= i."""
    start_m = tl.program_id(0) * BLOCK_M
    base = tl.program_id(1) * N * HEAD_DIM
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_m[:, None] < N, other=0.0)
    # A finite start: a row whose window misses the first key block keeps l = 0 and acc = 0 there instead of NaN.
    m_i = tl.full((BLOCK_M,), -1.0e30, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    lo = tl.maximum(start_m - WINDOW + 1, 0) // BLOCK_N * BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M, N)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + base + offs_n[None, :] * HEAD_DIM + offs_d[:, None], mask=offs_n[None, :] < N, other=0.0)
        s = tl.dot(q, k) * sm_scale
        dist = offs_m[:, None] - offs_n[None, :]
        s = tl.where((dist >= 0) & (dist < WINDOW) & (offs_n[None, :] < N), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_n[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], o.to(tl.float16), mask=offs_m[:, None] < N)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

B, H, N, D, WINDOW = 2, 2, 100, 64, 40


def forward():
    g = Graph("sliding_window_attention")
    q = g.buffer("q", "fp16", (B, H, N, D), "input")
    k = g.buffer("k", "fp16", (B, H, N, D), "input")
    v = g.buffer("v", "fp16", (B, H, N, D), "input")
    o = g.buffer("o", "fp16", (B, H, N, D), "output")
    g.launch(sliding_window_attn_fwd_kernel, (triton.cdiv(N, 32), B * H), q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=o, N=N,
             WINDOW=WINDOW, sm_scale=D**-0.5, HEAD_DIM=D, BLOCK_M=32, BLOCK_N=32)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
