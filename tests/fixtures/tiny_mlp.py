"""A two-layer MLP in pure Triton: the smallest model file, used to test the dumper and the harness."""

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
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    RELU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ w[N, K]^T + b, optionally ReLU."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def linear(g, x, prefix, n_out, relu, out_name, kind="activation"):
    M, K = x.shape
    y = g.buffer(out_name, "fp32", (M, n_out), kind)
    g.launch(
        linear_kernel, (triton.cdiv(M, 16), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", "fp32", (n_out, K), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", "fp32", (n_out,), "weight"), y_ptr=y,
        M=M, N=n_out, K=K, RELU=relu, BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
    )
    return y


def forward(batch=4):
    g = Graph(f"forward(batch={batch})")
    x = g.buffer("x", "fp32", (batch, 64), "input")
    h = linear(g, x, "fc1", 128, True, "h")
    linear(g, h, "fc2", 10, False, "logits", kind="output")
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
