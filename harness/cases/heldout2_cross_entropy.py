"""heldout2_cross_entropy: cross-entropy loss forward over 45 rows of 1000 bf16 logits with int32 targets: per
row an online max/sum over 256-wide chunks gives logsumexp, minus the target logit; each row atomically adds
its share to the mean.

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
def cross_entropy_fwd_kernel(logits_ptr, target_ptr, loss_ptr, mean_ptr, R, V, BLOCK_V: tl.constexpr):
    """loss[r] = logsumexp(logits[r, :]) - logits[r, target[r]]; mean[0] += loss[r] / R."""
    row = tl.program_id(0)
    base = logits_ptr + row * V
    m = float("-inf")
    s = 0.0
    for v0 in range(0, V, BLOCK_V):
        cols = v0 + tl.arange(0, BLOCK_V)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    target = tl.load(target_ptr + row)
    x_t = tl.load(base + target).to(tl.float32)
    loss = m + tl.log(s) - x_t
    tl.store(loss_ptr + row, loss)
    tl.atomic_add(mean_ptr, loss / R)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

R, V = 45, 1000


def forward():
    g = Graph("cross_entropy")
    logits = g.buffer("logits", "bf16", (R, V), "input")
    target = g.buffer("target", "i32", (R,), "input")
    loss = g.buffer("loss", "fp32", (R,), "output")
    mean = g.buffer("mean_loss", "fp32", (1,), "output")
    g.launch(cross_entropy_fwd_kernel, (R,), logits_ptr=logits, target_ptr=target, loss_ptr=loss, mean_ptr=mean,
             R=R, V=V, BLOCK_V=256)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
