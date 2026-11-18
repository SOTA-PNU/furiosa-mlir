"""heldout3_moe_topk_gating: mixture-of-experts router for 77 tokens over 12 experts (fp32 logits): a softmax over
the experts, then the top-2 expert indices (int32) and their probabilities per token, 16 tokens per program.

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
def topk_gating_kernel(logits_ptr, idx_ptr, w_ptr, T, E, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr):
    """p = softmax(logits[t, :]); idx[t, :] = the experts of the two largest p, w[t, :] = those p."""
    toks = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    experts = tl.arange(0, BLOCK_E)
    tmask = toks < T
    mask = tmask[:, None] & (experts[None, :] < E)
    x = tl.load(logits_ptr + toks[:, None] * E + experts[None, :], mask=mask, other=float("-inf"))
    e = tl.exp(x - tl.max(x, axis=1)[:, None])
    p = e / tl.sum(e, axis=1)[:, None]
    i1 = tl.argmax(p, axis=1)
    w1 = tl.max(p, axis=1)
    p = tl.where(experts[None, :] == i1[:, None], -1.0, p)
    i2 = tl.argmax(p, axis=1)
    w2 = tl.max(p, axis=1)
    tl.store(idx_ptr + toks * 2, i1, mask=tmask)
    tl.store(idx_ptr + toks * 2 + 1, i2, mask=tmask)
    tl.store(w_ptr + toks * 2, w1, mask=tmask)
    tl.store(w_ptr + toks * 2 + 1, w2, mask=tmask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

T, E = 77, 12


def forward():
    g = Graph("moe_topk_gating")
    logits = g.buffer("router_logits", "fp32", (T, E), "input")
    idx = g.buffer("topk_idx", "i32", (T, 2), "output")
    w = g.buffer("topk_w", "fp32", (T, 2), "output")
    g.launch(topk_gating_kernel, (triton.cdiv(T, 16),), logits_ptr=logits, idx_ptr=idx, w_ptr=w, T=T, E=E,
             BLOCK_T=16, BLOCK_E=16)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
