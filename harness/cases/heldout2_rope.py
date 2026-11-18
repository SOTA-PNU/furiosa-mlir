"""heldout2_rope: rotary position embedding (half-split pairs, as in GPT-NeoX/Llama) applied to bf16 q
(2x24x6x64) and k (2x24x2x64, grouped heads) with fp32 cos/sin tables; one token per program, its heads as a
masked tile of rows.

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
def rope_qk_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, S, HQ, HK, HALF: tl.constexpr,
                   BLOCK_H: tl.constexpr):
    """With x = [x1, x2] split on the head dim: out = [x1*cos - x2*sin, x2*cos + x1*sin], cos and sin at the
    token's position s, for every q head and every k head of token b * S + s."""
    tok = tl.program_id(0)
    s = tok % S
    d = tl.arange(0, HALF)
    h = tl.arange(0, BLOCK_H)
    cos = tl.load(cos_ptr + s * HALF + d)[None, :]
    sin = tl.load(sin_ptr + s * HALF + d)[None, :]

    qmask = (h[:, None] < HQ) & (d[None, :] < HALF)
    q_offs = (tok * HQ + h[:, None]) * (2 * HALF) + d[None, :]
    q1 = tl.load(q_ptr + q_offs, mask=qmask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q_offs + HALF, mask=qmask, other=0.0).to(tl.float32)
    tl.store(qo_ptr + q_offs, (q1 * cos - q2 * sin).to(tl.bfloat16), mask=qmask)
    tl.store(qo_ptr + q_offs + HALF, (q2 * cos + q1 * sin).to(tl.bfloat16), mask=qmask)

    kmask = (h[:, None] < HK) & (d[None, :] < HALF)
    k_offs = (tok * HK + h[:, None]) * (2 * HALF) + d[None, :]
    k1 = tl.load(k_ptr + k_offs, mask=kmask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k_offs + HALF, mask=kmask, other=0.0).to(tl.float32)
    tl.store(ko_ptr + k_offs, (k1 * cos - k2 * sin).to(tl.bfloat16), mask=kmask)
    tl.store(ko_ptr + k_offs + HALF, (k2 * cos + k1 * sin).to(tl.bfloat16), mask=kmask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

B, S, HQ, HK, D = 2, 24, 6, 2, 64


def forward():
    g = Graph("rope")
    q = g.buffer("q", "bf16", (B, S, HQ, D), "input")
    k = g.buffer("k", "bf16", (B, S, HK, D), "input")
    cos = g.buffer("cos", "fp32", (S, D // 2), "input")
    sin = g.buffer("sin", "fp32", (S, D // 2), "input")
    qo = g.buffer("q_rot", "bf16", (B, S, HQ, D), "output")
    ko = g.buffer("k_rot", "bf16", (B, S, HK, D), "output")
    g.launch(rope_qk_kernel, (B * S,), q_ptr=q, k_ptr=k, cos_ptr=cos, sin_ptr=sin, qo_ptr=qo, ko_ptr=ko, S=S, HQ=HQ,
             HK=HK, HALF=D // 2, BLOCK_H=8)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward()]
