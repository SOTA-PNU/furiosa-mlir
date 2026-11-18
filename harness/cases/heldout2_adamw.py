"""heldout2_adamw: two fused AdamW steps (PyTorch's formulation, decoupled weight decay, bias correction) in place
on 3000 fp32 parameters with fp32 moments and bf16 gradients; 1-D blocks, the last one masked. The parameters and
moments are state, so they start at zero and step 2 continues from step 1.

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
def adamw_step_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n_elements, lr, beta1, beta2, eps, weight_decay,
                      bias_correction1, bias_correction2, BLOCK: tl.constexpr):
    """p *= 1 - lr*wd; m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    p = tl.load(p_ptr + offs, mask=mask)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    p = p * (1.0 - lr * weight_decay)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    denom = tl.sqrt(v) / tl.sqrt(bias_correction2) + eps
    p = p - (lr / bias_correction1) * m / denom
    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

NUMEL, BLOCK = 3000, 1024
LR, BETA1, BETA2, EPS, WD = 1e-3, 0.9, 0.999, 1e-8, 1e-2


def step(t):
    g = Graph(f"adamw_step{t}")
    p = g.buffer("param", "fp32", (NUMEL,), "state")
    grad = g.buffer("grad", "bf16", (NUMEL,), "input")
    m = g.buffer("exp_avg", "fp32", (NUMEL,), "state")
    v = g.buffer("exp_avg_sq", "fp32", (NUMEL,), "state")
    g.launch(adamw_step_kernel, (triton.cdiv(NUMEL, BLOCK),), p_ptr=p, g_ptr=grad, m_ptr=m, v_ptr=v, n_elements=NUMEL,
             lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, weight_decay=WD, bias_correction1=1 - BETA1**t,
             bias_correction2=1 - BETA2**t, BLOCK=BLOCK)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [step(1), step(2)]
