"""LeNet-5 in pure Triton, for 1x28x28 inputs.

conv(1->6, 5x5, pad 2) -> tanh -> avgpool 2 -> conv(6->16, 5x5) -> tanh -> avgpool 2 -> flatten ->
fc 400->120 -> tanh -> fc 120->84 -> tanh -> fc 84->10. Weights are ``state_dict`` tensors named
conv1/conv2/fc1/fc2/fc3 (``.weight``/``.bias``), read in their raw layouts. Activations are NHWC; the
last pool writes the (C, H, W)-flattened rows fc1 reads.
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
def tanh(x):
    """tanh from exp alone, so the IR needs no libdevice call: the odd extension of (1 - e^-2x) / (1 + e^-2x)."""
    t = tl.exp(-2.0 * tl.abs(x))
    y = (1.0 - t) / (1.0 + t)
    return tl.where(x < 0.0, -y, y)


@triton.jit
def conv2d_tanh_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, H, W, C, OH, OW, CO,
    sxn, sxh, sxw, sxc,
    PAD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, CO] = tanh(conv(x, w) + b), stride 1, as an implicit GEMM: rows are output pixels (NHWC
    order), columns output channels. x is read through its strides; w is the raw ``[CO, C, KH, KW]`` tensor."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < CO
    n = offs_m // (OH * OW)
    oh = (offs_m // OW) % OH
    ow = offs_m % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in range(KH):
        ih = oh - PAD + kh
        for kw in range(KW):
            iw = ow - PAD + kw
            ok = m_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_rows = x_ptr + n * sxn + ih * sxh + iw * sxw
            w_cols = w_ptr + offs_n * (C * KH * KW) + kh * KW + kw
            for c0 in range(0, C, BLOCK_K):
                c = c0 + offs_k
                c_mask = c < C
                a = tl.load(x_rows[:, None] + c[None, :] * sxc, mask=ok[:, None] & c_mask[None, :], other=0.0)
                b = tl.load(w_cols[None, :] + c[:, None] * (KH * KW), mask=c_mask[:, None] & n_mask[None, :], other=0.0)
                acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = tanh(acc + tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :])
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * CO + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def avgpool2d_kernel(
    x_ptr, y_ptr,
    P, H, W, C, OH, OW,
    syn, syh, syw, syc,
    K: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Average pooling over K x K windows with stride K on an NHWC x. y goes through its strides, so the
    last pool can write the (C, H, W)-flattened rows fc1 reads."""
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_p < P)[:, None] & (offs_c < C)[None, :]
    n = offs_p // (OH * OW)
    oh = (offs_p // OW) % OH
    ow = offs_p % OW
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            rows = ((n * H + oh * K + kh) * W + ow * K + kw) * C
            acc += tl.load(x_ptr + rows[:, None] + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
    out = y_ptr + (n * syn + oh * syh + ow * syw)[:, None] + offs_c[None, :] * syc
    tl.store(out, (acc / (K * K)).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    TANH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ w[N, K]^T + b, then tanh if ``TANH``; w is the raw ``nn.Linear`` weight."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if TANH:
        acc = tanh(acc)
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def conv2d_tanh(g, x, prefix, cout, k, pad):
    """``tanh(conv(x) + bias)`` with the checkpoint's ``<prefix>.weight`` / ``<prefix>.bias``; x is the NCHW
    input or an NHWC activation, the result is NHWC."""
    if x.kind == "input":
        n, c, h, w = x.shape
        sx = (c * h * w, w, 1, h * w)
    else:
        n, h, w, c = x.shape
        sx = (h * w * c, w * c, c, 1)
    oh, ow = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    y = g.buffer(f"{prefix}.out", x.dtype, (n, oh, ow, cout), "activation")
    m = n * oh * ow
    g.launch(
        conv2d_tanh_kernel, (triton.cdiv(m, 64), triton.cdiv(cout, 16)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (cout, c, k, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (cout,), "weight"), y_ptr=y,
        M=m, H=h, W=w, C=c, OH=oh, OW=ow, CO=cout,
        sxn=sx[0], sxh=sx[1], sxw=sx[2], sxc=sx[3],
        PAD=pad, KH=k, KW=k, BLOCK_M=64, BLOCK_N=16, BLOCK_K=16,
    )
    return y


def avgpool2d(g, x, name, k, flatten=False):
    """K x K average pooling of an NHWC x into NHWC, or into ``[N, C*OH*OW]`` rows in torch.flatten's
    (C, H, W) order when ``flatten``."""
    n, h, w, c = x.shape
    oh, ow = h // k, w // k
    if flatten:
        y = g.buffer(f"{name}.out", x.dtype, (n, c * oh * ow), "activation")
        sy = (c * oh * ow, ow, 1, oh * ow)
    else:
        y = g.buffer(f"{name}.out", x.dtype, (n, oh, ow, c), "activation")
        sy = (oh * ow * c, ow * c, c, 1)
    p = n * oh * ow
    g.launch(
        avgpool2d_kernel, (triton.cdiv(p, 32), triton.cdiv(c, 16)),
        x_ptr=x, y_ptr=y, P=p, H=h, W=w, C=c, OH=oh, OW=ow,
        syn=sy[0], syh=sy[1], syw=sy[2], syc=sy[3], K=k, BLOCK_P=32, BLOCK_C=16,
    )
    return y


def linear(g, x, prefix, n_out, tanh, name=None, kind="activation"):
    """``x @ weight.T + bias`` (then tanh) with the checkpoint's ``<prefix>.weight`` / ``<prefix>.bias``."""
    m, k = x.shape
    y = g.buffer(name or f"{prefix}.out", x.dtype, (m, n_out), kind)
    g.launch(
        linear_kernel, (triton.cdiv(m, 16), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (n_out, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (n_out,), "weight"), y_ptr=y,
        M=m, N=n_out, K=k, TANH=tanh, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return y


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32"):
    """LeNet-5 on ``image`` [N, 1, 28, 28] -> ``logits`` [N, 10]."""
    g = Graph(f"lenet5.forward(batch={batch}, dtype={dtype})")
    x = g.buffer("image", dtype, (batch, 1, 28, 28), "input")
    x = conv2d_tanh(g, x, "conv1", 6, k=5, pad=2)
    x = avgpool2d(g, x, "pool1", k=2)
    x = conv2d_tanh(g, x, "conv2", 16, k=5, pad=0)
    x = avgpool2d(g, x, "pool2", k=2, flatten=True)
    x = linear(g, x, "fc1", 120, tanh=True)
    x = linear(g, x, "fc2", 84, tanh=True)
    linear(g, x, "fc3", 10, tanh=False, name="logits", kind="output")
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
