"""ResNet-101 (torchvision layout) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
and are read through strides, and each BatchNorm is applied in its conv's epilogue from the raw
``weight``/``bias``/``running_mean``/``running_var``. Activations are NHWC.
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

ACT_NONE = 0
ACT_RELU = 1


@triton.jit
def conv2d_bn_kernel(
    x_ptr, w_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr, r_ptr, y_ptr,
    M, H, W, Cin, OH, OW, Cout,
    sxn, sxh, sxw, sxc,
    stride_h, stride_w, pad_h, pad_w, eps,
    KH: tl.constexpr, KW: tl.constexpr, HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    """y = act(bn(conv(x)) + r) as an implicit GEMM: rows are output pixels, columns output channels.

    x is logically [N, H, W, Cin] with strides (sxn, sxh, sxw, sxc); w is a contiguous
    [Cout, Cin, KH, KW]; r and y are contiguous NHWC [M, Cout].
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(Cout, BLOCK_N)
    in_group = GROUP_M * num_n
    first_m = (pid // in_group) * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % in_group) % group_m
    pid_n = (pid % in_group) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < Cout
    n_img = offs_m // (OH * OW)
    pix = offs_m % (OH * OW)
    ih0 = (pix // OW) * stride_h - pad_h
    iw0 = (pix % OW) * stride_w - pad_w

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in range(KH):
        ih = ih0 + kh
        h_ok = m_mask & (ih >= 0) & (ih < H)
        for kw in range(KW):
            iw = iw0 + kw
            ok = h_ok & (iw >= 0) & (iw < W)
            x_row = x_ptr + n_img * sxn + ih * sxh + iw * sxw
            w_tap = w_ptr + kh * KW + kw
            for c0 in range(0, Cin, BLOCK_K):
                c = c0 + offs_c
                c_mask = c < Cin
                a = tl.load(x_row[:, None] + c[None, :] * sxc, mask=ok[:, None] & c_mask[None, :], other=0.0)
                b = tl.load(
                    w_tap + c[:, None] * (KH * KW) + offs_n[None, :] * (Cin * KH * KW),
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="ieee")

    gamma = tl.load(g_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    acc = (acc - mean[None, :]) * (gamma / tl.sqrt(var + eps))[None, :] + beta[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    out_offs = offs_m[:, None] * Cout + offs_n[None, :]
    if HAS_RES:
        acc += tl.load(r_ptr + out_offs, mask=mask, other=0.0).to(tl.float32)
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + out_offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def maxpool2d_kernel(
    x_ptr, y_ptr, M, H, W, C, OH, OW, stride, pad,
    K: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """KxK max pooling of a contiguous NHWC x into a contiguous NHWC y; padding never wins."""
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_mask = offs_p < M
    c_mask = offs_c < C
    n_img = offs_p // (OH * OW)
    pix = offs_p % (OH * OW)
    ih0 = (pix // OW) * stride - pad
    iw0 = (pix % OW) * stride - pad
    acc = tl.full((BLOCK_P, BLOCK_C), float("-inf"), dtype=tl.float32)
    for kh in tl.static_range(K):
        ih = ih0 + kh
        h_ok = p_mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = iw0 + kw
            ok = h_ok & (iw >= 0) & (iw < W)
            x = tl.load(
                x_ptr + ((n_img * H + ih) * W + iw)[:, None] * C + offs_c[None, :],
                mask=ok[:, None] & c_mask[None, :], other=float("-inf"),
            )
            acc = tl.maximum(acc, x.to(tl.float32))
    mask = p_mask[:, None] & c_mask[None, :]
    tl.store(y_ptr + offs_p[:, None] * C + offs_c[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def avgpool_kernel(x_ptr, y_ptr, HW, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[n, c] = mean over the HW pixels of a contiguous NHWC x."""
    n = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for p0 in range(0, HW, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        x = tl.load(
            x_ptr + (n * HW + offs_p)[:, None] * C + offs_c[None, :],
            mask=(offs_p < HW)[:, None] & c_mask[None, :], other=0.0,
        )
        acc += x.to(tl.float32)
    mean = tl.sum(acc, axis=0) / HW
    tl.store(y_ptr + n * C + offs_c, mean.to(y_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ w[N, K]^T + b, all contiguous."""
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
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

BN_EPS = 1e-5
LAYERS = (3, 4, 23, 3)
WIDTHS = (64, 128, 256, 512)
EXPANSION = 4


def conv_bn(g, x, conv, bn, cout, k, stride, pad, act, out_name, residual=None, nchw=False):
    """Launches ``act(bn(conv(x)) + residual)``; ``x`` is NHWC, or NCHW when ``nchw``. Returns NHWC."""
    dtype = x.dtype
    if nchw:
        N, Cin, H, W = x.shape
        strides = (Cin * H * W, W, 1, H * W)
    else:
        N, H, W, Cin = x.shape
        strides = (H * W * Cin, W * Cin, Cin, 1)
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    M = N * OH * OW
    y = g.buffer(out_name, dtype, (N, OH, OW, cout), "activation")
    block_m = 128 if M >= 8192 else 64
    block_n = 64 if cout <= 64 else 128
    g.launch(
        conv2d_bn_kernel, (triton.cdiv(M, block_m) * triton.cdiv(cout, block_n),),
        num_warps=8 if block_m * block_n >= 128 * 128 else 4,
        x_ptr=x, w_ptr=g.buffer(f"{conv}.weight", dtype, (cout, Cin, k, k), "weight"),
        g_ptr=g.buffer(f"{bn}.weight", dtype, (cout,), "weight"),
        beta_ptr=g.buffer(f"{bn}.bias", dtype, (cout,), "weight"),
        mean_ptr=g.buffer(f"{bn}.running_mean", dtype, (cout,), "weight"),
        var_ptr=g.buffer(f"{bn}.running_var", dtype, (cout,), "weight"),
        r_ptr=residual if residual is not None else y, y_ptr=y,
        M=M, H=H, W=W, Cin=Cin, OH=OH, OW=OW, Cout=cout,
        sxn=strides[0], sxh=strides[1], sxw=strides[2], sxc=strides[3],
        stride_h=stride, stride_w=stride, pad_h=pad, pad_w=pad, eps=BN_EPS,
        KH=k, KW=k, HAS_RES=residual is not None, ACT=act,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=16 if Cin < 32 else 32, GROUP_M=8,
    )
    return y


def bottleneck(g, x, prefix, width, stride, downsample):
    """torchvision's Bottleneck: 1x1 -> 3x3 (strided) -> 1x1, plus the identity or a 1x1 downsample."""
    identity = x
    if downsample:
        identity = conv_bn(g, x, f"{prefix}.downsample.0", f"{prefix}.downsample.1", width * EXPANSION, 1, stride, 0,
                           ACT_NONE, f"{prefix}.downsample")
    h = conv_bn(g, x, f"{prefix}.conv1", f"{prefix}.bn1", width, 1, 1, 0, ACT_RELU, f"{prefix}.conv1")
    h = conv_bn(g, h, f"{prefix}.conv2", f"{prefix}.bn2", width, 3, stride, 1, ACT_RELU, f"{prefix}.conv2")
    return conv_bn(g, h, f"{prefix}.conv3", f"{prefix}.bn3", width * EXPANSION, 1, 1, 0, ACT_RELU, f"{prefix}.out",
                   residual=identity)


def forward(batch=1, dtype="fp32"):
    g = Graph(f"forward(batch={batch}, dtype={dtype})")
    image = g.buffer("image", dtype, (batch, 3, 224, 224), "input")
    x = conv_bn(g, image, "conv1", "bn1", 64, 7, 2, 3, ACT_RELU, "conv1", nchw=True)

    N, H, W, C = x.shape
    OH, OW = (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1
    pooled = g.buffer("maxpool", dtype, (N, OH, OW, C), "activation")
    g.launch(
        maxpool2d_kernel, (triton.cdiv(N * OH * OW, 32), triton.cdiv(C, 64)),
        x_ptr=x, y_ptr=pooled, M=N * OH * OW, H=H, W=W, C=C, OH=OH, OW=OW, stride=2, pad=1,
        K=3, BLOCK_P=32, BLOCK_C=64,
    )
    x = pooled

    cin = 64
    for layer, (blocks, width) in enumerate(zip(LAYERS, WIDTHS), start=1):
        for i in range(blocks):
            stride = 2 if layer > 1 and i == 0 else 1
            downsample = i == 0 and (stride != 1 or cin != width * EXPANSION)
            x = bottleneck(g, x, f"layer{layer}.{i}", width, stride, downsample)
            cin = width * EXPANSION

    N, H, W, C = x.shape
    features = g.buffer("avgpool", dtype, (N, C), "activation")
    g.launch(avgpool_kernel, (N, triton.cdiv(C, 64)), x_ptr=x, y_ptr=features, HW=H * W, C=C, BLOCK_P=32, BLOCK_C=64)

    logits = g.buffer("logits", dtype, (N, 1000), "output")
    g.launch(
        linear_kernel, (triton.cdiv(N, 16), triton.cdiv(1000, 64)),
        x_ptr=features, w_ptr=g.buffer("fc.weight", dtype, (1000, C), "weight"),
        b_ptr=g.buffer("fc.bias", dtype, (1000,), "weight"), y_ptr=logits,
        M=N, N=1000, K=C, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
