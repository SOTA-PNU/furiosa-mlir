"""MobileNetV3-Large (torchvision layout) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
(depthwise ``[C, 1, KH, KW]``) and are read through strides, and each BatchNorm is applied in its conv's
epilogue from the raw ``weight``/``bias``/``running_mean``/``running_var``. Squeeze-and-excitation scales
the projection conv's input as it loads. Activations are NHWC.
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
ACT_HARDSWISH = 2
ACT_HARDSIGMOID = 3


@triton.jit
def activation(x, ACT: tl.constexpr):
    """The ACT_* activation of an fp32 tile, as torch.nn defines it."""
    if ACT == 1:
        x = tl.maximum(x, 0.0)
    elif ACT == 2:
        x = x * (tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0)
    elif ACT == 3:
        x = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    return x


@triton.jit
def batch_norm(acc, offs_n, n_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps):
    """Eval-mode BatchNorm over the columns of an fp32 tile, from its raw parameters."""
    gamma = tl.load(g_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    return (acc - mean[None, :]) * (gamma / tl.sqrt(var + eps))[None, :] + beta[None, :]


@triton.jit
def conv2d_bn_kernel(
    x_ptr, w_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr, r_ptr, s_ptr, y_ptr,
    M, H, W, Cin, OH, OW, Cout,
    sxn, sxh, sxw, sxc,
    stride_h, stride_w, pad_h, pad_w, eps,
    KH: tl.constexpr, KW: tl.constexpr, HAS_RES: tl.constexpr, HAS_SCALE: tl.constexpr, ACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    """y = act(bn(conv(x * s)) + r) as an implicit GEMM: rows are output pixels, columns output channels.

    x is logically [N, H, W, Cin] with strides (sxn, sxh, sxw, sxc); w is a contiguous
    [Cout, Cin, KH, KW]; s is a contiguous [N, Cin] input scale; r and y are contiguous NHWC [M, Cout].
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
                if HAS_SCALE:
                    s = tl.load(s_ptr + n_img[:, None] * Cin + c[None, :], mask=m_mask[:, None] & c_mask[None, :], other=0.0)
                    a = (a.to(tl.float32) * s.to(tl.float32)).to(w_ptr.dtype.element_ty)
                b = tl.load(
                    w_tap + c[:, None] * (KH * KW) + offs_n[None, :] * (Cin * KH * KW),
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="ieee")

    acc = batch_norm(acc, offs_n, n_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps)
    mask = m_mask[:, None] & n_mask[None, :]
    out_offs = offs_m[:, None] * Cout + offs_n[None, :]
    if HAS_RES:
        acc += tl.load(r_ptr + out_offs, mask=mask, other=0.0).to(tl.float32)
    acc = activation(acc, ACT)
    tl.store(y_ptr + out_offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def depthwise_bn_kernel(
    x_ptr, w_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr, y_ptr,
    M, H, W, C, OH, OW, stride, pad, eps,
    K: tl.constexpr, ACT: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """y = act(bn(depthwise_conv(x))) for contiguous NHWC x and y and a contiguous [C, 1, K, K] w."""
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_mask = offs_p < M
    c_mask = offs_c < C
    n_img = offs_p // (OH * OW)
    pix = offs_p % (OH * OW)
    ih0 = (pix // OW) * stride - pad
    iw0 = (pix % OW) * stride - pad
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for kh in tl.static_range(K):
        ih = ih0 + kh
        h_ok = p_mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = iw0 + kw
            ok = h_ok & (iw >= 0) & (iw < W)
            x = tl.load(
                x_ptr + ((n_img * H + ih) * W + iw)[:, None] * C + offs_c[None, :],
                mask=ok[:, None] & c_mask[None, :], other=0.0,
            )
            w = tl.load(w_ptr + offs_c * (K * K) + kh * K + kw, mask=c_mask, other=0.0)
            acc += x.to(tl.float32) * w.to(tl.float32)[None, :]
    acc = activation(batch_norm(acc, offs_c, c_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps), ACT)
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
    ACT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ w[N, K]^T + b), all contiguous."""
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
    acc = activation(acc, ACT)
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

BN_EPS = 1e-3
# torchvision's mobilenet_v3_large: (in, kernel, expanded, out, use_se, activation, stride) per block.
BLOCKS = (
    (16, 3, 16, 16, False, ACT_RELU, 1),
    (16, 3, 64, 24, False, ACT_RELU, 2),
    (24, 3, 72, 24, False, ACT_RELU, 1),
    (24, 5, 72, 40, True, ACT_RELU, 2),
    (40, 5, 120, 40, True, ACT_RELU, 1),
    (40, 5, 120, 40, True, ACT_RELU, 1),
    (40, 3, 240, 80, False, ACT_HARDSWISH, 2),
    (80, 3, 200, 80, False, ACT_HARDSWISH, 1),
    (80, 3, 184, 80, False, ACT_HARDSWISH, 1),
    (80, 3, 184, 80, False, ACT_HARDSWISH, 1),
    (80, 3, 480, 112, True, ACT_HARDSWISH, 1),
    (112, 3, 672, 112, True, ACT_HARDSWISH, 1),
    (112, 5, 672, 160, True, ACT_HARDSWISH, 2),
    (160, 5, 960, 160, True, ACT_HARDSWISH, 1),
    (160, 5, 960, 160, True, ACT_HARDSWISH, 1),
)
LAST_CHANNEL = 1280


def make_divisible(v, divisor=8):
    """torchvision's _make_divisible: round to a multiple of ``divisor`` without losing over 10%."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def bn_buffers(g, prefix, c, dtype):
    return {
        "g_ptr": g.buffer(f"{prefix}.weight", dtype, (c,), "weight"),
        "beta_ptr": g.buffer(f"{prefix}.bias", dtype, (c,), "weight"),
        "mean_ptr": g.buffer(f"{prefix}.running_mean", dtype, (c,), "weight"),
        "var_ptr": g.buffer(f"{prefix}.running_var", dtype, (c,), "weight"),
    }


def conv_bn(g, x, prefix, cout, k, stride, act, out_name, residual=None, scale=None, nchw=False):
    """Launches ``act(bn(conv(x * scale)) + residual)`` for the Conv2dNormActivation at ``prefix``
    (conv ``prefix.0``, BatchNorm ``prefix.1``); ``x`` is NHWC, or NCHW when ``nchw``. Returns NHWC."""
    dtype = x.dtype
    if nchw:
        N, Cin, H, W = x.shape
        strides = (Cin * H * W, W, 1, H * W)
    else:
        N, H, W, Cin = x.shape
        strides = (H * W * Cin, W * Cin, Cin, 1)
    pad = (k - 1) // 2
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    M = N * OH * OW
    y = g.buffer(out_name, dtype, (N, OH, OW, cout), "activation")
    block_m = 128 if M >= 8192 else 64
    block_n = 64 if cout <= 64 else 128
    g.launch(
        conv2d_bn_kernel, (triton.cdiv(M, block_m) * triton.cdiv(cout, block_n),),
        num_warps=8 if block_m * block_n >= 128 * 128 else 4,
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.0.weight", dtype, (cout, Cin, k, k), "weight"),
        **bn_buffers(g, f"{prefix}.1", cout, dtype),
        r_ptr=residual if residual is not None else y, s_ptr=scale if scale is not None else y, y_ptr=y,
        M=M, H=H, W=W, Cin=Cin, OH=OH, OW=OW, Cout=cout,
        sxn=strides[0], sxh=strides[1], sxw=strides[2], sxc=strides[3],
        stride_h=stride, stride_w=stride, pad_h=pad, pad_w=pad, eps=BN_EPS,
        KH=k, KW=k, HAS_RES=residual is not None, HAS_SCALE=scale is not None, ACT=act,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=16 if Cin < 32 else 32, GROUP_M=8,
    )
    return y


def depthwise_bn(g, x, prefix, k, stride, act, out_name):
    N, H, W, C = x.shape
    pad = (k - 1) // 2
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    M = N * OH * OW
    y = g.buffer(out_name, x.dtype, (N, OH, OW, C), "activation")
    g.launch(
        depthwise_bn_kernel, (triton.cdiv(M, 32), triton.cdiv(C, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.0.weight", x.dtype, (C, 1, k, k), "weight"),
        **bn_buffers(g, f"{prefix}.1", C, x.dtype), y_ptr=y,
        M=M, H=H, W=W, C=C, OH=OH, OW=OW, stride=stride, pad=pad, eps=BN_EPS,
        K=k, ACT=act, BLOCK_P=32, BLOCK_C=64,
    )
    return y


def avgpool(g, x, out_name):
    N, H, W, C = x.shape
    y = g.buffer(out_name, x.dtype, (N, C), "activation")
    g.launch(avgpool_kernel, (N, triton.cdiv(C, 64)), x_ptr=x, y_ptr=y, HW=H * W, C=C, BLOCK_P=32, BLOCK_C=64)
    return y


def linear(g, x, weight, bias, weight_shape, act, out_name, kind="activation"):
    """``act(x @ w^T + b)``; ``weight_shape`` is the checkpoint's (a 1x1 conv is ``[N, K, 1, 1]``)."""
    M, K = x.shape
    n_out = weight_shape[0]
    y = g.buffer(out_name, x.dtype, (M, n_out), kind)
    g.launch(
        linear_kernel, (triton.cdiv(M, 16), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(weight, x.dtype, weight_shape, "weight"),
        b_ptr=g.buffer(bias, x.dtype, (n_out,), "weight"), y_ptr=y,
        M=M, N=n_out, K=K, ACT=act, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def inverted_residual(g, x, index, cin, k, expanded, cout, use_se, act, stride):
    """torchvision's InvertedResidual at ``features.{index}``: expand, depthwise, SE, project."""
    prefix = f"features.{index}.block"
    j = 0
    h = x
    if expanded != cin:
        h = conv_bn(g, h, f"{prefix}.{j}", expanded, 1, 1, act, f"{prefix}.{j}")
        j += 1
    h = depthwise_bn(g, h, f"{prefix}.{j}", k, stride, act, f"{prefix}.{j}")
    j += 1
    scale = None
    if use_se:
        squeeze = make_divisible(expanded // 4)
        pooled = avgpool(g, h, f"{prefix}.{j}.pool")
        s = linear(g, pooled, f"{prefix}.{j}.fc1.weight", f"{prefix}.{j}.fc1.bias", (squeeze, expanded, 1, 1),
                   ACT_RELU, f"{prefix}.{j}.fc1")
        scale = linear(g, s, f"{prefix}.{j}.fc2.weight", f"{prefix}.{j}.fc2.bias", (expanded, squeeze, 1, 1),
                       ACT_HARDSIGMOID, f"{prefix}.{j}.scale")
        j += 1
    residual = x if stride == 1 and cin == cout else None
    return conv_bn(g, h, f"{prefix}.{j}", cout, 1, 1, ACT_NONE, f"features.{index}.out", residual=residual, scale=scale)


def forward(batch=1, dtype="fp32"):
    g = Graph(f"forward(batch={batch}, dtype={dtype})")
    image = g.buffer("image", dtype, (batch, 3, 224, 224), "input")
    x = conv_bn(g, image, "features.0", 16, 3, 2, ACT_HARDSWISH, "features.0", nchw=True)
    for index, block in enumerate(BLOCKS, start=1):
        x = inverted_residual(g, x, index, *block)
    last = BLOCKS[-1][3] * 6
    x = conv_bn(g, x, f"features.{len(BLOCKS) + 1}", last, 1, 1, ACT_HARDSWISH, f"features.{len(BLOCKS) + 1}")
    x = avgpool(g, x, "avgpool")
    x = linear(g, x, "classifier.0.weight", "classifier.0.bias", (LAST_CHANNEL, last), ACT_HARDSWISH, "classifier.0")
    linear(g, x, "classifier.3.weight", "classifier.3.bias", (1000, LAST_CHANNEL), ACT_NONE, "logits", kind="output")
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
