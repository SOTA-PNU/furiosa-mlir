"""EfficientNet-B0 (torchvision ``efficientnet_b0``) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
(depthwise ``[C, 1, K, K]``) and are read through strides. Batch norm is explicit, not folded: every conv
applies its eval-mode BatchNorm in the epilogue from the raw ``weight``/``bias``/``running_mean``/
``running_var``, then SiLU where torchvision has one. Squeeze-and-excitation is a global average pool and
one launch for both of its 1x1 convs (matrix-vector products: SiLU between, sigmoid after); the projection
conv multiplies its input by that per-channel gate as it loads. Activations are NHWC; grids are
(pixel blocks, channel blocks, image).
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
# Config (torchvision efficientnet_b0: width and depth multipliers 1.0)
# ---------------------------------------------------------------------------------------------------

# Per stage: (expand ratio, kernel, stride, input channels, output channels, blocks).
STAGES = (
    (1, 3, 1, 32, 16, 1),
    (6, 3, 2, 16, 24, 2),
    (6, 5, 2, 24, 40, 2),
    (6, 3, 2, 40, 80, 3),
    (6, 5, 1, 80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
)
STEM = 32
LAST = 1280
BN_EPS = 1e-5


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def bn_epilogue(acc, offs_c, c_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps):
    """Eval-mode BatchNorm over the columns (channels) of an fp32 tile."""
    gamma = tl.load(g_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_c, mask=c_mask, other=1.0).to(tl.float32)
    return (acc - mean[None, :]) * (gamma * tl.rsqrt(var + eps))[None, :] + beta[None, :]


@triton.jit
def conv_bn_kernel(
    x_ptr, w_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr, s_ptr, r_ptr, y_ptr,
    H, W, Cin, OH, OW, Cout, sxn, sxh, sxw, sxc, stride, pad, eps,
    K: tl.constexpr, SCALE: tl.constexpr, RESIDUAL: tl.constexpr, SILU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y = silu(bn(conv(x * s)) + r) for image program_id(2), an implicit GEMM: rows are its output pixels,
    columns output channels. The input scale s ([N, Cin], if ``SCALE``), the residual r (if ``RESIDUAL``)
    and the SiLU (if ``SILU``) are optional. x[n, h, w, c] is at n * sxn + h * sxh + w * sxw + c * sxc;
    w is the contiguous [Cout, Cin, K, K] conv weight; r and y are contiguous NHWC."""
    n = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < OH * OW
    n_mask = offs_n < Cout
    oh = offs_m // OW
    ow = offs_m % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in range(K):
        ih = oh * stride - pad + kh
        for kw in range(K):
            iw = ow * stride - pad + kw
            ok = m_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_rows = x_ptr + n * sxn + ih * sxh + iw * sxw
            for c0 in range(0, Cin, BLOCK_K):
                c = c0 + offs_k
                c_mask = c < Cin
                a = tl.load(x_rows[:, None] + c[None, :] * sxc, mask=ok[:, None] & c_mask[None, :], other=0.0)
                if SCALE:
                    s = tl.load(s_ptr + n * Cin + c, mask=c_mask, other=0.0).to(tl.float32)
                    a = (a.to(tl.float32) * s[None, :]).to(w_ptr.dtype.element_ty)
                b = tl.load(
                    w_ptr + offs_n[None, :] * (Cin * K * K) + c[:, None] * (K * K) + (kh * K + kw),
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = bn_epilogue(acc, offs_n, n_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps)
    mask = m_mask[:, None] & n_mask[None, :]
    out = (n * OH * OW + offs_m)[:, None] * Cout + offs_n[None, :]
    if RESIDUAL:
        acc += tl.load(r_ptr + out, mask=mask, other=0.0).to(tl.float32)
    if SILU:
        acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + out, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def depthwise_bn_silu_kernel(
    x_ptr, w_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr, y_ptr,
    H, W, C, OH, OW, stride, pad, eps,
    K: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """y = silu(bn(depthwise_conv(x))) for image program_id(2): a KxK conv per channel, zero padding ``pad``.
    x and y are contiguous NHWC; w is the contiguous [C, 1, K, K] weight."""
    n = tl.program_id(2)
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_mask = offs_p < OH * OW
    c_mask = offs_c < C
    oh = offs_p // OW
    ow = offs_p % OW
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for kh in tl.static_range(K):
        ih = oh * stride - pad + kh
        for kw in tl.static_range(K):
            iw = ow * stride - pad + kw
            ok = p_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x = tl.load(
                x_ptr + ((n * H + ih) * W + iw)[:, None] * C + offs_c[None, :],
                mask=ok[:, None] & c_mask[None, :], other=0.0,
            )
            w = tl.load(w_ptr + offs_c * (K * K) + (kh * K + kw), mask=c_mask, other=0.0)
            acc += x.to(tl.float32) * w.to(tl.float32)[None, :]
    acc = bn_epilogue(acc, offs_c, c_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps)
    acc = acc * tl.sigmoid(acc)
    out = (n * OH * OW + offs_p)[:, None] * C + offs_c[None, :]
    tl.store(y_ptr + out, acc.to(y_ptr.dtype.element_ty), mask=p_mask[:, None] & c_mask[None, :])


@triton.jit
def global_avgpool_kernel(x_ptr, y_ptr, HW, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[n, c] = mean of x[n, :, :, c] over the HW pixels of image program_id(1); x is contiguous NHWC."""
    n = tl.program_id(1)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for p0 in range(0, HW, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        x = tl.load(
            x_ptr + (n * HW + offs_p)[:, None] * C + offs_c[None, :],
            mask=(offs_p < HW)[:, None] & c_mask[None, :], other=0.0,
        )
        acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(y_ptr + n * C + offs_c, (acc / HW).to(y_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def squeeze_excite_kernel(
    p_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, s_ptr, C, S,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """s[n] = sigmoid(w2 @ silu(w1 @ p[n] + b1) + b2) for image n = program_id(0): SqueezeExcitation's two
    1x1 convs on the pooled p [N, C], as matrix-vector products. w1 is [S, C, 1, 1], w2 [C, S, 1, 1]."""
    n = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < S
    z = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        c_mask = offs_c < C
        p = tl.load(p_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptr + offs_s[:, None] * C + offs_c[None, :], mask=s_mask[:, None] & c_mask[None, :], other=0.0)
        z += tl.sum(w1.to(tl.float32) * p[None, :], axis=1)
    z += tl.load(b1_ptr + offs_s, mask=s_mask, other=0.0).to(tl.float32)
    z = z * tl.sigmoid(z)  # masked lanes stay 0
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        c_mask = offs_c < C
        w2 = tl.load(w2_ptr + offs_c[:, None] * S + offs_s[None, :], mask=c_mask[:, None] & s_mask[None, :], other=0.0)
        e = tl.sum(w2.to(tl.float32) * z[None, :], axis=1) + tl.load(b2_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(s_ptr + n * C + offs_c, tl.sigmoid(e).to(s_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ w[N, K]^T + b, all contiguous; w is the raw ``nn.Linear`` weight."""
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
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def bn(g, prefix, c, dtype):
    """The BatchNorm2d at ``prefix`` as kernel arguments."""
    return {
        "g_ptr": g.buffer(f"{prefix}.weight", dtype, (c,), "weight"),
        "beta_ptr": g.buffer(f"{prefix}.bias", dtype, (c,), "weight"),
        "mean_ptr": g.buffer(f"{prefix}.running_mean", dtype, (c,), "weight"),
        "var_ptr": g.buffer(f"{prefix}.running_var", dtype, (c,), "weight"),
    }


def conv_bn(g, x, prefix, cout, k, stride, silu, out_name, scale=None, residual=None, nchw=False):
    """The Conv2dNormActivation at ``prefix`` (conv ``.0``, BatchNorm ``.1``, SiLU if ``silu``) on NHWC x
    (NCHW if ``nchw``), its input scaled per channel by ``scale`` and ``residual`` added before the act."""
    dtype = x.dtype
    if nchw:
        n, cin, h, w = x.shape
        strides = dict(sxn=cin * h * w, sxh=w, sxw=1, sxc=h * w)
    else:
        n, h, w, cin = x.shape
        strides = dict(sxn=h * w * cin, sxh=w * cin, sxw=cin, sxc=1)
    pad = (k - 1) // 2
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    y = g.buffer(out_name, dtype, (n, oh, ow, cout), "activation")
    block_n = 32 if cout <= 32 else 64
    g.launch(
        conv_bn_kernel, (triton.cdiv(oh * ow, 64), triton.cdiv(cout, block_n), n),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.0.weight", dtype, (cout, cin, k, k), "weight"), **bn(g, f"{prefix}.1", cout, dtype),
        s_ptr=scale if scale is not None else y, r_ptr=residual if residual is not None else y, y_ptr=y,
        H=h, W=w, Cin=cin, OH=oh, OW=ow, Cout=cout, **strides, stride=stride, pad=pad, eps=BN_EPS,
        K=k, SCALE=scale is not None, RESIDUAL=residual is not None, SILU=silu,
        BLOCK_M=64, BLOCK_N=block_n, BLOCK_K=16 if cin < 32 else 32,
    )
    return y


def depthwise_bn_silu(g, x, prefix, k, stride):
    n, h, w, c = x.shape
    pad = (k - 1) // 2
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    y = g.buffer(prefix, x.dtype, (n, oh, ow, c), "activation")
    g.launch(
        depthwise_bn_silu_kernel, (triton.cdiv(oh * ow, 32), triton.cdiv(c, 64), n),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.0.weight", x.dtype, (c, 1, k, k), "weight"), **bn(g, f"{prefix}.1", c, x.dtype),
        y_ptr=y, H=h, W=w, C=c, OH=oh, OW=ow, stride=stride, pad=pad, eps=BN_EPS, K=k, BLOCK_P=32, BLOCK_C=64,
    )
    return y


def global_avgpool(g, x, out_name):
    n, h, w, c = x.shape
    y = g.buffer(out_name, x.dtype, (n, c), "activation")
    g.launch(global_avgpool_kernel, (triton.cdiv(c, 64), n), x_ptr=x, y_ptr=y, HW=h * w, C=c, BLOCK_P=32, BLOCK_C=64)
    return y


def squeeze_excite(g, x, prefix, squeeze):
    """torchvision's SqueezeExcitation at ``prefix`` on NHWC x: returns the [N, C] gate."""
    n, _, _, c = x.shape
    pooled = global_avgpool(g, x, f"{prefix}.avgpool")
    w = lambda name, *shape: g.buffer(f"{prefix}.{name}", x.dtype, shape, "weight")
    s = g.buffer(f"{prefix}.scale", x.dtype, (n, c), "activation")
    g.launch(
        squeeze_excite_kernel, (n,), p_ptr=pooled, w1_ptr=w("fc1.weight", squeeze, c, 1, 1), b1_ptr=w("fc1.bias", squeeze),
        w2_ptr=w("fc2.weight", c, squeeze, 1, 1), b2_ptr=w("fc2.bias", c), s_ptr=s, C=c, S=squeeze,
        BLOCK_C=128, BLOCK_S=triton.next_power_of_2(squeeze),
    )
    return s


def mbconv(g, x, prefix, expand, k, stride, cin, cout):
    """torchvision's MBConv at ``prefix``: expand (1x1, unless the ratio is 1), depthwise KxK, SE, project
    (1x1, no activation), plus the input when the shape is kept. Stochastic depth is the identity in eval."""
    j = 0
    h = x
    if expand != 1:
        h = conv_bn(g, h, f"{prefix}.block.0", cin * expand, 1, 1, True, f"{prefix}.block.0")
        j = 1
    h = depthwise_bn_silu(g, h, f"{prefix}.block.{j}", k, stride)
    scale = squeeze_excite(g, h, f"{prefix}.block.{j + 1}", max(1, cin // 4))
    residual = x if stride == 1 and cin == cout else None
    return conv_bn(g, h, f"{prefix}.block.{j + 2}", cout, 1, 1, False, prefix, scale=scale, residual=residual)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32", image_size=224):
    """EfficientNet-B0 on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]. Dropout is the identity in eval."""
    g = Graph(f"efficientnet_b0.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    image = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    x = conv_bn(g, image, "features.0", STEM, 3, 2, True, "features.0", nchw=True)
    for stage, (expand, k, stride, cin, cout, blocks) in enumerate(STAGES, start=1):
        for i in range(blocks):
            x = mbconv(g, x, f"features.{stage}.{i}", expand, k, stride if i == 0 else 1, cin if i == 0 else cout, cout)
    head = len(STAGES) + 1
    x = conv_bn(g, x, f"features.{head}", LAST, 1, 1, True, f"features.{head}")
    x = global_avgpool(g, x, "avgpool")
    logits = g.buffer("logits", dtype, (batch, 1000), "output")
    g.launch(
        linear_kernel, (triton.cdiv(batch, 16), triton.cdiv(1000, 64)),
        x_ptr=x, w_ptr=g.buffer("classifier.1.weight", dtype, (1000, LAST), "weight"),
        b_ptr=g.buffer("classifier.1.bias", dtype, (1000,), "weight"), y_ptr=logits,
        M=batch, N=1000, K=LAST, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
