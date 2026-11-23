"""DenseNet-121 (torchvision ``densenet121``) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
and are read through strides; each BatchNorm is applied from its raw ``weight``/``bias``/``running_mean``/
``running_var``. Activations are NHWC. A dense block's channel concatenation is one buffer holding all of
its channels: the stem's max pool (or the previous transition's average pool) writes the first ones, and
each dense layer reads the channels written so far and appends its 32 in place (``torch.cat(features, 1)``
without the copies). Pre-activation BN-ReLU runs as the prologue of the conv that follows it (norm1 on
conv1's input, a transition's norm on its conv's input); norm2-ReLU and the stem's norm0-ReLU run in their
conv's epilogue, so the 3x3 conv's zero padding pads the activated tensor, as in torchvision.
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
# Config (torchvision densenet121)
# ---------------------------------------------------------------------------------------------------

GROWTH = 32
BLOCKS = (6, 12, 24, 16)
INIT_FEATURES = 64
BN_SIZE = 4
BN_EPS = 1e-5


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def bn_relu(x, offs_c, c_mask, g_ptr, beta_ptr, mean_ptr, var_ptr, eps):
    """relu(BatchNorm(x)) in eval mode over the columns (channels) of an fp32 tile."""
    gamma = tl.load(g_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_c, mask=c_mask, other=1.0).to(tl.float32)
    return tl.maximum((x - mean[None, :]) * (gamma / tl.sqrt(var + eps))[None, :] + beta[None, :], 0.0)


@triton.jit
def conv_kernel(
    x_ptr, w_ptr,
    pre_g_ptr, pre_b_ptr, pre_m_ptr, pre_v_ptr, post_g_ptr, post_b_ptr, post_m_ptr, post_v_ptr,
    y_ptr, H, W, Cin, OH, OW, Cout, sxn, sxh, sxw, sxc, ldy, y_off, stride, pad, eps,
    K: tl.constexpr, PRE: tl.constexpr, POST: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y = post(conv(pre(x))) for image program_id(2), an implicit GEMM over its output pixels (rows) and
    output channels (columns); pre is relu(bn) of the input channels if ``PRE``, post relu(bn) of the output
    channels if ``POST``, else the identity. x[n, h, w, c] is at n * sxn + h * sxh + w * sxw + c * sxc for the
    first Cin channels; output pixel r of image n, channel j is stored at (n * OH * OW + r) * ldy + y_off + j,
    so y may be a channel slice of a wider NHWC buffer. w is the contiguous [Cout, Cin, K, K] weight."""
    n = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
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
                c = c0 + tl.arange(0, BLOCK_K)
                c_mask = c < Cin
                valid = ok[:, None] & c_mask[None, :]
                a = tl.load(x_rows[:, None] + c[None, :] * sxc, mask=valid, other=0.0)
                if PRE:
                    a = bn_relu(a.to(tl.float32), c, c_mask, pre_g_ptr, pre_b_ptr, pre_m_ptr, pre_v_ptr, eps)
                    a = tl.where(valid, a, 0.0).to(w_ptr.dtype.element_ty)
                b = tl.load(
                    w_ptr + offs_n[None, :] * (Cin * K * K) + c[:, None] * (K * K) + (kh * K + kw),
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="ieee")
    if POST:
        acc = bn_relu(acc, offs_n, n_mask, post_g_ptr, post_b_ptr, post_m_ptr, post_v_ptr, eps)
    y = y_ptr + (n * OH * OW + offs_m)[:, None] * ldy + y_off + offs_n[None, :]
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def maxpool3x3s2_kernel(x_ptr, y_ptr, H, W, C, OH, OW, ldy, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """3x3 max pooling, stride 2, padding 1 (padding never wins) of image program_id(2): x is contiguous NHWC;
    output pixel r, channel c goes to (n * OH * OW + r) * ldy + c."""
    n = tl.program_id(2)
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_mask = offs_p < OH * OW
    c_mask = offs_c < C
    oh = offs_p // OW
    ow = offs_p % OW
    acc = tl.full((BLOCK_P, BLOCK_C), float("-inf"), dtype=tl.float32)
    for kh in tl.static_range(3):
        ih = oh * 2 - 1 + kh
        for kw in tl.static_range(3):
            iw = ow * 2 - 1 + kw
            ok = p_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x = tl.load(x_ptr + ((n * H + ih) * W + iw)[:, None] * C + offs_c[None, :],
                        mask=ok[:, None] & c_mask[None, :], other=float("-inf"))
            acc = tl.maximum(acc, x.to(tl.float32))
    y = y_ptr + (n * OH * OW + offs_p)[:, None] * ldy + offs_c[None, :]
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=p_mask[:, None] & c_mask[None, :])


@triton.jit
def avgpool2x2_kernel(x_ptr, y_ptr, H, W, C, ldy, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """2x2 average pooling, stride 2, of image program_id(2): x is contiguous NHWC [N, H, W, C] (H, W even);
    output pixel r, channel c goes to (n * H * W / 4 + r) * ldy + c."""
    n = tl.program_id(2)
    OW = W // 2
    P = (H // 2) * OW
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_p < P)[:, None] & (offs_c < C)[None, :]
    top = (n * H + (offs_p // OW) * 2) * W + (offs_p % OW) * 2
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            acc += tl.load(x_ptr + (top + dy * W + dx)[:, None] * C + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
    y = y_ptr + (n * P + offs_p)[:, None] * ldy + offs_c[None, :]
    tl.store(y, (acc * 0.25).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def bn_relu_mean_kernel(
    x_ptr, g_ptr, b_ptr, m_ptr, v_ptr, y_ptr, HW, C, eps,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """y[n, c] = mean over the HW pixels of relu(bn(x[n, :, :, c])) for image program_id(1); x is contiguous
    NHWC: the final norm5, ReLU and adaptive average pool."""
    n = tl.program_id(1)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for p0 in range(0, HW, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        mask = (offs_p < HW)[:, None] & c_mask[None, :]
        x = tl.load(x_ptr + (n * HW + offs_p)[:, None] * C + offs_c[None, :], mask=mask, other=0.0)
        acc += tl.where(mask, bn_relu(x.to(tl.float32), offs_c, c_mask, g_ptr, b_ptr, m_ptr, v_ptr, eps), 0.0)
    tl.store(y_ptr + n * C + offs_c, (tl.sum(acc, axis=0) / HW).to(y_ptr.dtype.element_ty), mask=c_mask)


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


def bn(g, prefix, c, dtype, side=""):
    """The BatchNorm2d at ``prefix`` as kernel arguments <side>g_ptr, <side>b_ptr, <side>m_ptr, <side>v_ptr."""
    names = ("weight", "bias", "running_mean", "running_var")
    return {f"{side}{k}_ptr": g.buffer(f"{prefix}.{name}", dtype, (c,), "weight") for k, name in zip("gbmv", names)}


def conv(g, x, cin, weight, cout, k, stride, pad, y, y_off=0, pre=None, post=None, nchw=False):
    """y[..., y_off:y_off + cout] = post(conv(pre(x[..., :cin]))) with the raw conv ``weight``; pre and post
    name BatchNorms (each followed by a ReLU). x is NHWC (its first cin channels), or NCHW if ``nchw``."""
    dtype = x.dtype
    if nchw:
        n, c, h, w = x.shape
        strides = dict(sxn=c * h * w, sxh=w, sxw=1, sxc=h * w)
    else:
        n, h, w, c = x.shape
        strides = dict(sxn=h * w * c, sxh=w * c, sxw=c, sxc=1)
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    assert y.shape[1:3] == (oh, ow) and y_off + cout <= y.shape[3]
    unused = {f"{side}_{k}_ptr": y for side in ("pre", "post") for k in "gbmv"}
    block_n = 32 if cout <= 32 else 64
    g.launch(
        conv_kernel, (triton.cdiv(oh * ow, 64), triton.cdiv(cout, block_n), n),
        x_ptr=x, w_ptr=g.buffer(weight, dtype, (cout, cin, k, k), "weight"),
        **{**unused, **(bn(g, pre, cin, dtype, "pre_") if pre else {}), **(bn(g, post, cout, dtype, "post_") if post else {})},
        y_ptr=y, H=h, W=w, Cin=cin, OH=oh, OW=ow, Cout=cout, **strides, ldy=y.shape[3], y_off=y_off,
        stride=stride, pad=pad, eps=BN_EPS, K=k, PRE=pre is not None, POST=post is not None,
        BLOCK_M=64, BLOCK_N=block_n, BLOCK_K=16 if cin < 32 else 32,
    )
    return y


def dense_block(g, block, b, c0):
    """torchvision's _DenseBlock ``features.denseblock<b>`` on its buffer ``block``, whose first c0 channels
    are written: each _DenseLayer is relu(norm1) -> conv1 1x1 -> relu(norm2) -> conv2 3x3, appended."""
    n, h, w, _ = block.shape
    width = BN_SIZE * GROWTH
    for i in range(BLOCKS[b - 1]):
        prefix = f"features.denseblock{b}.denselayer{i + 1}"
        cin = c0 + i * GROWTH
        t = g.buffer(f"{prefix}.conv1", block.dtype, (n, h, w, width), "activation")
        conv(g, block, cin, f"{prefix}.conv1.weight", width, 1, 1, 0, t, pre=f"{prefix}.norm1", post=f"{prefix}.norm2")
        conv(g, t, width, f"{prefix}.conv2.weight", GROWTH, 3, 1, 1, block, y_off=cin)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32", image_size=224):
    """DenseNet-121 on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]."""
    g = Graph(f"densenet121.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    image = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    s = image_size // 2
    stem = g.buffer("features.conv0", dtype, (batch, s, s, INIT_FEATURES), "activation")
    conv(g, image, 3, "features.conv0.weight", INIT_FEATURES, 7, 2, 3, stem, post="features.norm0", nchw=True)

    c = INIT_FEATURES
    size = (s + 2 - 3) // 2 + 1
    block = g.buffer("features.denseblock1", dtype, (batch, size, size, c + BLOCKS[0] * GROWTH), "activation")
    g.launch(
        maxpool3x3s2_kernel, (triton.cdiv(size * size, 32), triton.cdiv(c, 64), batch), x_ptr=stem, y_ptr=block,
        H=s, W=s, C=c, OH=size, OW=size, ldy=block.shape[3], BLOCK_P=32, BLOCK_C=64,
    )
    for b in range(1, len(BLOCKS) + 1):
        dense_block(g, block, b, c)
        c += BLOCKS[b - 1] * GROWTH
        if b == len(BLOCKS):
            break
        # _Transition: relu(norm) -> conv 1x1 halving the channels -> 2x2 average pool into the next block.
        prefix = f"features.transition{b}"
        t = g.buffer(f"{prefix}.conv", dtype, (batch, size, size, c // 2), "activation")
        conv(g, block, c, f"{prefix}.conv.weight", c // 2, 1, 1, 0, t, pre=f"{prefix}.norm")
        c //= 2
        size //= 2
        block = g.buffer(f"features.denseblock{b + 1}", dtype, (batch, size, size, c + BLOCKS[b] * GROWTH), "activation")
        g.launch(
            avgpool2x2_kernel, (triton.cdiv(size * size, 32), triton.cdiv(c, 64), batch), x_ptr=t, y_ptr=block,
            H=2 * size, W=2 * size, C=c, ldy=block.shape[3], BLOCK_P=32, BLOCK_C=64,
        )

    pooled = g.buffer("avgpool", dtype, (batch, c), "activation")
    g.launch(
        bn_relu_mean_kernel, (triton.cdiv(c, 64), batch), x_ptr=block,
        **bn(g, "features.norm5", c, dtype),
        y_ptr=pooled, HW=size * size, C=c, eps=BN_EPS, BLOCK_P=16, BLOCK_C=64,
    )
    logits = g.buffer("logits", dtype, (batch, 1000), "output")
    g.launch(
        linear_kernel, (triton.cdiv(batch, 16), triton.cdiv(1000, 64)),
        x_ptr=pooled, w_ptr=g.buffer("classifier.weight", dtype, (1000, c), "weight"),
        b_ptr=g.buffer("classifier.bias", dtype, (1000,), "weight"), y_ptr=logits,
        M=batch, N=1000, K=c, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
