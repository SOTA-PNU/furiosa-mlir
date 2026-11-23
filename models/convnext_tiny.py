"""ConvNeXt-Tiny (torchvision ``convnext_tiny``) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
(depthwise ``[C, 1, 7, 7]``) and are read through strides. As in torchvision, the stream between blocks
is NCHW and each block works channels-last: the 7x7 depthwise conv reads NCHW and writes NHWC rows, the
LayerNorm and the two linears run over those rows, and the second linear's epilogue applies the layer
scale, adds the block's input and stores NCHW again. LayerNorm2d (the stem's, each downsampling layer's
and the classifier's) is a LayerNorm over the channels of an NCHW or NHWC tensor read through strides.
The stem (4x4 stride 4) and downsampling (2x2 stride 2) convs are non-overlapping patch GEMMs.
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
# Config (torchvision convnext_tiny)
# ---------------------------------------------------------------------------------------------------

DIMS = (96, 192, 384, 768)
DEPTHS = (3, 3, 9, 3)
EPS = 1e-6


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def patchify_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, Cin, OW, P_OUT, Cout,
    sxn, sxc, sxh, sxw, syn, syp, syc,
    P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """A PxP conv with stride P and bias over image program_id(2): y[p, co] = b[co] + sum over k = (c, kh,
    kw) of x[c, oh * P + kh, ow * P + kw] * w[co, k] for its P_OUT output pixels p = oh * OW + ow. x[n, c,
    h, w] is at n * sxn + c * sxc + h * sxh + w * sxw, y[n, p, co] at n * syn + p * syp + co * syc;
    w is the contiguous [Cout, Cin, P, P] weight."""
    n = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < P_OUT
    n_mask = offs_n < Cout
    x_rows = x_ptr + n * sxn + (offs_m // OW) * (P * sxh) + (offs_m % OW) * (P * sxw)
    K = Cin * P * P
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x_cols = (k // (P * P)) * sxc + ((k // P) % P) * sxh + (k % P) * sxw
        a = tl.load(x_rows[:, None] + x_cols[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    y = y_ptr + n * syn + offs_m[:, None] * syp + offs_n[None, :] * syc
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, R, C, eps,
    sxn, sxr, sxc, syn, syr, syc,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """LayerNorm over the C channels of rows r of image program_id(1), BLOCK_R rows per program: y = (x -
    mean) / sqrt(var + eps) * w + b with the biased variance. x[n, r, c] is at n * sxn + r * sxr + c * sxc,
    y likewise with the sy strides: an NCHW tensor is rows (pixels) with channel stride H * W."""
    n = tl.program_id(1)
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.arange(0, BLOCK_C)
    r_mask = offs_r < R
    c_mask = offs_c < C
    mask = r_mask[:, None] & c_mask[None, :]
    x = tl.load(x_ptr + n * sxn + offs_r[:, None] * sxr + offs_c[None, :] * sxc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    d = tl.where(mask, x - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
    w = tl.load(w_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    y = d * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(y_ptr + n * syn + offs_r[:, None] * syr + offs_c[None, :] * syc, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def dwconv7_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, H, W, C,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Depthwise 7x7 conv, zero padding 3, with bias, over image program_id(2): x is NCHW [N, C, H, W] and
    y the NHWC rows [N * H * W, C] (the block's permute to channels-last); w is the contiguous [C, 1, 7, 7]."""
    n = tl.program_id(2)
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_mask = offs_p < H * W
    c_mask = offs_c < C
    oh = offs_p // W
    ow = offs_p % W
    planes = x_ptr + (n * C + offs_c) * (H * W)
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for kh in range(7):
        ih = oh + kh - 3
        for kw in range(7):
            iw = ow + kw - 3
            ok = p_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x = tl.load(planes[None, :] + (ih * W + iw)[:, None], mask=ok[:, None] & c_mask[None, :], other=0.0)
            w = tl.load(w_ptr + offs_c * 49 + (kh * 7 + kw), mask=c_mask, other=0.0)
            acc += x.to(tl.float32) * w.to(tl.float32)[None, :]
    acc += tl.load(b_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)[None, :]
    y = y_ptr + (n * H * W + offs_p)[:, None] * C + offs_c[None, :]
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=p_mask[:, None] & c_mask[None, :])


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, gamma_ptr, res_ptr, y_ptr, M, N, K, HW,
    GELU: tl.constexpr, LAYER_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """acc = x[M, K] @ w[N, K]^T + b, w the raw ``nn.Linear`` weight. y[M, N] = gelu(acc) (exact, erf) if
    ``GELU``, else acc. With ``LAYER_SCALE`` the rows are the HW pixels of each image and the output is NCHW:
    y[n, j, p] = res[n, j, p] + gamma[j] * acc[n * HW + p, j], gamma the [N, 1, 1] layer scale."""
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
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    if LAYER_SCALE:
        gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out = ((offs_m // HW) * N * HW + offs_m % HW)[:, None] + offs_n[None, :] * HW
        acc = tl.load(res_ptr + out, mask=mask, other=0.0).to(tl.float32) + gamma[None, :] * acc
    else:
        out = offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptr + out, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def spatial_mean_kernel(x_ptr, y_ptr, C, HW, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr):
    """y[n, c] = mean of the contiguous HW plane x[n, c] (AdaptiveAvgPool2d(1) on NCHW), image program_id(1)."""
    n = tl.program_id(1)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_C, BLOCK_P), dtype=tl.float32)
    for p0 in range(0, HW, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        x = tl.load(
            x_ptr + (n * C + offs_c)[:, None] * HW + offs_p[None, :],
            mask=c_mask[:, None] & (offs_p < HW)[None, :], other=0.0,
        )
        acc += x.to(tl.float32)
    tl.store(y_ptr + n * C + offs_c, (tl.sum(acc, axis=1) / HW).to(y_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def nchw(shape):
    """(image, row, channel) strides of an NCHW tensor read as rows of pixels."""
    n, c, h, w = shape
    return c * h * w, 1, h * w


def nhwc(shape):
    """(image, row, channel) strides of an NHWC tensor (or [N * H * W, C] rows) of the same layout."""
    n, h, w, c = shape
    return h * w * c, c, 1


def layernorm(g, x, x_strides, prefix, y, y_strides, n, rows, c):
    """torch LayerNorm(c, eps=1e-6) at ``prefix`` over the c channels of n images of ``rows`` rows each."""
    block_c = triton.next_power_of_2(c)
    block_r = max(1, 4096 // block_c)
    g.launch(
        layernorm_kernel, (triton.cdiv(rows, block_r), n),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (c,), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (c,), "weight"), y_ptr=y, R=rows, C=c, eps=EPS,
        sxn=x_strides[0], sxr=x_strides[1], sxc=x_strides[2], syn=y_strides[0], syr=y_strides[1], syc=y_strides[2],
        BLOCK_R=block_r, BLOCK_C=block_c,
    )
    return y


def patchify(g, x, x_strides, prefix, cin, cout, p, y, y_strides, h, w):
    """The PxP stride-P conv with bias at ``prefix`` on an [N, cin, h, w] x read through x_strides
    (image, channel, row, column); y strides are (image, pixel, channel)."""
    n = x.shape[0]
    oh, ow = h // p, w // p
    g.launch(
        patchify_kernel, (triton.cdiv(oh * ow, 64), triton.cdiv(cout, 64), n),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (cout, cin, p, p), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (cout,), "weight"), y_ptr=y,
        Cin=cin, OW=ow, P_OUT=oh * ow, Cout=cout, sxn=x_strides[0], sxc=x_strides[1], sxh=x_strides[2],
        sxw=x_strides[3], syn=y_strides[0], syp=y_strides[1], syc=y_strides[2],
        P=p, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16 if cin * p * p < 64 else 32,
    )
    return y


def linear(g, x, prefix, n_out, y, gelu=False, gamma=None, res=None, hw=1):
    m, k = x.shape
    g.launch(
        linear_kernel, (triton.cdiv(m, 64), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (n_out, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (n_out,), "weight"),
        gamma_ptr=gamma if gamma is not None else y, res_ptr=res if res is not None else y, y_ptr=y,
        M=m, N=n_out, K=k, HW=hw, GELU=gelu, LAYER_SCALE=gamma is not None, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return y


def cn_block(g, x, prefix):
    """torchvision's CNBlock at ``prefix`` on the NCHW x: x + layer_scale * (dwconv7, permute, LayerNorm,
    Linear(C, 4C), GELU, Linear(4C, C), permute). Stochastic depth is the identity in eval."""
    n, c, h, w = x.shape
    rows = lambda name, width: g.buffer(f"{prefix}.{name}", x.dtype, (n * h * w, width), "activation")
    t = rows("block.0", c)
    g.launch(
        dwconv7_kernel, (triton.cdiv(h * w, 32), triton.cdiv(c, 64), n),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.block.0.weight", x.dtype, (c, 1, 7, 7), "weight"),
        b_ptr=g.buffer(f"{prefix}.block.0.bias", x.dtype, (c,), "weight"), y_ptr=t, H=h, W=w, C=c,
        BLOCK_P=32, BLOCK_C=64,
    )
    t = layernorm(g, t, (h * w * c, c, 1), f"{prefix}.block.2", rows("block.2", c), (h * w * c, c, 1), n, h * w, c)
    t = linear(g, t, f"{prefix}.block.3", 4 * c, rows("block.4", 4 * c), gelu=True)
    gamma = g.buffer(f"{prefix}.layer_scale", x.dtype, (c, 1, 1), "weight")
    return linear(g, t, f"{prefix}.block.5", c, g.buffer(prefix, x.dtype, x.shape, "activation"), gamma=gamma, res=x,
                  hw=h * w)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32", image_size=224):
    """ConvNeXt-Tiny on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]."""
    g = Graph(f"convnext_tiny.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    image = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    s = image_size
    # Stem: conv 4x4/4 (NCHW -> NHWC rows), then LayerNorm2d into the NCHW stream.
    h = s // 4
    stem = g.buffer("features.0.0", dtype, (batch, h, h, DIMS[0]), "activation")
    patchify(g, image, (3 * s * s, s * s, s, 1), "features.0.0", 3, DIMS[0], 4, stem, nhwc(stem.shape), s, s)
    x = g.buffer("features.0", dtype, (batch, DIMS[0], h, h), "activation")
    layernorm(g, stem, nhwc(stem.shape), "features.0.1", x, nchw(x.shape), batch, h * h, DIMS[0])
    for stage, (c, depth) in enumerate(zip(DIMS, DEPTHS)):
        for i in range(depth):
            x = cn_block(g, x, f"features.{2 * stage + 1}.{i}")
        if stage + 1 < len(DIMS):
            # Downsampling: LayerNorm2d (NCHW -> NHWC), then conv 2x2/2 back to NCHW.
            prefix = f"features.{2 * stage + 2}"
            t = g.buffer(f"{prefix}.0", dtype, (batch, h, h, c), "activation")
            layernorm(g, x, nchw(x.shape), f"{prefix}.0", t, nhwc(t.shape), batch, h * h, c)
            x = g.buffer(prefix, dtype, (batch, DIMS[stage + 1], h // 2, h // 2), "activation")
            patchify(g, t, (h * h * c, 1, h * c, c), f"{prefix}.1", c, DIMS[stage + 1], 2, x, nchw(x.shape), h, h)
            h //= 2
    c = DIMS[-1]
    pooled = g.buffer("avgpool", dtype, (batch, c), "activation")
    g.launch(spatial_mean_kernel, (triton.cdiv(c, 64), batch), x_ptr=x, y_ptr=pooled, C=c, HW=h * h, BLOCK_C=64, BLOCK_P=64)
    normed = layernorm(g, pooled, (c, 0, 1), "classifier.0", g.buffer("classifier.0", dtype, (batch, c), "activation"),
                       (c, 0, 1), batch, 1, c)
    linear(g, normed, "classifier.2", 1000, g.buffer("logits", dtype, (batch, 1000), "output"))
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
