"""Swin-T (torchvision ``swin_t``) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name. The residual stream is the NHWC tokens as
``[N * H * W, C]`` rows; layer norms and linears are position-wise, so they run over all rows in that order
and only attention sees windows. The window attention launch does the cyclic shift itself: window w of
the image rolled by -shift holds the 7x7 tokens at ((row + shift) mod H, (col + shift) mod W) of the
stream, and each output is stored back to the token its query came from, which is the reverse roll. Scores
get the relative position bias, gathered from ``relative_position_bias_table`` at the index torchvision's
``relative_position_index`` holds (computed from the token offsets), and in shifted blocks -100 between
tokens of different regions of the rolled image, as torchvision's mask. Patch merging gathers each 2x2
neighbourhood into 4C channels (torchvision's x0, x1, x2, x3 order) and normalizes them in one launch.
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
# Config (torchvision swin_t)
# ---------------------------------------------------------------------------------------------------

PATCH = 4
EMBED = 96
DEPTHS = (2, 2, 6, 2)
HEADS = (3, 6, 12, 24)
WINDOW = 7
MLP_RATIO = 4
EPS = 1e-5


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def patch_embed_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, C, IMG, GRID, D,
    PATCH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[m, :] = conv(x, w)[n, :, p] + b for the M = N * GRID^2 patches m = (n, p), p row-major: the PATCH x
    PATCH stride-PATCH conv as an im2col GEMM over the NCHW image x [N, C, IMG, IMG], column k = (c, kh, kw)
    of the raw [D, C, PATCH, PATCH] weight. y is the NHWC token rows [M, D]."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < D
    p = offs_m % (GRID * GRID)
    x_rows = x_ptr + (offs_m // (GRID * GRID)) * (C * IMG * IMG) + (p // GRID) * (PATCH * IMG) + (p % GRID) * PATCH
    K = C * PATCH * PATCH
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x_cols = (k // (PATCH * PATCH)) * (IMG * IMG) + ((k // PATCH) % PATCH) * IMG + k % PATCH
        a = tl.load(x_rows[:, None] + x_cols[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * D + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    """y[r] = (x[r] - mean) / sqrt(var + eps) * w + b over the N columns of row r = program_id(0), the
    biased variance as torch.nn.LayerNorm; x and y are contiguous [rows, N]."""
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + r * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * N + offs, (d * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, res_ptr, y_ptr, M, N, K,
    BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ w[N, K]^T (+ b)) (+ res[M, N]), act the exact (erf) GELU if ``GELU``, else the
    identity; w is the raw ``nn.Linear`` weight, all contiguous."""
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
    if BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    mask = m_mask[:, None] & n_mask[None, :]
    out = offs_m[:, None] * N + offs_n[None, :]
    if RESIDUAL:
        acc += tl.load(res_ptr + out, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + out, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def window_attention_kernel(
    qkv_ptr, table_ptr, o_ptr, H, W, sm_scale,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, WS: tl.constexpr, SHIFT: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Window attention of head program_id(1) in window program_id(0) (row-major over the H/WS x W/WS
    windows) of image program_id(2), on the image cyclically shifted by -SHIFT. qkv is [N * H * W, 3 * C]
    (q | k | v, C = HEADS * HEAD_DIM, as ``attn.qkv`` packs them) and o is [N * H * W, C], both NHWC token
    rows. table is ``relative_position_bias_table`` [(2 WS - 1)^2, HEADS]."""
    C: tl.constexpr = HEADS * HEAD_DIM
    win = tl.program_id(0)
    head = tl.program_id(1)
    img = tl.program_id(2)
    t = tl.arange(0, BLOCK_T)
    t_mask = t < WS * WS
    ty = t // WS
    tx = t % WS
    ry = (win // (W // WS)) * WS + ty  # the token's position in the rolled image
    rx = (win % (W // WS)) * WS + tx
    rows = img * H * W + ((ry + SHIFT) % H) * W + (rx + SHIFT) % W  # and in the stream
    d = tl.arange(0, HEAD_DIM)
    base = qkv_ptr + head * HEAD_DIM
    q = tl.load(base + rows[:, None] * (3 * C) + d[None, :], mask=t_mask[:, None], other=0.0)
    k = tl.load(base + C + rows[None, :] * (3 * C) + d[:, None], mask=t_mask[None, :], other=0.0)
    v = tl.load(base + 2 * C + rows[:, None] * (3 * C) + d[None, :], mask=t_mask[:, None], other=0.0)
    q = (q.to(tl.float32) * sm_scale).to(k.dtype)
    s = tl.dot(q, k, input_precision="ieee")
    rel = (ty[:, None] - ty[None, :] + WS - 1) * (2 * WS - 1) + (tx[:, None] - tx[None, :] + WS - 1)
    s += tl.load(table_ptr + rel * HEADS + head, mask=t_mask[:, None] & t_mask[None, :], other=0.0).to(tl.float32)
    if SHIFT > 0:
        # torchvision's regions of the rolled image: [0, H - WS), [H - WS, H - SHIFT), [H - SHIFT, H) per axis.
        region_y = (ry >= H - WS).to(tl.int32) + (ry >= H - SHIFT).to(tl.int32)
        region_x = (rx >= W - WS).to(tl.int32) + (rx >= W - SHIFT).to(tl.int32)
        region = region_y * 3 + region_x
        s += tl.where(region[:, None] != region[None, :], -100.0, 0.0)
    s = tl.where(t_mask[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    l = tl.sum(p, axis=1)
    out = tl.dot(p.to(v.dtype), v, input_precision="ieee") / l[:, None]
    tl.store(o_ptr + rows[:, None] * C + head * HEAD_DIM + d[None, :], out.to(o_ptr.dtype.element_ty),
             mask=t_mask[:, None])


@triton.jit
def patch_merge_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, H, W, C, eps, BLOCK_K: tl.constexpr):
    """y[r] = layer_norm(cat(x[2i, 2j], x[2i + 1, 2j], x[2i, 2j + 1], x[2i + 1, 2j + 1])) over its 4C channels
    for the output token r = program_id(0) = (n, i, j), row-major over [N, H / 2, W / 2]; x is the NHWC
    token rows [N * H * W, C] (H and W even), y the rows [N * H * W / 4, 4C]."""
    r = tl.program_id(0)
    OW = W // 2
    n = r // ((H // 2) * OW)
    i = (r // OW) % (H // 2)
    j = r % OW
    k = tl.arange(0, BLOCK_K)
    mask = k < 4 * C
    part = k // C
    src = (n * H + 2 * i + part % 2) * W + 2 * j + part // 2
    x = tl.load(x_ptr + src * C + k % C, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / (4 * C)
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / (4 * C) + eps)
    w = tl.load(w_ptr + k, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * (4 * C) + k, (d * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def token_mean_kernel(x_ptr, y_ptr, T, C, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """y[n, c] = mean over the T token rows of image program_id(1) of x [N * T, C]."""
    n = tl.program_id(1)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        x = tl.load(x_ptr + (n * T + offs_t)[:, None] * C + offs_c[None, :],
                    mask=(offs_t < T)[:, None] & c_mask[None, :], other=0.0)
        acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(y_ptr + n * C + offs_c, (acc / T).to(y_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def layernorm(g, x, prefix, y):
    """torch.nn.LayerNorm(C, eps=1e-5) at ``prefix`` over the rows of x into y."""
    rows, c = x.shape
    g.launch(
        layernorm_kernel, (rows,), x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (c,), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (c,), "weight"), y_ptr=y, N=c, eps=EPS,
        BLOCK_N=triton.next_power_of_2(c),
    )
    return y


def linear(g, x, prefix, n_out, y, bias=True, gelu=False, res=None):
    """y = act(x @ <prefix>.weight^T + <prefix>.bias) (+ res)."""
    m, k = x.shape
    g.launch(
        linear_kernel, (triton.cdiv(m, 64), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (n_out, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (n_out,), "weight") if bias else y,
        res_ptr=res if res is not None else y, y_ptr=y, M=m, N=n_out, K=k,
        BIAS=bias, GELU=gelu, RESIDUAL=res is not None, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return y


def swin_block(g, h, prefix, batch, size, heads, shift):
    """torchvision's SwinTransformerBlock at ``prefix`` on the [N * size^2, C] stream h: h + attn(norm1(h)),
    then that + mlp(norm2(that)). Stochastic depth and dropout are the identity in eval."""
    rows, c = h.shape
    act = lambda name, width: g.buffer(f"{prefix}.{name}", h.dtype, (rows, width), "activation")
    x = layernorm(g, h, f"{prefix}.norm1", act("norm1", c))
    qkv = linear(g, x, f"{prefix}.attn.qkv", 3 * c, act("attn.qkv", 3 * c))
    attn = act("attn", c)
    windows = (size // WINDOW) ** 2
    g.launch(
        window_attention_kernel, (windows, heads, batch), qkv_ptr=qkv,
        table_ptr=g.buffer(f"{prefix}.attn.relative_position_bias_table", h.dtype, ((2 * WINDOW - 1) ** 2, heads), "weight"),
        o_ptr=attn, H=size, W=size, sm_scale=(c // heads) ** -0.5,
        HEADS=heads, HEAD_DIM=c // heads, WS=WINDOW, SHIFT=shift, BLOCK_T=triton.next_power_of_2(WINDOW * WINDOW),
    )
    h = linear(g, attn, f"{prefix}.attn.proj", c, act("attn.out", c), res=h)
    x = layernorm(g, h, f"{prefix}.norm2", act("norm2", c))
    x = linear(g, x, f"{prefix}.mlp.0", MLP_RATIO * c, act("mlp.1", MLP_RATIO * c), gelu=True)
    return linear(g, x, f"{prefix}.mlp.3", c, g.buffer(prefix, h.dtype, (rows, c), "activation"), res=h)


def patch_merging(g, h, prefix, batch, size):
    """torchvision's PatchMerging at ``prefix``: 2x2 neighbours to 4C channels, LayerNorm, Linear(4C, 2C)
    without bias. size is even here, so torchvision's padding is empty."""
    rows, c = h.shape
    merged = g.buffer(f"{prefix}.norm", h.dtype, (rows // 4, 4 * c), "activation")
    g.launch(
        patch_merge_norm_kernel, (rows // 4,), x_ptr=h,
        w_ptr=g.buffer(f"{prefix}.norm.weight", h.dtype, (4 * c,), "weight"),
        b_ptr=g.buffer(f"{prefix}.norm.bias", h.dtype, (4 * c,), "weight"), y_ptr=merged,
        H=size, W=size, C=c, eps=EPS, BLOCK_K=triton.next_power_of_2(4 * c),
    )
    return linear(g, merged, f"{prefix}.reduction", 2 * c, g.buffer(prefix, h.dtype, (rows // 4, 2 * c), "activation"),
                  bias=False)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32", image_size=224):
    """Swin-T on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]."""
    g = Graph(f"swin_t.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    image = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    size = image_size // PATCH
    m = batch * size * size
    patches = g.buffer("features.0.0", dtype, (m, EMBED), "activation")
    g.launch(
        patch_embed_kernel, (triton.cdiv(m, 64), triton.cdiv(EMBED, 64)),
        x_ptr=image, w_ptr=g.buffer("features.0.0.weight", dtype, (EMBED, 3, PATCH, PATCH), "weight"),
        b_ptr=g.buffer("features.0.0.bias", dtype, (EMBED,), "weight"), y_ptr=patches,
        M=m, C=3, IMG=image_size, GRID=size, D=EMBED, PATCH=PATCH, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16,
    )
    h = layernorm(g, patches, "features.0.2", g.buffer("features.0", dtype, (m, EMBED), "activation"))
    for stage, (depth, heads) in enumerate(zip(DEPTHS, HEADS)):
        for i in range(depth):
            # Odd blocks shift by half a window, unless one window covers the image (torchvision then uses 0).
            shift = WINDOW // 2 if i % 2 == 1 and WINDOW < size else 0
            h = swin_block(g, h, f"features.{2 * stage + 1}.{i}", batch, size, heads, shift)
        if stage + 1 < len(DEPTHS):
            h = patch_merging(g, h, f"features.{2 * stage + 2}", batch, size)
            size //= 2
    rows, c = h.shape
    h = layernorm(g, h, "norm", g.buffer("norm", dtype, (rows, c), "activation"))
    pooled = g.buffer("avgpool", dtype, (batch, c), "activation")
    g.launch(token_mean_kernel, (triton.cdiv(c, 64), batch), x_ptr=h, y_ptr=pooled, T=size * size, C=c,
             BLOCK_T=64, BLOCK_C=64)
    linear(g, pooled, "head", 1000, g.buffer("logits", dtype, (batch, 1000), "output"))
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
