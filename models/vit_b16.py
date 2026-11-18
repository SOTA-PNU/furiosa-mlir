"""ViT-B/16 (torchvision ``vit_b_16``) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name, unpacked: the 16x16 stride-16 patch conv
reads ``conv_proj.weight`` ``[768, 3, 16, 16]`` as a ``[768, 768]`` matrix (an im2col GEMM over the NCHW
image), and attention reads q, k and v as column ranges of the packed ``in_proj`` output. The residual
stream is ``[N * 197, 768]`` rows (class token first); every layer norm and linear runs over all tokens,
except the final layer norm and the head, which only the class token reaches.
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
# Config (torchvision vit_b_16)
# ---------------------------------------------------------------------------------------------------

DIM = 768
HEADS = 12
MLP_DIM = 3072
LAYERS = 12
PATCH = 16
EPS = 1e-6


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def patch_embed_kernel(
    x_ptr, w_ptr, b_ptr, pos_ptr, h_ptr,
    M, C, IMG, GRID, D,
    PATCH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """h[n, 1 + p, :] = conv(x, w)[n, :, p] + b + pos[1 + p] for the M = N * GRID^2 patches, p row-major over
    the patch grid: an im2col GEMM whose column k is (c, kh, kw) of the raw ``[D, C, PATCH, PATCH]`` weight.
    x is NCHW ``[N, C, IMG, IMG]``; h is ``[N, 1 + GRID^2, D]``."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < D
    P = GRID * GRID
    n = offs_m // P
    p = offs_m % P
    x_rows = x_ptr + n * (C * IMG * IMG) + (p // GRID) * (PATCH * IMG) + (p % GRID) * PATCH
    K = C * PATCH * PATCH
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        x_cols = (k // (PATCH * PATCH)) * (IMG * IMG) + ((k // PATCH) % PATCH) * IMG + k % PATCH
        a = tl.load(x_rows[:, None] + x_cols[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    mask = m_mask[:, None] & n_mask[None, :]
    acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    acc += tl.load(pos_ptr + (1 + p)[:, None] * D + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    rows = n * (P + 1) + 1 + p
    tl.store(h_ptr + rows[:, None] * D + offs_n[None, :], acc.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def class_token_kernel(cls_ptr, pos_ptr, h_ptr, T, D, BLOCK_D: tl.constexpr):
    """h[n, 0, :] = class_token + pos[0] for image n = program_id(0); h is ``[N, T, D]``."""
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v = tl.load(cls_ptr + offs, mask=mask, other=0.0).to(tl.float32) + tl.load(pos_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + tl.program_id(0) * T * D + offs, v.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, x_stride, N, eps, BLOCK_N: tl.constexpr):
    """y[r] = (x[r] - mean) / sqrt(var + eps) * w + b over the N columns of row r = program_id(0), with the
    biased variance as torch.nn.LayerNorm; x rows are x_stride elements apart, y rows N."""
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + r * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * N + offs, (d * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, res_ptr, y_ptr, M, N, K,
    GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ w[N, K]^T + b) (+ res[M, N] if ``RESIDUAL``), act the exact (erf) GELU if
    ``GELU``, else the identity; w is the raw ``nn.Linear`` weight."""
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
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    mask = m_mask[:, None] & n_mask[None, :]
    if RESIDUAL:
        acc += tl.load(res_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def attention_kernel(
    qkv_ptr, o_ptr, T, sm_scale,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Unmasked softmax attention of head program_id(1) of image program_id(2), query rows block
    program_id(0). qkv is ``[N * T, 3 * D]`` (q | k | v, D = HEADS * HEAD_DIM, as ``in_proj`` packs them) and
    o is ``[N * T, D]``. All T keys fit one BLOCK_T tile, so the softmax is taken over the whole row."""
    D: tl.constexpr = HEADS * HEAD_DIM
    head = tl.program_id(1)
    img = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < T
    t_mask = offs_t < T
    base = qkv_ptr + img * T * (3 * D) + head * HEAD_DIM
    q = tl.load(base + offs_m[:, None] * (3 * D) + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    k = tl.load(base + D + offs_t[None, :] * (3 * D) + offs_d[:, None], mask=t_mask[None, :], other=0.0)
    s = tl.dot(q, k, input_precision="ieee") * sm_scale
    s = tl.where(t_mask[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    l = tl.sum(p, axis=1)
    v = tl.load(base + 2 * D + offs_t[:, None] * (3 * D) + offs_d[None, :], mask=t_mask[:, None], other=0.0)
    out = tl.dot(p.to(v.dtype), v, input_precision="ieee") / l[:, None]
    o = o_ptr + (img * T + offs_m)[:, None] * D + head * HEAD_DIM + offs_d[None, :]
    tl.store(o, out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def embed(g, image, dtype):
    """Patch embedding, class token and position embedding: the NCHW image to the ``[N * T, DIM]`` stream."""
    n, c, size, _ = image.shape
    grid = size // PATCH
    t = grid * grid + 1
    h = g.buffer("encoder.input", dtype, (n * t, DIM), "activation")
    pos = g.buffer("encoder.pos_embedding", dtype, (1, t, DIM), "weight")
    m = n * grid * grid
    g.launch(
        patch_embed_kernel, (triton.cdiv(m, 64), triton.cdiv(DIM, 64)),
        x_ptr=image, w_ptr=g.buffer("conv_proj.weight", dtype, (DIM, c, PATCH, PATCH), "weight"),
        b_ptr=g.buffer("conv_proj.bias", dtype, (DIM,), "weight"), pos_ptr=pos, h_ptr=h,
        M=m, C=c, IMG=size, GRID=grid, D=DIM, PATCH=PATCH, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    g.launch(
        class_token_kernel, (n,), cls_ptr=g.buffer("class_token", dtype, (1, 1, DIM), "weight"), pos_ptr=pos,
        h_ptr=h, T=t, D=DIM, BLOCK_D=triton.next_power_of_2(DIM),
    )
    return h, t


def layernorm(g, x, prefix, y, x_stride=DIM):
    """torch.nn.LayerNorm(DIM) with ``<prefix>.weight`` / ``<prefix>.bias`` over y's rows of x."""
    g.launch(
        layernorm_kernel, (y.shape[0],), x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (DIM,), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (DIM,), "weight"), y_ptr=y,
        x_stride=x_stride, N=DIM, eps=EPS, BLOCK_N=triton.next_power_of_2(DIM),
    )
    return y


def linear(g, x, weight, bias, y, gelu=False, res=None):
    """y = act(x @ weight.T + bias) (+ res)."""
    m, k = x.shape
    n = weight.shape[0]
    block_m = 16 if m <= 16 else 64
    g.launch(
        linear_kernel, (triton.cdiv(m, block_m), triton.cdiv(n, 64)),
        x_ptr=x, w_ptr=weight, b_ptr=bias, res_ptr=res if res is not None else y, y_ptr=y,
        M=m, N=n, K=k, GELU=gelu, RESIDUAL=res is not None, BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def encoder_block(g, i, h, batch, t, dtype):
    """torchvision's EncoderBlock on the stream h: h + attn(ln_1(h)), then that + mlp(ln_2(that))."""
    prefix = f"encoder.layers.encoder_layer_{i}"
    w = lambda name, *shape: g.buffer(f"{prefix}.{name}", dtype, shape, "weight")
    act = lambda name, width: g.buffer(f"{prefix}.{name}", dtype, (batch * t, width), "activation")

    x = layernorm(g, h, f"{prefix}.ln_1", act("ln_1.out", DIM))
    qkv = linear(g, x, w("self_attention.in_proj_weight", 3 * DIM, DIM), w("self_attention.in_proj_bias", 3 * DIM),
                 act("self_attention.qkv", 3 * DIM))
    attn = act("self_attention.out", DIM)
    g.launch(
        attention_kernel, (triton.cdiv(t, 32), HEADS, batch), qkv_ptr=qkv, o_ptr=attn, T=t,
        sm_scale=(DIM // HEADS) ** -0.5, HEADS=HEADS, HEAD_DIM=DIM // HEADS, BLOCK_M=32, BLOCK_T=triton.next_power_of_2(t),
    )
    h = linear(g, attn, w("self_attention.out_proj.weight", DIM, DIM), w("self_attention.out_proj.bias", DIM),
               act("attention_residual", DIM), res=h)

    x = layernorm(g, h, f"{prefix}.ln_2", act("ln_2.out", DIM))
    x = linear(g, x, w("mlp.0.weight", MLP_DIM, DIM), w("mlp.0.bias", MLP_DIM), act("mlp.1.out", MLP_DIM), gelu=True)
    return linear(g, x, w("mlp.3.weight", DIM, MLP_DIM), w("mlp.3.bias", DIM), act("out", DIM), res=h)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=1, dtype="fp32", image_size=224):
    """ViT-B/16 on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]. Dropout is the identity in eval."""
    g = Graph(f"vit_b16.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    image = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    h, t = embed(g, image, dtype)
    for i in range(LAYERS):
        h = encoder_block(g, i, h, batch, t, dtype)
    cls = layernorm(g, h, "encoder.ln", g.buffer("encoder.ln.out", dtype, (batch, DIM), "activation"), x_stride=t * DIM)
    linear(g, cls, g.buffer("heads.head.weight", dtype, (1000, DIM), "weight"),
           g.buffer("heads.head.bias", dtype, (1000,), "weight"), g.buffer("logits", dtype, (batch, 1000), "output"))
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
