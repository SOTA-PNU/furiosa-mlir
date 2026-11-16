"""Qwen3-0.6B (Qwen/Qwen3-0.6B) in pure Triton: greedy text generation as prefill and decode launch graphs.

Weights are the raw checkpoint tensors by their safetensors names, with no packing; the output head
reads ``model.embed_tokens.weight`` (tied embeddings). Each layer's K/V cache is a "state" buffer of
``[max_seq, 8 * 128]`` rows, so prefill and decode share it, and the argmax writes the next token into
the "next_token" state that the following decode graph embeds: a decode loop is repeated
``run(decode(pos))`` with no host compute.
"""

import math
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
# Config (Qwen/Qwen3-0.6B config.json)
# ---------------------------------------------------------------------------------------------------

HIDDEN = 1024
LAYERS = 28
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
INTER = 3072
VOCAB = 151936
ROPE_THETA = 1e6
EPS = 1e-6
MAX_SEQ = 2048

# log2(ROPE_THETA) split so that (2i / HEAD_DIM) * LOG2_THETA_HI is exact in fp32.
LOG2_THETA = math.log2(ROPE_THETA)
LOG2_THETA_HI = int(LOG2_THETA * 8192) / 8192
LOG2_THETA_LO = LOG2_THETA - LOG2_THETA_HI


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def embedding_kernel(ids_ptr, emb_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    """out[t, :] = emb[ids[t], :]."""
    t = tl.program_id(0)
    row = tl.load(ids_ptr + t)
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        tl.store(out_ptr + t * H + offs, tl.load(emb_ptr + row * H + offs, mask=mask), mask=mask)


@triton.jit
def rmsnorm_kernel(h_ptr, res_ptr, w_ptr, y_ptr, row0, N, eps, HAS_RES: tl.constexpr, BLOCK_N: tl.constexpr):
    """For row r = row0 + t: h[r] += res[r] (in place, if HAS_RES), then y[t] = rmsnorm(h[r]) * w.

    Rounds where Qwen3RMSNorm does: the normalized row to h's dtype before the weight multiply.
    """
    t = tl.program_id(0)
    r = row0 + t
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    dt = h_ptr.dtype.element_ty
    x = tl.load(h_ptr + r * N + offs, mask=mask, other=0.0)
    if HAS_RES:
        x = (x.to(tl.float32) + tl.load(res_ptr + r * N + offs, mask=mask, other=0.0).to(tl.float32)).to(dt)
        tl.store(h_ptr + r * N + offs, x, mask=mask)
    x = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    normed = (x * rstd).to(dt).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + t * N + offs, (normed * w).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, y_ptr, y_row0, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[y_row0 + m, n] = x[m, :] . w[n, :] for row-major x [M, K], w [N, K] and y [*, N]."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=m_mask[:, None] & (k[None, :] < K), other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=(k[:, None] < K) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    y = y_ptr + (y_row0 + offs_m[:, None]) * N + offs_n[None, :]
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def swiglu_kernel(
    x_ptr, wg_ptr, wu_ptr, y_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y = silu(x @ wg^T) * (x @ wu^T), rounding each product to y's dtype as the torch MLP does."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=m_mask[:, None] & (k[None, :] < K), other=0.0)
        w_mask = (k[:, None] < K) & n_mask[None, :]
        gate = tl.dot(a, tl.load(wg_ptr + offs_n[None, :] * K + k[:, None], mask=w_mask, other=0.0), gate, input_precision="ieee")
        up = tl.dot(a, tl.load(wu_ptr + offs_n[None, :] * K + k[:, None], mask=w_mask, other=0.0), up, input_precision="ieee")
    dt = y_ptr.dtype.element_ty
    gate = gate.to(dt).to(tl.float32)
    act = (gate * tl.sigmoid(gate)).to(dt).to(tl.float32)
    y = (act * up.to(dt).to(tl.float32)).to(dt)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def qk_norm_rope_kernel(
    q_ptr, k_ptr, qw_ptr, kw_ptr, start_pos, eps, log2_theta_hi, log2_theta_lo,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """In place, for token t at position start_pos + t: RMSNorm each q head (q [T, Q_HEADS * D]) and
    each k head (cache rows start_pos + t of [max_seq, KV_HEADS * D]), then rotate-half RoPE."""
    t = tl.program_id(0)
    head = tl.program_id(1)
    pos = start_pos + t
    if head < Q_HEADS:
        x_ptr = q_ptr + t * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM
        w_ptr = qw_ptr
    else:
        x_ptr = k_ptr + pos * (KV_HEADS * HEAD_DIM) + (head - Q_HEADS) * HEAD_DIM
        w_ptr = kw_ptr
    HALF: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, HALF)
    dt = q_ptr.dtype.element_ty
    x1 = tl.load(x_ptr + i).to(tl.float32)
    x2 = tl.load(x_ptr + HALF + i).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / HEAD_DIM + eps)
    x1 = ((x1 * rstd).to(dt).to(tl.float32) * tl.load(w_ptr + i).to(tl.float32)).to(dt).to(tl.float32)
    x2 = ((x2 * rstd).to(dt).to(tl.float32) * tl.load(w_ptr + HALF + i).to(tl.float32)).to(dt).to(tl.float32)

    # inv_freq = 1 / theta^(2i/D): exponent split into an exact integer part and a small fraction.
    e = (2 * i).to(tl.float32) / HEAD_DIM
    xh = e * log2_theta_hi
    n = tl.floor(xh)
    two_n = ((n.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)
    inv_freq = tl.fdiv(1.0, tl.exp2((xh - n) + e * log2_theta_lo) * two_n, ieee_rounding=True)
    angle = pos.to(tl.float32) * inv_freq
    # Reduce the angle to [-pi, pi] (Cody-Waite, 2*pi in three parts) before the hardware sin/cos.
    turns = tl.floor(angle * 0.15915494309189535 + 0.5)
    angle = tl.fma(-turns, 6.28125, angle)
    angle = tl.fma(-turns, 0.0019353071693331003, angle)
    angle = tl.fma(-turns, 1.0253131677018246e-11, angle)
    cos = tl.cos(angle).to(dt).to(tl.float32)
    sin = tl.sin(angle).to(dt).to(tl.float32)
    out1 = (x1 * cos).to(dt).to(tl.float32) - (x2 * sin).to(dt).to(tl.float32)
    out2 = (x2 * cos).to(dt).to(tl.float32) + (x1 * sin).to(dt).to(tl.float32)
    tl.store(x_ptr + i, out1.to(dt))
    tl.store(x_ptr + HALF + i, out2.to(dt))


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, start_pos, T, sm_scale,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Causal GQA flash attention of queries at positions start_pos + [0, T) over cache rows
    [0, start_pos + T). q and o are [T, Q_HEADS * D]; k and v are [max_seq, KV_HEADS * D]."""
    head = tl.program_id(1)
    kv_head = head // (Q_HEADS // KV_HEADS)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < T
    q_pos = start_pos + offs_m
    q = tl.load(q_ptr + offs_m[:, None] * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM + offs_d[None, :], mask=m_mask[:, None], other=0.0)

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    kv_end = start_pos + tl.minimum(T, (tl.program_id(0) + 1) * BLOCK_M)
    kv_row = KV_HEADS * HEAD_DIM
    for n0 in range(0, kv_end, BLOCK_N):
        n = n0 + offs_n
        n_mask = n < kv_end
        k = tl.load(k_ptr + n[None, :] * kv_row + kv_head * HEAD_DIM + offs_d[:, None], mask=n_mask[None, :], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        s = tl.where(n_mask[None, :] & (n[None, :] <= q_pos[:, None]), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + n[:, None] * kv_row + kv_head * HEAD_DIM + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee")
        m_i = m_new
    out = acc / l_i[:, None]
    o = o_ptr + offs_m[:, None] * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM + offs_d[None, :]
    tl.store(o, out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def argmax_kernel(x_ptr, out_ptr, row, N, BLOCK: tl.constexpr):
    """out[0] = the first index of the maximum of x[row, :N]."""
    offs = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    best_i = tl.zeros((BLOCK,), dtype=tl.int32)
    for n0 in range(0, N, BLOCK):
        v = tl.load(x_ptr + row * N + n0 + offs, mask=n0 + offs < N, other=float("-inf")).to(tl.float32)
        better = v > best
        best = tl.where(better, v, best)
        best_i = tl.where(better, n0 + offs, best_i)
    top = tl.max(best, axis=0)
    tl.store(out_ptr, tl.min(tl.where(best == top, best_i, N), axis=0))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def _gemm_blocks(M, dtype, fused=False):
    """Fixed tiles per launch: weight-streaming tiles for a few rows, square tiles otherwise."""
    if M <= 16:
        return dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128), dict(num_warps=4, num_stages=3)
    return dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 if fused else 64), dict(num_warps=4, num_stages=2 if dtype == "fp32" else 3)


def embedding(g, ids, emb, out):
    g.launch(embedding_kernel, (ids.shape[0],), ids_ptr=ids, emb_ptr=emb, out_ptr=out, H=HIDDEN, BLOCK_H=triton.next_power_of_2(HIDDEN))


def rmsnorm(g, h, res, weight, y, row0=0):
    """y = rmsnorm(h[row0:] (+= res)) * weight over y's rows."""
    g.launch(
        rmsnorm_kernel, (y.shape[0],), h_ptr=h, res_ptr=res if res is not None else h, w_ptr=weight, y_ptr=y,
        row0=row0, N=HIDDEN, eps=EPS, HAS_RES=res is not None, BLOCK_N=triton.next_power_of_2(HIDDEN),
    )


def linear(g, x, weight, y, y_row0=0):
    M, K = x.shape
    N = weight.shape[0]
    blocks, options = _gemm_blocks(M, x.dtype)
    grid = (triton.cdiv(M, blocks["BLOCK_M"]), triton.cdiv(N, blocks["BLOCK_N"]))
    g.launch(linear_kernel, grid, **options, x_ptr=x, w_ptr=weight, y_ptr=y, y_row0=y_row0, M=M, N=N, K=K, **blocks)


def swiglu(g, x, w_gate, w_up, y):
    M, K = x.shape
    N = w_gate.shape[0]
    blocks, options = _gemm_blocks(M, x.dtype, fused=True)
    grid = (triton.cdiv(M, blocks["BLOCK_M"]), triton.cdiv(N, blocks["BLOCK_N"]))
    g.launch(swiglu_kernel, grid, **options, x_ptr=x, wg_ptr=w_gate, wu_ptr=w_up, y_ptr=y, M=M, N=N, K=K, **blocks)


def qk_norm_rope(g, q, k_cache, q_norm, k_norm, start_pos):
    g.launch(
        qk_norm_rope_kernel, (q.shape[0], Q_HEADS + KV_HEADS), num_warps=1,
        q_ptr=q, k_ptr=k_cache, qw_ptr=q_norm, kw_ptr=k_norm, start_pos=start_pos, eps=EPS,
        log2_theta_hi=LOG2_THETA_HI, log2_theta_lo=LOG2_THETA_LO, Q_HEADS=Q_HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM,
    )


def attention(g, q, k_cache, v_cache, out, start_pos):
    T = q.shape[0]
    block_m = 16 if T <= 16 else 64
    fp32 = q.dtype == "fp32"
    g.launch(
        attention_kernel, (triton.cdiv(T, block_m), Q_HEADS), num_warps=4, num_stages=2 if fp32 else 3,
        q_ptr=q, k_ptr=k_cache, v_ptr=v_cache, o_ptr=out, start_pos=start_pos, T=T, sm_scale=HEAD_DIM**-0.5,
        Q_HEADS=Q_HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM, BLOCK_M=block_m, BLOCK_N=32 if fp32 else 64,
    )


def argmax(g, logits, out, row):
    g.launch(argmax_kernel, (1,), num_warps=8, x_ptr=logits, out_ptr=out, row=row, N=logits.shape[1], BLOCK=4096)


def layer(g, i, h, prev_mlp, T, start_pos, dtype, max_seq, tag):
    """One decoder layer on the residual stream h [T, HIDDEN]; prev_mlp is the previous layer's MLP
    output still to be added to h (None for the first layer)."""
    w = lambda name, *shape: g.buffer(f"model.layers.{i}.{name}.weight", dtype, shape, "weight")
    act = lambda name, width: g.buffer(f"{tag}.{name}", dtype, (T, width), "activation")
    k_cache = g.buffer(f"kv_cache.{i}.k", dtype, (max_seq, KV_HEADS * HEAD_DIM), "state")
    v_cache = g.buffer(f"kv_cache.{i}.v", dtype, (max_seq, KV_HEADS * HEAD_DIM), "state")
    x, q, attn, o = act("x", HIDDEN), act("q", Q_HEADS * HEAD_DIM), act("attn", Q_HEADS * HEAD_DIM), act("o", HIDDEN)

    rmsnorm(g, h, prev_mlp, w("input_layernorm", HIDDEN), x)
    linear(g, x, w("self_attn.q_proj", Q_HEADS * HEAD_DIM, HIDDEN), q)
    linear(g, x, w("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN), k_cache, y_row0=start_pos)
    linear(g, x, w("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN), v_cache, y_row0=start_pos)
    qk_norm_rope(g, q, k_cache, w("self_attn.q_norm", HEAD_DIM), w("self_attn.k_norm", HEAD_DIM), start_pos)
    attention(g, q, k_cache, v_cache, attn, start_pos)
    linear(g, attn, w("self_attn.o_proj", HIDDEN, Q_HEADS * HEAD_DIM), o)

    rmsnorm(g, h, o, w("post_attention_layernorm", HIDDEN), x)
    gated, mlp = act("gated", INTER), act("mlp", HIDDEN)
    swiglu(g, x, w("mlp.gate_proj", INTER, HIDDEN), w("mlp.up_proj", INTER, HIDDEN), gated)
    linear(g, gated, w("mlp.down_proj", HIDDEN, INTER), mlp)
    return mlp


def forward(g, ids, T, start_pos, dtype, max_seq, tag, all_logits):
    """Embeds ids [T], runs every layer, and writes logits plus the greedy next token."""
    emb = g.buffer("model.embed_tokens.weight", dtype, (VOCAB, HIDDEN), "weight")
    h = g.buffer(f"{tag}.h", dtype, (T, HIDDEN), "activation")
    embedding(g, ids, emb, h)
    mlp = None
    for i in range(LAYERS):
        mlp = layer(g, i, h, mlp, T, start_pos, dtype, max_seq, tag)

    rows = T if all_logits else 1
    x = g.buffer(f"{tag}.final", dtype, (rows, HIDDEN), "activation")
    rmsnorm(g, h, mlp, g.buffer("model.norm.weight", dtype, (HIDDEN,), "weight"), x, row0=T - rows)
    logits = g.buffer("logits", dtype, (rows, VOCAB), "output")
    linear(g, x, emb, logits)  # tied head: the embedding table is the [VOCAB, HIDDEN] weight
    argmax(g, logits, g.buffer("next_token", "i32", (1,), "state"), row=rows - 1)
    return g


def prefill(seq_len, dtype="bf16", max_seq=MAX_SEQ, all_logits=False):
    """Prompt input_ids [seq_len] at positions [0, seq_len): fills the KV cache, writes "next_token"
    and "logits" ([seq_len, VOCAB] if all_logits, else the last row)."""
    g = Graph(f"prefill(seq_len={seq_len}, dtype={dtype}, max_seq={max_seq})")
    ids = g.buffer("input_ids", "i32", (seq_len,), "input")
    return forward(g, ids, seq_len, 0, dtype, max_seq, f"prefill{seq_len}", all_logits)


def decode(start_pos, dtype="bf16", max_seq=MAX_SEQ):
    """One step: embeds "next_token" at position start_pos, then overwrites it with the next one."""
    g = Graph(f"decode(start_pos={start_pos}, dtype={dtype}, max_seq={max_seq})")
    ids = g.buffer("next_token", "i32", (1,), "state")
    return forward(g, ids, 1, start_pos, dtype, max_seq, "decode", False)


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [prefill(seq_len=32), decode(start_pos=32)]
