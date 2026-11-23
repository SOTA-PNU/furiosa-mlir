"""Llama 3.1 8B (Llama-3.1-8B-Instruct) in pure Triton: greedy text generation as prefill and decode launch graphs.

Written from the math and transformers' LlamaForCausalLM: RMSNorm, grouped-query attention (32 query heads over 8
key/value heads, no q/k norms, no biases) with rotary embeddings under Llama 3's frequency scaling, a SwiGLU MLP and
an untied head. Weights are the raw checkpoint tensors by their safetensors names. Each layer's K/V cache is a
"state" buffer of ``[max_seq, 8 * 128]`` rows that prefill and decode share, and the argmax writes the next token
into the "next_token" state that the following decode graph embeds, so decoding is repeated ``run(decode(pos))``.
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
# Config (meta-llama/Llama-3.1-8B-Instruct config.json)
# ---------------------------------------------------------------------------------------------------

HIDDEN = 4096
LAYERS = 32
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
INTER = 14336
VOCAB = 128256
EPS = 1e-5
MAX_SEQ = 2048
# rope_parameters: rope_type "llama3"
ROPE_THETA = 500000.0
ROPE_FACTOR = 8.0
LOW_FREQ_FACTOR = 1.0
HIGH_FREQ_FACTOR = 4.0
OLD_CONTEXT = 8192


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def embedding_kernel(ids_ptr, table_ptr, out_ptr, H, BLOCK: tl.constexpr):
    """out[t, cols] = table[ids[t], cols] over program (t, j)'s column block."""
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    row = tl.load(ids_ptr + t)
    tl.store(out_ptr + t * H + cols, tl.load(table_ptr + row * H + cols, mask=mask), mask=mask)


@triton.jit
def add_rmsnorm_kernel(x_ptr, delta_ptr, w_ptr, out_ptr, row0, N, eps, ADD: tl.constexpr, BLOCK: tl.constexpr):
    """Row r = row0 + program: x[r] += delta[r] in place (if ADD), then out[program] = x[r] * rsqrt(mean(x[r]^2) + eps)
    rounded to x's dtype, times w (LlamaRMSNorm). Two passes over BLOCK-wide chunks: the sum of squares, then the
    scaled row; each chunk is stored after it is read."""
    p = tl.program_id(0)
    r = row0 + p
    dt = x_ptr.dtype.element_ty
    sq = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + r * N + cols, mask=mask, other=0.0)
        if ADD:
            x = (x.to(tl.float32) + tl.load(delta_ptr + r * N + cols, mask=mask, other=0.0).to(tl.float32)).to(dt)
        x = x.to(tl.float32)
        sq += x * x
    rstd = tl.rsqrt(tl.sum(sq, axis=0) / N + eps)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + r * N + cols, mask=mask, other=0.0)
        if ADD:
            x = (x.to(tl.float32) + tl.load(delta_ptr + r * N + cols, mask=mask, other=0.0).to(tl.float32)).to(dt)
            tl.store(x_ptr + r * N + cols, x, mask=mask)
        normed = (x.to(tl.float32) * rstd).to(dt).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + p * N + cols, (normed * w).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, w_ptr, c_ptr, c_row0, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """c[c_row0 + m, n] = sum_k a[m, k] * w[n, k]: nn.Linear without bias, a [M, K] and w [N, K] row-major."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[:, None] * K + rk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = rk[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask, other=0.0)
        w = tl.load(w_ptrs, mask=(rn[:, None] < N) & k_mask, other=0.0)
        acc += tl.dot(a, tl.trans(w))
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    c_ptrs = c_ptr + (c_row0 + rm[:, None]) * N + rn[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gate_up_kernel(
    a_ptr, wg_ptr, wu_ptr, c_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """c = silu(a @ wg^T) * (a @ wu^T), each product rounded to c's dtype as LlamaMLP's bf16 ops round."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    w_off = rn[:, None] * K + rk[None, :]
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = rk[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask, other=0.0)
        w_mask = (rn[:, None] < N) & k_mask
        gate += tl.dot(a, tl.trans(tl.load(wg_ptr + w_off + k * BLOCK_K, mask=w_mask, other=0.0)))
        up += tl.dot(a, tl.trans(tl.load(wu_ptr + w_off + k * BLOCK_K, mask=w_mask, other=0.0)))
        a_ptrs += BLOCK_K
    dt = c_ptr.dtype.element_ty
    g = gate.to(dt).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(dt).to(tl.float32)
    out = (act * up.to(dt).to(tl.float32)).to(dt)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def rope_kernel(
    x_ptr, row0, start_pos, log2_theta, factor, low_freq_factor, high_freq_factor, old_context,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """In place, every head of row row0 + t of x [*, HEADS * HEAD_DIM], the token at position start_pos + t:
    rotate-half RoPE with Llama 3's inverse frequencies (transformers' _compute_llama3_parameters, in fp32)."""
    t = tl.program_id(0)
    pos = (start_pos + t).to(tl.float32)
    HALF: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, HALF)
    inv_freq = tl.exp2(-((2 * i).to(tl.float32) / HEAD_DIM) * log2_theta)  # theta^(-2i/d)
    wavelen = 6.283185307179586 / inv_freq
    low_freq_wavelen = old_context / low_freq_factor
    high_freq_wavelen = old_context / high_freq_factor
    inv_freq = tl.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    inv_freq = tl.where((wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen), smoothed, inv_freq)
    dt = x_ptr.dtype.element_ty
    angle = pos * inv_freq
    cos = tl.cos(angle).to(dt).to(tl.float32)[None, :]
    sin = tl.sin(angle).to(dt).to(tl.float32)[None, :]

    h = tl.arange(0, HEADS)[:, None]
    lo = x_ptr + (row0 + t) * (HEADS * HEAD_DIM) + h * HEAD_DIM + i[None, :]
    x1 = tl.load(lo).to(tl.float32)
    x2 = tl.load(lo + HALF).to(tl.float32)
    # q * cos + rotate_half(q) * sin, each product rounded as the bf16 torch ops round.
    tl.store(lo, ((x1 * cos).to(dt).to(tl.float32) - (x2 * sin).to(dt).to(tl.float32)).to(dt))
    tl.store(lo + HALF, ((x2 * cos).to(dt).to(tl.float32) + (x1 * sin).to(dt).to(tl.float32)).to(dt))


@triton.jit
def gqa_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, start_pos, T, sm_scale,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Causal attention of queries at positions start_pos + [0, T) over cache rows [0, start_pos + T). One program
    per (block of BLOCK_T queries, KV head) holds the GROUP query heads that share the KV head as rows
    t * GROUP + j, so each K/V tile is loaded once for the group. q and o are [T, Q_HEADS * D]; k and v are
    [max_seq, KV_HEADS * D]."""
    GROUP: tl.constexpr = Q_HEADS // KV_HEADS
    kv_head = tl.program_id(1)
    r = tl.arange(0, BLOCK_T * GROUP)
    t = tl.program_id(0) * BLOCK_T + r // GROUP
    head = kv_head * GROUP + r % GROUP
    d = tl.arange(0, HEAD_DIM)
    t_mask = t < T
    q_rows = t[:, None] * (Q_HEADS * HEAD_DIM) + head[:, None] * HEAD_DIM + d[None, :]
    q = tl.load(q_ptr + q_rows, mask=t_mask[:, None], other=0.0)

    scale = sm_scale * 1.4426950408889634  # softmax in base 2
    m_i = tl.full((BLOCK_T * GROUP,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T * GROUP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T * GROUP, HEAD_DIM), dtype=tl.float32)
    kv_end = start_pos + tl.minimum(T, (tl.program_id(0) + 1) * BLOCK_T)
    for n0 in range(0, kv_end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        n_mask = n < kv_end
        kv_rows = n[:, None] * (KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + d[None, :]
        k = tl.load(k_ptr + kv_rows, mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(n_mask[None, :] & (n[None, :] <= start_pos + t[:, None]), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_rows, mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(o_ptr + q_rows, (acc / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=t_mask[:, None])


@triton.jit
def argmax_kernel(x_ptr, out_ptr, row, N, BLOCK: tl.constexpr):
    """out[0] = the first index of the maximum of x[row, :N], a chunk at a time."""
    best = tl.max(tl.full((BLOCK,), float("-inf"), dtype=tl.float32), axis=0)
    best_i = tl.argmax(tl.zeros((BLOCK,), dtype=tl.int32), axis=0)
    for n0 in range(0, N, BLOCK):
        cols = n0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * N + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        m = tl.max(v, axis=0)
        best_i = tl.where(m > best, n0 + tl.argmax(v, axis=0), best_i)  # an earlier chunk keeps a tie
        best = tl.maximum(best, m)
    tl.store(out_ptr, best_i)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def _gemm_launch(M):
    """Tiles of up to 64 rows (16 at least, for tl.dot), 64 columns and 64 of K."""
    return dict(BLOCK_M=min(64, max(16, triton.next_power_of_2(M))), BLOCK_N=64, BLOCK_K=64)


def embedding(g, ids, table, out):
    H = table.shape[1]
    g.launch(embedding_kernel, (ids.shape[0], triton.cdiv(H, 1024)), ids_ptr=ids, table_ptr=table, out_ptr=out, H=H, BLOCK=1024)


def add_rmsnorm(g, x, delta, weight, out, row0=0):
    """out = rmsnorm(x[row0:] (+= delta)) * weight over out's rows."""
    g.launch(
        add_rmsnorm_kernel, (out.shape[0],), x_ptr=x, delta_ptr=delta if delta is not None else x, w_ptr=weight,
        out_ptr=out, row0=row0, N=weight.shape[0], eps=EPS, ADD=delta is not None, BLOCK=1024,
    )


def matmul(g, a, weight, c, c_row0=0):
    M, K = a.shape
    N = weight.shape[0]
    blocks = _gemm_launch(M)
    grid = (triton.cdiv(M, blocks["BLOCK_M"]), triton.cdiv(N, blocks["BLOCK_N"]))
    g.launch(matmul_kernel, grid, a_ptr=a, w_ptr=weight, c_ptr=c, c_row0=c_row0, M=M, N=N, K=K, **blocks)


def gate_up(g, a, w_gate, w_up, c):
    M, K = a.shape
    N = w_gate.shape[0]
    blocks = _gemm_launch(M)
    grid = (triton.cdiv(M, blocks["BLOCK_M"]), triton.cdiv(N, blocks["BLOCK_N"]))
    g.launch(gate_up_kernel, grid, a_ptr=a, wg_ptr=w_gate, wu_ptr=w_up, c_ptr=c, M=M, N=N, K=K, **blocks)


def rope(g, x, row0, start_pos, T, heads):
    g.launch(
        rope_kernel, (T,), x_ptr=x, row0=row0, start_pos=start_pos, log2_theta=math.log2(ROPE_THETA),
        factor=ROPE_FACTOR, low_freq_factor=LOW_FREQ_FACTOR, high_freq_factor=HIGH_FREQ_FACTOR,
        old_context=float(OLD_CONTEXT), HEADS=heads, HEAD_DIM=HEAD_DIM,
    )


def attention(g, q, k_cache, v_cache, out, start_pos):
    T = q.shape[0]
    group = Q_HEADS // KV_HEADS
    block_t = max(16 // group, min(16, triton.next_power_of_2(T)))  # tl.dot wants 16 rows at least
    g.launch(
        gqa_attention_kernel, (triton.cdiv(T, block_t), KV_HEADS),
        q_ptr=q, k_ptr=k_cache, v_ptr=v_cache, o_ptr=out, start_pos=start_pos, T=T, sm_scale=HEAD_DIM**-0.5,
        Q_HEADS=Q_HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM, BLOCK_T=block_t, BLOCK_N=64,
    )


def argmax(g, logits, out, row):
    g.launch(argmax_kernel, (1,), num_warps=8, x_ptr=logits, out_ptr=out, row=row, N=logits.shape[1], BLOCK=4096)


def layer(g, i, h, prev_mlp, T, start_pos, dtype, max_seq, tag):
    """One decoder layer on the residual stream h [T, HIDDEN]; prev_mlp is the previous layer's MLP output still to
    be added to h (None for the first layer)."""
    w = lambda name, *shape: g.buffer(f"model.layers.{i}.{name}.weight", dtype, shape, "weight")
    act = lambda name, width: g.buffer(f"{tag}.{name}", dtype, (T, width), "activation")
    QD, KVD = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    k_cache = g.buffer(f"kv_cache.{i}.k", dtype, (max_seq, KVD), "state")
    v_cache = g.buffer(f"kv_cache.{i}.v", dtype, (max_seq, KVD), "state")
    x, q, attn, o = act("x", HIDDEN), act("q", QD), act("attn", QD), act("o", HIDDEN)

    add_rmsnorm(g, h, prev_mlp, w("input_layernorm", HIDDEN), x)
    matmul(g, x, w("self_attn.q_proj", QD, HIDDEN), q)
    matmul(g, x, w("self_attn.k_proj", KVD, HIDDEN), k_cache, c_row0=start_pos)
    matmul(g, x, w("self_attn.v_proj", KVD, HIDDEN), v_cache, c_row0=start_pos)
    rope(g, q, 0, start_pos, T, Q_HEADS)
    rope(g, k_cache, start_pos, start_pos, T, KV_HEADS)
    attention(g, q, k_cache, v_cache, attn, start_pos)
    matmul(g, attn, w("self_attn.o_proj", HIDDEN, QD), o)

    add_rmsnorm(g, h, o, w("post_attention_layernorm", HIDDEN), x)
    gated, mlp = act("gated", INTER), act("mlp", HIDDEN)
    gate_up(g, x, w("mlp.gate_proj", INTER, HIDDEN), w("mlp.up_proj", INTER, HIDDEN), gated)
    matmul(g, gated, w("mlp.down_proj", HIDDEN, INTER), mlp)
    return mlp


def forward(g, ids, T, start_pos, dtype, max_seq, tag, all_logits):
    """Embeds ids [T], runs every layer, and writes logits plus the greedy next token."""
    h = g.buffer(f"{tag}.h", dtype, (T, HIDDEN), "activation")
    embedding(g, ids, g.buffer("model.embed_tokens.weight", dtype, (VOCAB, HIDDEN), "weight"), h)
    mlp = None
    for i in range(LAYERS):
        mlp = layer(g, i, h, mlp, T, start_pos, dtype, max_seq, tag)

    rows = T if all_logits else 1
    x = g.buffer(f"{tag}.final", dtype, (rows, HIDDEN), "activation")
    add_rmsnorm(g, h, mlp, g.buffer("model.norm.weight", dtype, (HIDDEN,), "weight"), x, row0=T - rows)
    logits = g.buffer("logits", dtype, (rows, VOCAB), "output")
    matmul(g, x, g.buffer("lm_head.weight", dtype, (VOCAB, HIDDEN), "weight"), logits)
    argmax(g, logits, g.buffer("next_token", "i32", (1,), "state"), row=rows - 1)
    return g


def prefill(seq_len, dtype="bf16", max_seq=MAX_SEQ, all_logits=False):
    """Prompt input_ids [seq_len] at positions [0, seq_len): fills the KV cache, writes "next_token" and "logits"
    ([seq_len, VOCAB] if all_logits, else the last row)."""
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
