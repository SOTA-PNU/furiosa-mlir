"""Qwen2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct) in pure Triton: greedy text generation as prefill and decode graphs.

Written from the math and transformers' Qwen2ForCausalLM: RMSNorm, grouped-query attention (28 query heads over 4
key/value heads of 128) whose q, k and v projections carry biases, rotary embeddings (theta 1e6, no q/k norm, no
scaling), a SwiGLU MLP and an untied head. Weights are the raw checkpoint tensors by their safetensors names. Each
layer's K/V cache is a "state" buffer of ``[max_seq, 4 * 128]`` rows that prefill and decode share, and the argmax
writes the next token into the "next_token" state that the following decode graph embeds.
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
# Config (Qwen/Qwen2.5-7B-Instruct config.json)
# ---------------------------------------------------------------------------------------------------

HIDDEN = 3584
LAYERS = 28
Q_HEADS = 28
KV_HEADS = 4
HEAD_DIM = 128
INTER = 18944
VOCAB = 152064
ROPE_THETA = 1e6
EPS = 1e-6
MAX_SEQ = 2048


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def embedding_kernel(ids_ptr, table_ptr, out_ptr, H, BLOCK: tl.constexpr):
    """out[t, :H] = table[ids[t], :H] in one masked block."""
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    tl.store(out_ptr + t * H + cols, tl.load(table_ptr + tl.load(ids_ptr + t) * H + cols, mask=mask), mask=mask)


@triton.jit
def rmsnorm_kernel(h_ptr, res_ptr, w_ptr, y_ptr, row0, N, eps, RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    """Row r = row0 + program: h[r] = h[r] + res[r] (stored back, if RESIDUAL), then y[program] =
    (h[r] * rsqrt(mean(h[r]^2) + eps)) rounded to h's dtype, times w, as Qwen2RMSNorm."""
    p = tl.program_id(0)
    r = row0 + p
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    dt = h_ptr.dtype.element_ty
    h = tl.load(h_ptr + r * N + cols, mask=mask, other=0.0).to(tl.float32)
    if RESIDUAL:
        h = (h + tl.load(res_ptr + r * N + cols, mask=mask, other=0.0).to(tl.float32)).to(dt).to(tl.float32)
        tl.store(h_ptr + r * N + cols, h.to(dt), mask=mask)
    var = tl.sum(h * h, axis=0) / N
    normed = (h * tl.rsqrt(var + eps)).to(dt).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + p * N + cols, (w * normed).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, y_row0, M, N, K, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[y_row0 + m, :] = x[m, :] @ w^T (+ b), nn.Linear over x [M, K] and w [N, K]: the bias joins the fp32
    accumulator and the sum rounds once."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m[:, None] * K + k[None, :], mask=(m[:, None] < M) & (k[None, :] < K), other=0.0)
        wt = tl.load(w_ptr + n[None, :] * K + k[:, None], mask=(k[:, None] < K) & (n[None, :] < N), other=0.0)
        acc = tl.dot(x, wt, acc)
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + (y_row0 + m[:, None]) * N + n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def swiglu_kernel(
    x_ptr, wg_ptr, wu_ptr, y_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y = silu(x @ wg^T) * (x @ wu^T), rounding gate, silu, up and the product to y's dtype as Qwen2MLP does."""
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m[:, None] * K + k[None, :], mask=(m[:, None] < M) & (k[None, :] < K), other=0.0)
        w_off = n[None, :] * K + k[:, None]
        w_mask = (k[:, None] < K) & (n[None, :] < N)
        gate = tl.dot(x, tl.load(wg_ptr + w_off, mask=w_mask, other=0.0), gate)
        up = tl.dot(x, tl.load(wu_ptr + w_off, mask=w_mask, other=0.0), up)
    dt = y_ptr.dtype.element_ty
    gate = gate.to(dt).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(dt).to(tl.float32)
    tl.store(y_ptr + m[:, None] * N + n[None, :], (silu * up.to(dt).to(tl.float32)).to(dt),
             mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _rotate_half(ptrs, mask, cos, sin, HALF: tl.constexpr):
    """x * cos + rotate_half(x) * sin in place over [heads, HALF] pointers to each head's first half, each product
    rounded to x's dtype as the bf16 torch ops round."""
    dt = ptrs.dtype.element_ty
    x1 = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptrs + HALF, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, ((x1 * cos).to(dt).to(tl.float32) - (x2 * sin).to(dt).to(tl.float32)).to(dt), mask=mask)
    tl.store(ptrs + HALF, ((x2 * cos).to(dt).to(tl.float32) + (x1 * sin).to(dt).to(tl.float32)).to(dt), mask=mask)


@triton.jit
def rope_kernel(
    q_ptr, k_ptr, start_pos, log2_theta,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, Q_PAD: tl.constexpr, KV_PAD: tl.constexpr,
):
    """In place for token t at position start_pos + t: RoPE on every head of q row t ([T, Q_HEADS * D]) and of
    k cache row start_pos + t ([max_seq, KV_HEADS * D]); heads are padded to powers of two and masked."""
    t = tl.program_id(0)
    pos = start_pos + t
    HALF: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, HALF)
    inv_freq = tl.exp2(-((2 * i).to(tl.float32) / HEAD_DIM) * log2_theta)  # theta^(-2i/d)
    angle = pos.to(tl.float32) * inv_freq
    dt = q_ptr.dtype.element_ty
    cos = tl.cos(angle).to(dt).to(tl.float32)[None, :]
    sin = tl.sin(angle).to(dt).to(tl.float32)[None, :]
    hq = tl.arange(0, Q_PAD)[:, None]
    _rotate_half(q_ptr + t * (Q_HEADS * HEAD_DIM) + hq * HEAD_DIM + i[None, :], hq < Q_HEADS, cos, sin, HALF)
    hk = tl.arange(0, KV_PAD)[:, None]
    _rotate_half(k_ptr + pos * (KV_HEADS * HEAD_DIM) + hk * HEAD_DIM + i[None, :], hk < KV_HEADS, cos, sin, HALF)


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, start_pos, T, sm_scale,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Causal attention of query head program_id(0) at positions start_pos + [0, T) (block program_id(1)) over
    cache rows [0, start_pos + T) of its KV head. q and o are [T, Q_HEADS * D]; k and v are [max_seq, KV_HEADS * D]."""
    head = tl.program_id(0)
    kv_col = (head // (Q_HEADS // KV_HEADS)) * HEAD_DIM
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + m[:, None] * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM + d[None, :], mask=m[:, None] < T, other=0.0)
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    end = start_pos + tl.minimum(T, (tl.program_id(1) + 1) * BLOCK_M)
    for n0 in range(0, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + n[None, :] * (KV_HEADS * HEAD_DIM) + kv_col + d[:, None], mask=n[None, :] < end, other=0.0)
        s = tl.dot(q, kt) * sm_scale
        s = tl.where((n[None, :] < end) & (n[None, :] <= start_pos + m[:, None]), s, float("-inf"))
        new_max = tl.maximum(row_max, tl.max(s, axis=1))
        p = tl.exp(s - new_max[:, None])
        rescale = tl.exp(row_max - new_max)
        denom = denom * rescale + tl.sum(p, axis=1)
        v = tl.load(v_ptr + n[:, None] * (KV_HEADS * HEAD_DIM) + kv_col + d[None, :], mask=n[:, None] < end, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * rescale[:, None])
        row_max = new_max
    o = o_ptr + m[:, None] * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM + d[None, :]
    tl.store(o, (acc / denom[:, None]).to(o_ptr.dtype.element_ty), mask=m[:, None] < T)


@triton.jit
def chunk_max_kernel(x_ptr, max_ptr, idx_ptr, row, N, BLOCK: tl.constexpr):
    """Program c: max_ptr[c], idx_ptr[c] = the maximum of x[row, c * BLOCK:(c + 1) * BLOCK] and its first index."""
    c = tl.program_id(0)
    cols = c * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + row * N + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
    tl.store(max_ptr + c, tl.max(v, axis=0))
    tl.store(idx_ptr + c, c * BLOCK + tl.argmax(v, axis=0))


@triton.jit
def pick_max_kernel(max_ptr, idx_ptr, out_ptr, CHUNKS, BLOCK: tl.constexpr):
    """out[0] = idx[c] of the first chunk c holding the largest max: the first index of the row's maximum."""
    c = tl.arange(0, BLOCK)
    v = tl.load(max_ptr + c, mask=c < CHUNKS, other=float("-inf"))
    tl.store(out_ptr, tl.load(idx_ptr + tl.argmax(v, axis=0)))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def _tiles(M):
    """Weight-streaming tiles: 16 rows (tl.dot's least) for decode, 32 for a prompt."""
    return dict(BLOCK_M=16 if M <= 16 else 32, BLOCK_N=64, BLOCK_K=64)


def embedding(g, ids, table, out):
    H = table.shape[1]
    g.launch(embedding_kernel, (ids.shape[0],), ids_ptr=ids, table_ptr=table, out_ptr=out, H=H, BLOCK=triton.next_power_of_2(H))


def rmsnorm(g, h, res, weight, y, row0=0):
    """y = rmsnorm(h[row0:] (+= res)) * weight over y's rows."""
    N = weight.shape[0]
    g.launch(
        rmsnorm_kernel, (y.shape[0],), h_ptr=h, res_ptr=res if res is not None else h, w_ptr=weight, y_ptr=y,
        row0=row0, N=N, eps=EPS, RESIDUAL=res is not None, BLOCK=triton.next_power_of_2(N),
    )


def linear(g, x, weight, y, bias=None, y_row0=0):
    M, K = x.shape
    N = weight.shape[0]
    tiles = _tiles(M)
    grid = (triton.cdiv(M, tiles["BLOCK_M"]), triton.cdiv(N, tiles["BLOCK_N"]))
    g.launch(
        linear_kernel, grid, x_ptr=x, w_ptr=weight, b_ptr=bias if bias is not None else weight, y_ptr=y,
        y_row0=y_row0, M=M, N=N, K=K, HAS_BIAS=bias is not None, **tiles,
    )


def swiglu(g, x, w_gate, w_up, y):
    M, K = x.shape
    N = w_gate.shape[0]
    tiles = _tiles(M)
    grid = (triton.cdiv(M, tiles["BLOCK_M"]), triton.cdiv(N, tiles["BLOCK_N"]))
    g.launch(swiglu_kernel, grid, x_ptr=x, wg_ptr=w_gate, wu_ptr=w_up, y_ptr=y, M=M, N=N, K=K, **tiles)


def rope(g, q, k_cache, start_pos):
    g.launch(
        rope_kernel, (q.shape[0],), q_ptr=q, k_ptr=k_cache, start_pos=start_pos, log2_theta=math.log2(ROPE_THETA),
        Q_HEADS=Q_HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM,
        Q_PAD=triton.next_power_of_2(Q_HEADS), KV_PAD=triton.next_power_of_2(KV_HEADS),
    )


def attention(g, q, k_cache, v_cache, out, start_pos):
    T = q.shape[0]
    block_m = 16 if T <= 16 else 32
    g.launch(
        attention_kernel, (Q_HEADS, triton.cdiv(T, block_m)),
        q_ptr=q, k_ptr=k_cache, v_ptr=v_cache, o_ptr=out, start_pos=start_pos, T=T, sm_scale=HEAD_DIM**-0.5,
        Q_HEADS=Q_HEADS, KV_HEADS=KV_HEADS, HEAD_DIM=HEAD_DIM, BLOCK_M=block_m, BLOCK_N=32,
    )


def argmax(g, logits, out, row, tag):
    """Two launches: each 4096-wide chunk's maximum and first index, then the first chunk holding the largest."""
    N = logits.shape[1]
    chunks = triton.cdiv(N, 4096)
    maxes = g.buffer(f"{tag}.chunk_max", "fp32", (chunks,), "activation")
    idxs = g.buffer(f"{tag}.chunk_idx", "i32", (chunks,), "activation")
    g.launch(chunk_max_kernel, (chunks,), x_ptr=logits, max_ptr=maxes, idx_ptr=idxs, row=row, N=N, BLOCK=4096)
    g.launch(pick_max_kernel, (1,), max_ptr=maxes, idx_ptr=idxs, out_ptr=out, CHUNKS=chunks, BLOCK=triton.next_power_of_2(chunks))


def layer(g, i, h, prev_mlp, T, start_pos, dtype, max_seq, tag):
    """One decoder layer on the residual stream h [T, HIDDEN]; prev_mlp is the previous layer's MLP output still to
    be added to h (None for the first layer)."""
    p = lambda name, *shape: g.buffer(f"model.layers.{i}.{name}", dtype, shape, "weight")
    act = lambda name, width: g.buffer(f"{tag}.{name}", dtype, (T, width), "activation")
    QD, KVD = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    k_cache = g.buffer(f"kv_cache.{i}.k", dtype, (max_seq, KVD), "state")
    v_cache = g.buffer(f"kv_cache.{i}.v", dtype, (max_seq, KVD), "state")
    x, q, attn, o = act("x", HIDDEN), act("q", QD), act("attn", QD), act("o", HIDDEN)

    rmsnorm(g, h, prev_mlp, p("input_layernorm.weight", HIDDEN), x)
    for name, y, width, row0 in (("q", q, QD, 0), ("k", k_cache, KVD, start_pos), ("v", v_cache, KVD, start_pos)):
        proj = f"self_attn.{name}_proj"
        linear(g, x, p(f"{proj}.weight", width, HIDDEN), y, bias=p(f"{proj}.bias", width), y_row0=row0)
    rope(g, q, k_cache, start_pos)
    attention(g, q, k_cache, v_cache, attn, start_pos)
    linear(g, attn, p("self_attn.o_proj.weight", HIDDEN, QD), o)

    rmsnorm(g, h, o, p("post_attention_layernorm.weight", HIDDEN), x)
    gated, mlp = act("gated", INTER), act("mlp", HIDDEN)
    swiglu(g, x, p("mlp.gate_proj.weight", INTER, HIDDEN), p("mlp.up_proj.weight", INTER, HIDDEN), gated)
    linear(g, gated, p("mlp.down_proj.weight", HIDDEN, INTER), mlp)
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
    rmsnorm(g, h, mlp, g.buffer("model.norm.weight", dtype, (HIDDEN,), "weight"), x, row0=T - rows)
    logits = g.buffer("logits", dtype, (rows, VOCAB), "output")
    linear(g, x, g.buffer("lm_head.weight", dtype, (VOCAB, HIDDEN), "weight"), logits)
    argmax(g, logits, g.buffer("next_token", "i32", (1,), "state"), rows - 1, tag)
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
