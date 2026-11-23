"""GPT-2 124M (openai-community/gpt2) in pure Triton: greedy text generation as prefill and decode launch graphs, fp32.

Weights are the raw checkpoint tensors by their safetensors names, unpacked: every projection is a GPT-2
``Conv1D``, whose weight is ``[in, out]`` (y = x @ w + b), and ``attn.c_attn`` projects q | k | v in one GEMM.
The output head reads ``wte.weight`` ``[VOCAB, 768]`` (tied embeddings). Each layer's K/V cache is a "state"
buffer of ``[max_seq, 768]`` rows, so prefill and decode share it, and the argmax writes the next token into
the "next_token" state that the following decode graph embeds.
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
# Config (openai-community/gpt2 config.json)
# ---------------------------------------------------------------------------------------------------

HIDDEN = 768
LAYERS = 12
HEADS = 12
HEAD_DIM = 64
INTER = 3072
VOCAB = 50257
MAX_SEQ = 1024  # n_positions
EPS = 1e-5


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def embedding_kernel(ids_ptr, wte_ptr, wpe_ptr, h_ptr, start_pos, N, BLOCK_N: tl.constexpr):
    """h[t] = wte[ids[t]] + wpe[start_pos + t] for token t = program_id(0)."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    tok = tl.load(ids_ptr + t)
    x = tl.load(wte_ptr + tok * N + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(wpe_ptr + (start_pos + t) * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + t * N + offs, x.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, g_ptr, b_ptr, y_ptr, row0, N, eps, BLOCK_N: tl.constexpr):
    """y[t] = (x[row0 + t] - mean) / sqrt(var + eps) * g + b for t = program_id(0), with the biased variance as
    torch.nn.LayerNorm."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + (row0 + t) * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + t * N + offs, (d * rstd * g + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def tanh(x):
    """tanh from exp alone (Triton has no tanh outside libdevice): the odd extension of (1 - e^-2x) / (1 + e^-2x)."""
    t = tl.exp(-2.0 * tl.abs(x))
    y = (1.0 - t) / (1.0 + t)
    return tl.where(x < 0.0, -y, y)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, res_ptr, y_ptr, M, N, K, stride_wk, stride_wn,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ W + b) (+ res[M, N] if ``RESIDUAL``), W[k, n] at w + k * stride_wk + n * stride_wn
    (a Conv1D weight [K, N]: (N, 1); the tied head's [N, K] embedding: (1, K)). act is GPT-2's tanh GELU
    (``gelu_new``) if ``GELU``, else the identity."""
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
        b = tl.load(w_ptr + k[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if GELU:
        acc = 0.5 * acc * (1.0 + tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))
    mask = m_mask[:, None] & n_mask[None, :]
    if RESIDUAL:
        acc += tl.load(res_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def kv_write_kernel(qkv_ptr, k_ptr, v_ptr, start_pos, D, BLOCK_D: tl.constexpr):
    """Cache rows start_pos + t of k and v ``[max_seq, D]`` = the k and v thirds of qkv[t] ``[T, 3 * D]``,
    t = program_id(0)."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    row = qkv_ptr + t * (3 * D) + offs
    tl.store(k_ptr + (start_pos + t) * D + offs, tl.load(row + D, mask=mask), mask=mask)
    tl.store(v_ptr + (start_pos + t) * D + offs, tl.load(row + 2 * D, mask=mask), mask=mask)


@triton.jit
def attention_kernel(
    qkv_ptr, k_ptr, v_ptr, o_ptr, start_pos, T, sm_scale,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Causal flash attention (online softmax) of head program_id(1) for the queries at positions start_pos +
    [0, T) in the q third of qkv ``[T, 3 * D]``, over cache rows [0, start_pos + T) of k and v ``[max_seq, D]``;
    o is ``[T, D]``."""
    D: tl.constexpr = HEADS * HEAD_DIM
    head = tl.program_id(1)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < T
    q_pos = start_pos + offs_m
    q = tl.load(qkv_ptr + offs_m[:, None] * (3 * D) + head * HEAD_DIM + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    kv_end = start_pos + T
    for n0 in range(0, kv_end, BLOCK_N):
        n = n0 + offs_n
        n_mask = n < kv_end
        k = tl.load(k_ptr + n[None, :] * D + head * HEAD_DIM + offs_d[:, None], mask=n_mask[None, :], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * sm_scale
        s = tl.where(n_mask[None, :] & (n[None, :] <= q_pos[:, None]), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + n[:, None] * D + head * HEAD_DIM + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee")
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * D + head * HEAD_DIM + offs_d[None, :], out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def argmax_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """out[0] = the first index of the maximum of x[:N]."""
    offs = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    best_i = tl.zeros((BLOCK,), dtype=tl.int32)
    for n0 in range(0, N, BLOCK):
        v = tl.load(x_ptr + n0 + offs, mask=n0 + offs < N, other=float("-inf")).to(tl.float32)
        better = v > best
        best = tl.where(better, v, best)
        best_i = tl.where(better, n0 + offs, best_i)
    top = tl.max(best, axis=0)
    tl.store(out_ptr, tl.min(tl.where(best == top, best_i, N), axis=0))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def layernorm(g, x, prefix, y, row0=0):
    """torch.nn.LayerNorm(HIDDEN) with ``<prefix>.weight`` / ``<prefix>.bias`` over y's rows of x from row0."""
    g.launch(
        layernorm_kernel, (y.shape[0],), x_ptr=x, g_ptr=g.buffer(f"{prefix}.weight", x.dtype, (HIDDEN,), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (HIDDEN,), "weight"), y_ptr=y, row0=row0, N=HIDDEN, eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
    )
    return y


def linear(g, x, w, b, y, gelu=False, res=None, tied=False):
    """y = act(x @ w + b) (+ res) for a Conv1D weight w [K, N]; tied: y = x @ w^T for w [N, K], no bias."""
    m, k = x.shape
    n = y.shape[1]
    block_m = 16 if m <= 16 else 64
    g.launch(
        linear_kernel, (triton.cdiv(m, block_m), triton.cdiv(n, 64)),
        x_ptr=x, w_ptr=w, b_ptr=b if b is not None else w, res_ptr=res if res is not None else y, y_ptr=y,
        M=m, N=n, K=k, stride_wk=1 if tied else n, stride_wn=k if tied else 1,
        HAS_BIAS=b is not None, GELU=gelu, RESIDUAL=res is not None, BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def block(g, i, h, T, start_pos, dtype, max_seq, tag):
    """GPT2Block (pre-LN) on the residual stream h [T, HIDDEN]: h + attn(ln_1(h)), then that + mlp(ln_2(that))."""
    w = lambda name, *shape: g.buffer(f"h.{i}.{name}", dtype, shape, "weight")
    act = lambda name, width: g.buffer(f"{tag}.h.{i}.{name}", dtype, (T, width), "activation")
    k_cache = g.buffer(f"kv_cache.{i}.k", dtype, (max_seq, HIDDEN), "state")
    v_cache = g.buffer(f"kv_cache.{i}.v", dtype, (max_seq, HIDDEN), "state")

    x = layernorm(g, h, f"h.{i}.ln_1", act("ln_1", HIDDEN))
    qkv = linear(g, x, w("attn.c_attn.weight", HIDDEN, 3 * HIDDEN), w("attn.c_attn.bias", 3 * HIDDEN), act("qkv", 3 * HIDDEN))
    g.launch(kv_write_kernel, (T,), qkv_ptr=qkv, k_ptr=k_cache, v_ptr=v_cache, start_pos=start_pos, D=HIDDEN,
             BLOCK_D=triton.next_power_of_2(HIDDEN))
    attn = act("attn", HIDDEN)
    block_m = 16 if T <= 16 else 64
    g.launch(
        attention_kernel, (triton.cdiv(T, block_m), HEADS), num_stages=2, qkv_ptr=qkv, k_ptr=k_cache, v_ptr=v_cache,
        o_ptr=attn, start_pos=start_pos, T=T, sm_scale=HEAD_DIM**-0.5, HEADS=HEADS, HEAD_DIM=HEAD_DIM,
        BLOCK_M=block_m, BLOCK_N=32,
    )
    h = linear(g, attn, w("attn.c_proj.weight", HIDDEN, HIDDEN), w("attn.c_proj.bias", HIDDEN), act("attn_residual", HIDDEN), res=h)

    x = layernorm(g, h, f"h.{i}.ln_2", act("ln_2", HIDDEN))
    x = linear(g, x, w("mlp.c_fc.weight", HIDDEN, INTER), w("mlp.c_fc.bias", INTER), act("mlp.c_fc", INTER), gelu=True)
    return linear(g, x, w("mlp.c_proj.weight", INTER, HIDDEN), w("mlp.c_proj.bias", HIDDEN), act("out", HIDDEN), res=h)


def forward(g, ids, T, start_pos, dtype, max_seq, tag):
    """Embeds ids [T] at positions start_pos + [0, T), runs every block, and writes the last token's logits plus
    the greedy next token."""
    wte = g.buffer("wte.weight", dtype, (VOCAB, HIDDEN), "weight")
    h = g.buffer(f"{tag}.embed", dtype, (T, HIDDEN), "activation")
    g.launch(embedding_kernel, (T,), ids_ptr=ids, wte_ptr=wte, wpe_ptr=g.buffer("wpe.weight", dtype, (MAX_SEQ, HIDDEN), "weight"),
             h_ptr=h, start_pos=start_pos, N=HIDDEN, BLOCK_N=triton.next_power_of_2(HIDDEN))
    for i in range(LAYERS):
        h = block(g, i, h, T, start_pos, dtype, max_seq, tag)
    x = layernorm(g, h, "ln_f", g.buffer(f"{tag}.ln_f", dtype, (1, HIDDEN), "activation"), row0=T - 1)
    logits = linear(g, x, wte, None, g.buffer("logits", dtype, (1, VOCAB), "output"), tied=True)
    g.launch(argmax_kernel, (1,), num_warps=8, x_ptr=logits, out_ptr=g.buffer("next_token", "i32", (1,), "state"), N=VOCAB, BLOCK=4096)
    return g


def prefill(seq_len, dtype="fp32", max_seq=MAX_SEQ):
    """Prompt input_ids [seq_len] at positions [0, seq_len): fills the KV cache, writes the last row's "logits" and
    "next_token"."""
    g = Graph(f"gpt2.prefill(seq_len={seq_len}, dtype={dtype}, max_seq={max_seq})")
    ids = g.buffer("input_ids", "i32", (seq_len,), "input")
    return forward(g, ids, seq_len, 0, dtype, max_seq, f"prefill{seq_len}")


def decode(start_pos, dtype="fp32", max_seq=MAX_SEQ):
    """One step: embeds "next_token" at position start_pos, then overwrites it with the next one."""
    g = Graph(f"gpt2.decode(start_pos={start_pos}, dtype={dtype}, max_seq={max_seq})")
    ids = g.buffer("next_token", "i32", (1,), "state")
    return forward(g, ids, 1, start_pos, dtype, max_seq, "decode")


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [prefill(seq_len=32), decode(start_pos=32)]
