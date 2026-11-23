"""BERT-base, uncased (google-bert/bert-base-uncased) in pure Triton: the encoder and its pooler, fp32.

Weights are the raw checkpoint tensors by their safetensors names (``bert.``-prefixed, layer norms as
``gamma`` / ``beta``), unpacked. The input is a batch of sequences padded to one length: ``input_ids`` and
``token_type_ids`` ``[B, T]`` and ``lengths`` ``[B]``, the real tokens of each sequence (HF's
``attention_mask[b, t] = t < lengths[b]``). Every token position is computed, padding included; attention
only reaches a sequence's real keys. Outputs: ``last_hidden_state`` ``[B, T, 768]`` and ``pooler_output``
``[B, 768]`` (tanh of a linear on each sequence's first token).
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
# Config (google-bert/bert-base-uncased config.json)
# ---------------------------------------------------------------------------------------------------

HIDDEN = 768
LAYERS = 12
HEADS = 12
HEAD_DIM = 64
INTER = 3072
VOCAB = 30522
MAX_POS = 512
TYPES = 2
EPS = 1e-12


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def embed_layernorm_kernel(
    ids_ptr, types_ptr, word_ptr, pos_ptr, type_ptr, g_ptr, b_ptr, y_ptr, T, N, eps, BLOCK_N: tl.constexpr,
):
    """Row r = program_id(0) (token t = r % T): y[r] = LayerNorm(word[ids[r]] + type[types[r]] + pos[t])."""
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    word = tl.load(ids_ptr + r)
    kind = tl.load(types_ptr + r)
    x = tl.load(word_ptr + word * N + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(type_ptr + kind * N + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(pos_ptr + (r % T) * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * N + offs, (d * rstd * g + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, g_ptr, b_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    """y[r] = (x[r] - mean) / sqrt(var + eps) * g + b over the N columns of row r = program_id(0), with the
    biased variance as torch.nn.LayerNorm."""
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + r * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * N + offs, (d * rstd * g + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def tanh(x):
    """tanh from exp alone (Triton has no tanh outside libdevice): the odd extension of (1 - e^-2x) / (1 + e^-2x)."""
    t = tl.exp(-2.0 * tl.abs(x))
    y = (1.0 - t) / (1.0 + t)
    return tl.where(x < 0.0, -y, y)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, res_ptr, y_ptr, M, N, K, x_stride,
    ACT: tl.constexpr, RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ w[N, K]^T + b) (+ res[M, N] if ``RESIDUAL``), w the raw ``nn.Linear`` weight and
    x rows x_stride elements apart. act: "gelu" the exact (erf) GELU, "tanh", or "" the identity."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        a = tl.load(x_ptr + offs_m[:, None] * x_stride + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if ACT == "gelu":
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    elif ACT == "tanh":
        acc = tanh(acc)
    mask = m_mask[:, None] & n_mask[None, :]
    if RESIDUAL:
        acc += tl.load(res_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, len_ptr, o_ptr, T, sm_scale,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Bidirectional softmax attention of head program_id(1) of sequence program_id(2), query rows block
    program_id(0), over the sequence's first len[b] keys (the rest is padding). q, k, v and o are
    ``[B * T, HEADS * HEAD_DIM]``; all T keys fit one BLOCK_T tile, so the softmax is taken over the whole row."""
    D: tl.constexpr = HEADS * HEAD_DIM
    head = tl.program_id(1)
    seq = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < T
    t_mask = offs_t < tl.load(len_ptr + seq)
    base = seq * T * D + head * HEAD_DIM
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    k = tl.load(k_ptr + base + offs_t[None, :] * D + offs_d[:, None], mask=t_mask[None, :], other=0.0)
    s = tl.dot(q, k, input_precision="ieee") * sm_scale
    s = tl.where(t_mask[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    l = tl.sum(p, axis=1)
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :], mask=t_mask[:, None], other=0.0)
    out = tl.dot(p.to(v.dtype), v, input_precision="ieee") / l[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def layernorm(g, x, prefix, y):
    """torch.nn.LayerNorm(HIDDEN) with ``<prefix>.gamma`` / ``<prefix>.beta`` over the rows of x [M, HIDDEN]."""
    g.launch(
        layernorm_kernel, (x.shape[0],), x_ptr=x, g_ptr=g.buffer(f"{prefix}.gamma", x.dtype, (HIDDEN,), "weight"),
        b_ptr=g.buffer(f"{prefix}.beta", x.dtype, (HIDDEN,), "weight"), y_ptr=y, N=HIDDEN, eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
    )
    return y


def linear(g, x, prefix, y, act="", res=None, x_stride=None):
    """y = act(x @ <prefix>.weight^T + <prefix>.bias) (+ res) over y's rows (x rows x_stride apart)."""
    k = x.shape[-1]
    m, n = y.shape[0], y.shape[-1]
    block_m = 16 if m <= 16 else 64
    g.launch(
        linear_kernel, (triton.cdiv(m, block_m), triton.cdiv(n, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (n, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (n,), "weight"), res_ptr=res if res is not None else y, y_ptr=y,
        M=m, N=n, K=k, x_stride=x_stride or k, ACT=act, RESIDUAL=res is not None, BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def encoder_layer(g, i, h, lengths, batch, seq, dtype, out=None):
    """BertLayer (post-LN): a = LN(h + attn(h)), then LN(a + ffn(a)), written to ``out`` if given."""
    prefix = f"bert.encoder.layer.{i}"
    act = lambda name, width: g.buffer(f"{prefix}.{name}", dtype, (batch * seq, width), "activation")
    q, k, v = (linear(g, h, f"{prefix}.attention.self.{n}", act(f"attention.self.{n}.out", HIDDEN)) for n in ("query", "key", "value"))
    ctx = act("attention.self.out", HIDDEN)
    g.launch(
        attention_kernel, (triton.cdiv(seq, 32), HEADS, batch), q_ptr=q, k_ptr=k, v_ptr=v, len_ptr=lengths, o_ptr=ctx,
        T=seq, sm_scale=HEAD_DIM**-0.5, HEADS=HEADS, HEAD_DIM=HEAD_DIM, BLOCK_M=32, BLOCK_T=triton.next_power_of_2(seq),
    )
    a = linear(g, ctx, f"{prefix}.attention.output.dense", act("attention.output.dense.out", HIDDEN), res=h)
    a = layernorm(g, a, f"{prefix}.attention.output.LayerNorm", act("attention.output", HIDDEN))
    x = linear(g, a, f"{prefix}.intermediate.dense", act("intermediate.out", INTER), act="gelu")
    x = linear(g, x, f"{prefix}.output.dense", act("output.dense.out", HIDDEN), res=a)
    return layernorm(g, x, f"{prefix}.output.LayerNorm", out if out is not None else act("output", HIDDEN))


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def forward(batch=2, seq=128, dtype="fp32"):
    """BertModel on a padded batch -> ``last_hidden_state`` and ``pooler_output``. Dropout is the identity in eval."""
    g = Graph(f"bert_base.forward(batch={batch}, seq={seq}, dtype={dtype})")
    ids = g.buffer("input_ids", "i32", (batch, seq), "input")
    types = g.buffer("token_type_ids", "i32", (batch, seq), "input")
    lengths = g.buffer("lengths", "i32", (batch,), "input")
    w = lambda name, *shape: g.buffer(f"bert.embeddings.{name}", dtype, shape, "weight")
    h = g.buffer("bert.embeddings.out", dtype, (batch * seq, HIDDEN), "activation")
    g.launch(
        embed_layernorm_kernel, (batch * seq,), ids_ptr=ids, types_ptr=types,
        word_ptr=w("word_embeddings.weight", VOCAB, HIDDEN), pos_ptr=w("position_embeddings.weight", MAX_POS, HIDDEN),
        type_ptr=w("token_type_embeddings.weight", TYPES, HIDDEN), g_ptr=w("LayerNorm.gamma", HIDDEN),
        b_ptr=w("LayerNorm.beta", HIDDEN), y_ptr=h, T=seq, N=HIDDEN, eps=EPS, BLOCK_N=triton.next_power_of_2(HIDDEN),
    )
    last = g.buffer("last_hidden_state", dtype, (batch, seq, HIDDEN), "output")
    for i in range(LAYERS):
        h = encoder_layer(g, i, h, lengths, batch, seq, dtype, out=last if i == LAYERS - 1 else None)
    linear(g, last, "bert.pooler.dense", g.buffer("pooler_output", dtype, (batch, HIDDEN), "output"), act="tanh",
           x_stride=seq * HIDDEN)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=2, seq=128, dtype="fp32")]
