"""Whisper tiny (openai/whisper-tiny) in pure Triton: the encoder and one decoder step, fp32.

Weights are the raw checkpoint tensors by their safetensors names, unpacked. The encoder runs both 3-tap
convolutions as implicit GEMMs over the ``[1, 80, 3000]`` log-mel input (the second strided by 2), adds the
fixed sinusoidal position table (``model.encoder.embed_positions.weight``, sinusoids(1500, 384)) in the second
one's epilogue, then 4 pre-LN layers and a final layer norm into the "encoder_hidden_states" state ``[1500, 384]``.
The decoder graph reads that state, embeds the prompt ``decoder_input_ids`` (e.g. the start-of-transcript
tokens) with learned positions, runs 4 pre-LN layers of causal self-attention, cross-attention to the encoder
states and an MLP, and writes the last token's logits through the head tied to ``model.decoder.embed_tokens``
plus the greedy next token.
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
# Config (openai/whisper-tiny config.json)
# ---------------------------------------------------------------------------------------------------

D = 384
HEADS = 6
HEAD_DIM = 64
FFN = 1536
ENC_LAYERS = 4
DEC_LAYERS = 4
MELS = 80
FRAMES = 3000
SRC_POS = 1500  # max_source_positions: FRAMES / 2
TGT_POS = 448  # max_target_positions
VOCAB = 51865
EPS = 1e-5
SOT = (50258, 50259, 50359, 50363)  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>


# ---------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def gelu(x):
    """The exact (erf) GELU."""
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def conv1d_gelu_kernel(
    x_ptr, w_ptr, b_ptr, pos_ptr, y_ptr, L_in, L_out, C_in, N, stride_xc, stride_xt,
    STRIDE: tl.constexpr, HAS_POS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[t, co] = gelu(sum_{ci, j} w[co, ci, j] * x[ci, t * STRIDE + j - 1] + b[co]) (+ pos[t, co] if ``HAS_POS``):
    nn.Conv1d with 3 taps and padding 1 as an implicit GEMM whose column k is (ci, j) of the raw ``[N, C_in, 3]``
    weight. x element (ci, l) is at x + ci * stride_xc + l * stride_xt; y is ``[L_out, N]`` (time-major)."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < L_out
    n_mask = offs_n < N
    K = C_in * 3
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        l = offs_m[:, None] * STRIDE + (k % 3)[None, :] - 1
        a_mask = m_mask[:, None] & k_mask[None, :] & (l >= 0) & (l < L_in)
        a = tl.load(x_ptr + (k // 3)[None, :] * stride_xc + l * stride_xt, mask=a_mask, other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = gelu(acc + tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :])
    mask = m_mask[:, None] & n_mask[None, :]
    if HAS_POS:
        acc += tl.load(pos_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def embedding_kernel(ids_ptr, tok_ptr, pos_ptr, h_ptr, N, BLOCK_N: tl.constexpr):
    """h[t] = tok[ids[t]] + pos[t] for token t = program_id(0)."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(tok_ptr + tl.load(ids_ptr + t) * N + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(pos_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
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
def linear_kernel(
    x_ptr, w_ptr, b_ptr, res_ptr, y_ptr, M, N, K,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = act(x[M, K] @ w[N, K]^T (+ b if ``HAS_BIAS``)) (+ res[M, N] if ``RESIDUAL``), w the raw
    ``nn.Linear`` weight; act the exact GELU if ``GELU``, else the identity."""
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
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if GELU:
        acc = gelu(acc)
    mask = m_mask[:, None] & n_mask[None, :]
    if RESIDUAL:
        acc += tl.load(res_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, Tq, Tk, sm_scale,
    CAUSAL: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Flash attention (online softmax) of head program_id(1), query rows block program_id(0), over the Tk keys;
    query m sees keys n <= m if ``CAUSAL``. q and o are ``[Tq, HEADS * HEAD_DIM]``, k and v ``[Tk, HEADS * HEAD_DIM]``."""
    DM: tl.constexpr = HEADS * HEAD_DIM
    head = tl.program_id(1)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < Tq
    q = tl.load(q_ptr + offs_m[:, None] * DM + head * HEAD_DIM + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        n = n0 + offs_n
        n_mask = n < Tk
        k = tl.load(k_ptr + n[None, :] * DM + head * HEAD_DIM + offs_d[:, None], mask=n_mask[None, :], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * sm_scale
        keep = n_mask[None, :]
        if CAUSAL:
            keep = keep & (n[None, :] <= offs_m[:, None])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + n[:, None] * DM + head * HEAD_DIM + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee")
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * DM + head * HEAD_DIM + offs_d[None, :], out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


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
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def layernorm(g, x, prefix, y, row0=0):
    """torch.nn.LayerNorm(D) with ``<prefix>.weight`` / ``<prefix>.bias`` over y's rows of x from row0."""
    g.launch(
        layernorm_kernel, (y.shape[0],), x_ptr=x, g_ptr=g.buffer(f"{prefix}.weight", x.dtype, (D,), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (D,), "weight"), y_ptr=y, row0=row0, N=D, eps=EPS,
        BLOCK_N=triton.next_power_of_2(D),
    )
    return y


def linear(g, x, w, b, y, gelu=False, res=None):
    """y = act(x @ w^T (+ b)) (+ res)."""
    m, k = x.shape
    n = y.shape[1]
    block_m = 16 if m <= 16 else 64
    g.launch(
        linear_kernel, (triton.cdiv(m, block_m), triton.cdiv(n, 64)),
        x_ptr=x, w_ptr=w, b_ptr=b if b is not None else w, res_ptr=res if res is not None else y, y_ptr=y,
        M=m, N=n, K=k, HAS_BIAS=b is not None, GELU=gelu, RESIDUAL=res is not None, BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def attention(g, prefix, x, kv, h, act, causal=False):
    """h + out_proj(softmax(q k^T / 8) v) of WhisperAttention ``<prefix>`` (k_proj has no bias), with q from x
    and k, v from kv (x itself for self-attention)."""
    dtype = x.dtype
    w = lambda name, *shape: g.buffer(f"{prefix}.{name}", dtype, shape, "weight")
    tq, tk = x.shape[0], kv.shape[0]
    q = linear(g, x, w("q_proj.weight", D, D), w("q_proj.bias", D), act("q", tq))
    k = linear(g, kv, w("k_proj.weight", D, D), None, act("k", tk))
    v = linear(g, kv, w("v_proj.weight", D, D), w("v_proj.bias", D), act("v", tk))
    o = act("attn", tq)
    block_m = 16 if tq <= 16 else 64
    g.launch(
        attention_kernel, (triton.cdiv(tq, block_m), HEADS), num_stages=2, q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=o,
        Tq=tq, Tk=tk, sm_scale=HEAD_DIM**-0.5, CAUSAL=causal, HEADS=HEADS, HEAD_DIM=HEAD_DIM, BLOCK_M=block_m, BLOCK_N=64,
    )
    return linear(g, o, w("out_proj.weight", D, D), w("out_proj.bias", D), act("out", tq), res=h)


def mlp(g, prefix, h, act):
    """h + fc2(gelu(fc1(final_layer_norm(h)))) of layer ``<prefix>``."""
    t = h.shape[0]
    w = lambda name, *shape: g.buffer(f"{prefix}.{name}", h.dtype, shape, "weight")
    x = layernorm(g, h, f"{prefix}.final_layer_norm", act("final_layer_norm", t))
    x = linear(g, x, w("fc1.weight", FFN, D), w("fc1.bias", FFN), act("fc1", t, FFN), gelu=True)
    return linear(g, x, w("fc2.weight", D, FFN), w("fc2.bias", D), act("fc2", t), res=h)


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------


def encoder(dtype="fp32"):
    """WhisperEncoder: ``input_features`` [1, 80, 3000] -> the "encoder_hidden_states" state [1500, 384]."""
    g = Graph(f"whisper_tiny.encoder(dtype={dtype})")
    w = lambda name, *shape: g.buffer(f"model.encoder.{name}", dtype, shape, "weight")
    feats = g.buffer("input_features", dtype, (1, MELS, FRAMES), "input")
    conv1 = g.buffer("encoder.conv1.out", dtype, (FRAMES, D), "activation")
    h = g.buffer("encoder.embed", dtype, (SRC_POS, D), "activation")
    for x, wname, y, l_in, c_in, strides, stride in (
        (feats, "conv1", conv1, FRAMES, MELS, (FRAMES, 1), 1),
        (conv1, "conv2", h, FRAMES, D, (1, D), 2),
    ):
        g.launch(
            conv1d_gelu_kernel, (triton.cdiv(y.shape[0], 64), triton.cdiv(D, 64)),
            x_ptr=x, w_ptr=w(f"{wname}.weight", D, c_in, 3), b_ptr=w(f"{wname}.bias", D),
            pos_ptr=w("embed_positions.weight", SRC_POS, D), y_ptr=y, L_in=l_in, L_out=y.shape[0], C_in=c_in, N=D,
            stride_xc=strides[0], stride_xt=strides[1], STRIDE=stride, HAS_POS=stride == 2, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )
    for i in range(ENC_LAYERS):
        prefix = f"model.encoder.layers.{i}"
        act = lambda name, t, width=D: g.buffer(f"encoder.layers.{i}.{name}", dtype, (t, width), "activation")
        x = layernorm(g, h, f"{prefix}.self_attn_layer_norm", act("self_attn_layer_norm", SRC_POS))
        h = attention(g, f"{prefix}.self_attn", x, x, h, lambda name, t: act(f"self_attn.{name}", t))
        h = mlp(g, prefix, h, act)
    layernorm(g, h, "model.encoder.layer_norm", g.buffer("encoder_hidden_states", dtype, (SRC_POS, D), "state"))
    return g


def decoder(prompt_len=len(SOT), dtype="fp32"):
    """WhisperDecoder on ``decoder_input_ids`` [prompt_len] at positions [0, prompt_len), attending to the
    "encoder_hidden_states" state, then the tied head: the last token's ``logits`` and ``next_token``."""
    g = Graph(f"whisper_tiny.decoder(prompt_len={prompt_len}, dtype={dtype})")
    t = prompt_len
    ids = g.buffer("decoder_input_ids", "i32", (t,), "input")
    enc = g.buffer("encoder_hidden_states", dtype, (SRC_POS, D), "state")
    tok = g.buffer("model.decoder.embed_tokens.weight", dtype, (VOCAB, D), "weight")
    h = g.buffer("decoder.embed", dtype, (t, D), "activation")
    g.launch(embedding_kernel, (t,), ids_ptr=ids, tok_ptr=tok,
             pos_ptr=g.buffer("model.decoder.embed_positions.weight", dtype, (TGT_POS, D), "weight"), h_ptr=h, N=D,
             BLOCK_N=triton.next_power_of_2(D))
    for i in range(DEC_LAYERS):
        prefix = f"model.decoder.layers.{i}"
        act = lambda name, rows, width=D: g.buffer(f"decoder.layers.{i}.{name}", dtype, (rows, width), "activation")
        x = layernorm(g, h, f"{prefix}.self_attn_layer_norm", act("self_attn_layer_norm", t))
        h = attention(g, f"{prefix}.self_attn", x, x, h, lambda name, rows: act(f"self_attn.{name}", rows), causal=True)
        x = layernorm(g, h, f"{prefix}.encoder_attn_layer_norm", act("encoder_attn_layer_norm", t))
        h = attention(g, f"{prefix}.encoder_attn", x, enc, h, lambda name, rows: act(f"encoder_attn.{name}", rows))
        h = mlp(g, prefix, h, act)
    x = layernorm(g, h, "model.decoder.layer_norm", g.buffer("decoder.layer_norm", dtype, (1, D), "activation"), row0=t - 1)
    logits = linear(g, x, tok, None, g.buffer("logits", dtype, (1, VOCAB), "output"))
    g.launch(argmax_kernel, (1,), num_warps=8, x_ptr=logits, out_ptr=g.buffer("next_token", "i32", (1,), "output"), N=VOCAB, BLOCK=4096)
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [encoder(), decoder()]
