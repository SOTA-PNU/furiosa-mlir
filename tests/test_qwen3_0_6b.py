"""models/qwen3_0_6b.py: each kernel as a small graph against torch, then the whole model against HF."""

import json
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb

from harness import ROOT, TORCH_DTYPES, load_model, run

M = load_model("models/qwen3_0_6b.py")
DTYPES = ["fp32", "bf16"]
KV = M.KV_HEADS * M.HEAD_DIM
QD = M.Q_HEADS * M.HEAD_DIM


@pytest.fixture(scope="module")
def snap():
    return snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json", "*.safetensors", "*.txt"])


def graph_of(build, **bufs):
    """A graph declaring ``bufs`` (name -> (dtype, shape, kind)) and running ``build(g, **buffers)``."""
    g = M.Graph("test")
    build(g, **{name: g.buffer(name, *spec) for name, spec in bufs.items()})
    return g


def tol(dtype):
    return dict(atol=1e-5, rtol=1e-5) if dtype == "fp32" else dict(atol=1e-2, rtol=1e-2)


def test_config_matches_checkpoint(snap):
    c = json.load(open(f"{snap}/config.json"))
    assert (c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], c["vocab_size"]) == (M.HIDDEN, M.LAYERS, M.INTER, M.VOCAB)
    assert (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) == (M.Q_HEADS, M.KV_HEADS, M.HEAD_DIM)
    assert (c["rope_theta"], c["rms_norm_eps"], c["tie_word_embeddings"]) == (M.ROPE_THETA, M.EPS, True)
    assert c["rope_scaling"] is None and not c["attention_bias"] and c["hidden_act"] == "silu"
    assert abs(M.LOG2_THETA_HI + M.LOG2_THETA_LO - torch.log2(torch.tensor(M.ROPE_THETA, dtype=torch.float64)).item()) < 1e-12


def test_embedding():
    emb = torch.randn(M.VOCAB, M.HIDDEN, device="cuda")
    ids = torch.tensor([0, 5, M.VOCAB - 1, 5], device="cuda", dtype=torch.int32)
    g = graph_of(M.embedding, ids=("i32", (4,), "input"), emb=("fp32", emb.shape, "weight"), out=("fp32", (4, M.HIDDEN), "output"))
    assert torch.equal(run(g, {"ids": ids, "emb": emb})["out"], emb[ids.long()])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_res,row0,rows", [(False, 0, 9), (True, 0, 9), (True, 8, 1)])
def test_rmsnorm(dtype, has_res, row0, rows):
    dt = TORCH_DTYPES[dtype]
    h = torch.randn(9, M.HIDDEN, device="cuda", dtype=dt)
    res = torch.randn(9, M.HIDDEN, device="cuda", dtype=dt)
    norm = Qwen3RMSNorm(M.HIDDEN, eps=M.EPS).cuda().to(dt)
    norm.weight.data.normal_()
    bufs = dict(h=(dtype, h.shape, "input"), w=(dtype, (M.HIDDEN,), "weight"), y=(dtype, (rows, M.HIDDEN), "output"))
    if has_res:
        bufs["res"] = (dtype, res.shape, "input")
    g = graph_of(lambda g, h, w, y, res=None: M.rmsnorm(g, h, res, w, y, row0=row0), **bufs)
    out = run(g, {"h": h.clone(), "res": res, "w": norm.weight.data})
    want_h = h + res if has_res else h
    torch.testing.assert_close(out["h"][row0:], want_h[row0:], **tol(dtype))
    with torch.no_grad():
        torch.testing.assert_close(out["y"], norm(want_h[row0:row0 + rows]), **tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T,start_pos", [(7, 0), (1, 40), (5, 3000)])
def test_qk_norm_rope(snap, dtype, T, start_pos):
    dt = TORCH_DTYPES[dtype]
    max_seq = start_pos + T + 4
    q = torch.randn(T, QD, device="cuda", dtype=dt)
    k_cache = torch.zeros(max_seq, KV, device="cuda", dtype=dt)
    k_cache[start_pos:start_pos + T] = torch.randn(T, KV, device="cuda", dtype=dt)
    q_norm, k_norm = (Qwen3RMSNorm(M.HEAD_DIM, eps=M.EPS).cuda().to(dt) for _ in range(2))
    q_norm.weight.data.normal_()
    k_norm.weight.data.normal_()
    g = graph_of(
        lambda g, q, k, qw, kw: M.qk_norm_rope(g, q, k, qw, kw, start_pos),
        q=(dtype, q.shape, "input"), k=(dtype, k_cache.shape, "state"), qw=(dtype, (M.HEAD_DIM,), "weight"), kw=(dtype, (M.HEAD_DIM,), "weight"),
    )
    out = run(g, {"q": q.clone(), "k": k_cache.clone(), "qw": q_norm.weight.data, "kw": k_norm.weight.data})

    rope = Qwen3RotaryEmbedding(AutoConfig.from_pretrained(snap)).cuda()
    positions = torch.arange(start_pos, start_pos + T, device="cuda")
    with torch.no_grad():
        qh = q_norm(q.view(1, T, M.Q_HEADS, M.HEAD_DIM)).transpose(1, 2)
        kh = k_norm(k_cache[start_pos:start_pos + T].view(1, T, M.KV_HEADS, M.HEAD_DIM)).transpose(1, 2)
        cos, sin = rope(qh, positions[None])
        hf = apply_rotary_pos_emb(qh, kh, cos, sin)
        # The same rotation with exact angles: inv_freq and pos * inv_freq in fp64.
        inv = 1.0 / M.ROPE_THETA ** (torch.arange(0, M.HEAD_DIM, 2, device="cuda", dtype=torch.float64) / M.HEAD_DIM)
        angle = (positions[:, None].double() * inv).repeat(1, 2)
        exact = apply_rotary_pos_emb(qh.double(), kh.double(), angle.cos()[None], angle.sin()[None])
    got = (out["q"], out["k"][start_pos:start_pos + T])
    for got_x, hf_x, exact_x in zip(got, hf, exact):
        hf_x, exact_x = (x.transpose(1, 2).reshape(T, -1) for x in (hf_x, exact_x))
        if dtype == "fp32":
            # HF's inv_freq comes from torch's powf, within an ulp of ours but not equal; a position of
            # thousands magnifies that, so fp32 is held to HF's own error against the exact rotation.
            assert (got_x.double() - exact_x).abs().max() <= 2 * (hf_x.double() - exact_x).abs().max() + 2e-6
        else:
            torch.testing.assert_close(got_x, hf_x, **tol(dtype))
    assert not out["k"][:start_pos].any() and not out["k"][start_pos + T:].any()


def attention_reference(q, k, v, start_pos):
    T = q.shape[0]
    S = start_pos + T
    qh = q.float().view(T, M.Q_HEADS, M.HEAD_DIM).transpose(0, 1)
    group = M.Q_HEADS // M.KV_HEADS
    kh = k[:S].float().view(S, M.KV_HEADS, M.HEAD_DIM).transpose(0, 1).repeat_interleave(group, 0)
    vh = v[:S].float().view(S, M.KV_HEADS, M.HEAD_DIM).transpose(0, 1).repeat_interleave(group, 0)
    scores = qh @ kh.transpose(1, 2) * M.HEAD_DIM**-0.5
    causal = torch.arange(S, device="cuda")[None, :] <= (start_pos + torch.arange(T, device="cuda"))[:, None]
    p = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    return (p @ vh).transpose(0, 1).reshape(T, -1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T,start_pos", [(37, 0), (100, 0), (20, 13), (1, 0), (1, 77)])
def test_attention(dtype, T, start_pos):
    dt = TORCH_DTYPES[dtype]
    max_seq = 128
    q = torch.randn(T, QD, device="cuda", dtype=dt)
    # Rows past start_pos + T hold NaN: reading any of them would poison the output.
    k = torch.full((max_seq, KV), float("nan"), device="cuda", dtype=dt)
    v = torch.full((max_seq, KV), float("nan"), device="cuda", dtype=dt)
    k[:start_pos + T] = torch.randn(start_pos + T, KV, device="cuda", dtype=dt)
    v[:start_pos + T] = torch.randn(start_pos + T, KV, device="cuda", dtype=dt)
    g = graph_of(
        lambda g, q, k, v, o: M.attention(g, q, k, v, o, start_pos),
        q=(dtype, q.shape, "input"), k=(dtype, k.shape, "state"), v=(dtype, v.shape, "state"), o=(dtype, q.shape, "output"),
    )
    got = run(g, {"q": q, "k": k, "v": v})["o"]
    want = attention_reference(q, k, v, start_pos)
    torch.testing.assert_close(got.float(), want, atol=1e-4 if dtype == "fp32" else 2e-2, rtol=1e-4 if dtype == "fp32" else 2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M_rows", [1, 23])
def test_linear_writes_rows_from_offset(dtype, M_rows):
    dt = TORCH_DTYPES[dtype]
    x = torch.randn(M_rows, M.HIDDEN, device="cuda", dtype=dt)
    w = torch.randn(KV, M.HIDDEN, device="cuda", dtype=dt) / M.HIDDEN**0.5
    y = torch.zeros(64, KV, device="cuda", dtype=dt)
    g = graph_of(lambda g, x, w, y: M.linear(g, x, w, y, y_row0=9), x=(dtype, x.shape, "input"), w=(dtype, w.shape, "weight"), y=(dtype, y.shape, "state"))
    got = run(g, {"x": x, "w": w, "y": y})["y"]
    torch.testing.assert_close(got[9:9 + M_rows].float(), x.float() @ w.float().T, **tol(dtype))
    assert not got[:9].any() and not got[9 + M_rows:].any()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M_rows", [1, 23])
def test_swiglu(dtype, M_rows):
    dt = TORCH_DTYPES[dtype]
    x = torch.randn(M_rows, M.HIDDEN, device="cuda", dtype=dt)
    wg = torch.randn(M.INTER, M.HIDDEN, device="cuda", dtype=dt) / M.HIDDEN**0.5
    wu = torch.randn(M.INTER, M.HIDDEN, device="cuda", dtype=dt) / M.HIDDEN**0.5
    g = graph_of(M.swiglu, x=(dtype, x.shape, "input"), w_gate=(dtype, wg.shape, "weight"), w_up=(dtype, wu.shape, "weight"), y=(dtype, (M_rows, M.INTER), "output"))
    got = run(g, {"x": x, "w_gate": wg, "w_up": wu})["y"]
    torch.testing.assert_close(got, F.silu(F.linear(x, wg)) * F.linear(x, wu), **tol(dtype))


@pytest.mark.parametrize("row", [0, 2])
def test_argmax_picks_first_maximum(row):
    logits = torch.randn(3, M.VOCAB, device="cuda")
    logits[row, [70000, 151935, 4097]] = 50.0
    logits[(row + 1) % 3, 0] = 100.0
    g = graph_of(lambda g, x, out: M.argmax(g, x, out, row), x=("fp32", logits.shape, "input"), out=("i32", (1,), "output"))
    assert run(g, {"x": logits})["out"].item() == 4097
    logits[row, 151935] = 60.0
    assert run(g, {"x": logits})["out"].item() == 151935


def test_dump_mlir(tmp_path):
    out = tmp_path / "qwen3.triton.mlir"
    subprocess.run([sys.executable, "scripts/dump_mlir.py", "models/qwen3_0_6b.py", "-o", str(out)], cwd=ROOT, check=True)
    assert out.read_text().count("tt.func public @") == 12


# ---------------------------------------------------------------------------------------------------
# Whole model vs HF transformers
# ---------------------------------------------------------------------------------------------------

PROMPT = "Explain what a KV cache is in one paragraph."
NEW_TOKENS = 32
MAX_SEQ = 128


@pytest.fixture(scope="module")
def prompt_ids(snap):
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    ids = ids if torch.is_tensor(ids) else ids["input_ids"]
    return ids[0].cuda()


@pytest.fixture(scope="module")
def hf_reference(snap, prompt_ids):
    """HF logits for the prompt in both dtypes and its fp32 greedy continuation, computed first and
    freed so the HF model and ours are never on the GPU together."""
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = {}
    for dtype in DTYPES:
        hf = AutoModelForCausalLM.from_pretrained(snap, dtype=TORCH_DTYPES[dtype]).cuda().eval()
        with torch.no_grad():
            out = hf(prompt_ids[None], use_cache=True)
            ref[f"logits_{dtype}"] = out.logits[0].float()
            if dtype == "fp32":
                tokens, cache = [int(out.logits[0, -1].argmax())], out.past_key_values
                for _ in range(NEW_TOKENS - 1):
                    step = hf(torch.tensor([[tokens[-1]]], device="cuda"), past_key_values=cache, use_cache=True)
                    cache = step.past_key_values
                    tokens.append(int(step.logits[0, -1].argmax()))
                ref["tokens_fp32"] = tokens
        del hf, out
        torch.cuda.empty_cache()
    return ref


@pytest.fixture(scope="module")
def checkpoint(snap, hf_reference):
    weights = load_file(f"{snap}/model.safetensors", device="cuda")
    assert torch.equal(weights.pop("lm_head.weight"), weights["model.embed_tokens.weight"])
    return weights


def weights_as(checkpoint, dtype):
    return {name: w.to(TORCH_DTYPES[dtype]) for name, w in checkpoint.items()}


def test_prefill_fp32_logits_match_hf(checkpoint, prompt_ids, hf_reference):
    T = len(prompt_ids)
    out = run(M.prefill(T, dtype="fp32", max_seq=MAX_SEQ, all_logits=True), {"input_ids": prompt_ids.int(), **weights_as(checkpoint, "fp32")})
    want = hf_reference["logits_fp32"]
    torch.testing.assert_close(out["logits"], want, atol=1e-3, rtol=1e-4)
    assert torch.equal(out["logits"].argmax(-1), want.argmax(-1))
    assert out["next_token"].item() == want[-1].argmax().item()


def test_greedy_fp32_matches_hf(checkpoint, prompt_ids, hf_reference):
    T = len(prompt_ids)
    tensors = {"input_ids": prompt_ids.int(), **weights_as(checkpoint, "fp32")}
    run(M.prefill(T, dtype="fp32", max_seq=MAX_SEQ), tensors)
    tokens = [tensors["next_token"].item()]
    for pos in range(T, T + NEW_TOKENS - 1):
        run(M.decode(pos, dtype="fp32", max_seq=MAX_SEQ), tensors)
        tokens.append(tensors["next_token"].item())
    assert tokens == hf_reference["tokens_fp32"]


def test_prefill_bf16_as_accurate_as_hf_bf16(checkpoint, prompt_ids, hf_reference):
    """bf16 rounds differently from HF's bf16 kernels, so the bar is HF bf16's own error vs fp32."""
    T = len(prompt_ids)
    got = run(M.prefill(T, dtype="bf16", max_seq=MAX_SEQ, all_logits=True), {"input_ids": prompt_ids.int(), **checkpoint})["logits"].float()
    fp32, hf_bf16 = hf_reference["logits_fp32"], hf_reference["logits_bf16"]
    assert (got - fp32).abs().mean() <= 1.5 * (hf_bf16 - fp32).abs().mean()
    assert (got.argmax(-1) == hf_bf16.argmax(-1)).float().mean() >= 0.85
    assert (got.argmax(-1) == fp32.argmax(-1)).float().mean() >= (hf_bf16.argmax(-1) == fp32.argmax(-1)).float().mean() - 0.1
