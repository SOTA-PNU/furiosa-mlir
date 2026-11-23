"""Every case the harness knows: its pure-Triton file, how its inputs are made, and its tolerances."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


@dataclass(frozen=True)
class Case:
    path: str  # pure-Triton file defining graphs()
    exact: Tol = Tol(rtol=1e-5, atol=1e-5)  # L1 per launch: the raised IR keeps the case's dtypes
    graph: Tol | None = None  # L1 over a whole graph (None: same as exact); see harness/README.md
    npu: Tol = Tol(rtol=2e-2, atol=1e-2)  # L3/L4: dots run in bf16 with fp32 accumulation on the NPU
    int_ranges: dict = field(default_factory=dict)  # integer input buffer -> [low, high)
    float_ranges: dict = field(default_factory=dict)  # float input buffer -> [low, high), uniform (a function's domain)
    weights: str | None = None  # a loader in harness/weights.py; None means seeded random weights
    top1: str | None = None  # L3/L4: this output's per-row argmax must equal golden's
    seed: int = 0


CASES = {
    "vector_add": Case("harness/cases/vector_add.py", npu=Tol(rtol=0, atol=0)),
    "rmsnorm": Case("harness/cases/rmsnorm.py", npu=Tol(rtol=8e-3, atol=0)),  # one bf16 ulp at max|golden|
    "matmul": Case("harness/cases/matmul.py", npu=Tol(rtol=8e-3, atol=0)),
    "embedding": Case("harness/cases/embedding.py", npu=Tol(rtol=0, atol=0), int_ranges={"ids": (0, 1024)}),
    # Triton's own tutorial kernels, verbatim: code the compiler's authors did not write.
    "tutorial_vector_add": Case("harness/cases/tutorial_vector_add.py"),
    "tutorial_softmax": Case("harness/cases/tutorial_softmax.py"),
    "tutorial_matmul": Case("harness/cases/tutorial_matmul.py", exact=Tol(rtol=2e-3, atol=2e-3)),
    "tutorial_layer_norm": Case("harness/cases/tutorial_layer_norm.py", exact=Tol(rtol=2e-3, atol=2e-3)),
    "tutorial_dropout": Case("harness/cases/tutorial_dropout.py"),
    "tutorial_fused_attention": Case("harness/cases/tutorial_fused_attention.py", exact=Tol(rtol=2e-3, atol=2e-3)),
    "tutorial_persistent_matmul": Case("harness/cases/tutorial_persistent_matmul.py", exact=Tol(rtol=2e-3, atol=2e-3)),
    "tutorial_extern": Case("harness/cases/tutorial_extern.py", float_ranges={"x": (-1.0, 1.0)}),  # asin's domain
    # The fp16 forward feeds the backward: chained over the graph, dk drifts 3.4e-3 of max|dk| from the GPU (each
    # launch alone is within `exact`); the graph tolerance is twice that.
    "tutorial_attention_bwd": Case("harness/cases/tutorial_attention_bwd.py", exact=Tol(rtol=2e-3, atol=2e-3),
                                   graph=Tol(rtol=7e-3, atol=2e-3)),
    # Held-out kernels, written apart from the corpus (harness/README.md). NPU tolerances: data movement
    # and max are exact; contractions allow twice the measured drift of rounding their operands to bf16
    # (bmm 0.34%, depthwise 0.32% of max|out|); a bf16 output allows one bf16 ulp at max|golden|; fp32
    # through Vector Engine exp/log keeps the exact tolerance: those functions are within 1.7 f32 ulps on
    # npu3 (harness/characterize/npu3.jsonl), so only the order of the fp32 sums differs.
    "heldout_transpose": Case("harness/cases/heldout_transpose.py", npu=Tol(rtol=0, atol=0)),
    "heldout_bmm": Case("harness/cases/heldout_bmm.py", npu=Tol(rtol=7e-3, atol=0)),
    "heldout_maxpool": Case("harness/cases/heldout_maxpool.py", npu=Tol(rtol=0, atol=0)),
    "heldout_gelu": Case("harness/cases/heldout_gelu.py", npu=Tol(rtol=8e-3, atol=0)),
    "heldout_logsumexp": Case("harness/cases/heldout_logsumexp.py", npu=Tol(rtol=1e-5, atol=0)),
    "heldout_depthwise": Case("harness/cases/heldout_depthwise.py", npu=Tol(rtol=6.5e-3, atol=0)),
    # Second held-out batch, by the same rules: a bf16 output allows one bf16 ulp (fp16: one fp16 ulp) at
    # max|golden|; fp32 elementwise math and reductions keep the exact tolerance; the direct conv is a
    # contraction (twice the measured drift of rounding x and w to bf16: 0.20% of max|y|); q is exact. AdamW drops
    # the 1e-5 absolute floor at L1 too: its parameters are ~1e-3, so that floor would accept 1% errors.
    "heldout2_layernorm_bwd": Case("harness/cases/heldout2_layernorm_bwd.py", npu=Tol(rtol=8e-3, atol=0),
                                   float_ranges={"rstd": (0.5, 2.0)}),
    "heldout2_rmsnorm_bwd": Case("harness/cases/heldout2_rmsnorm_bwd.py", npu=Tol(rtol=1e-3, atol=0)),
    "heldout2_rope": Case("harness/cases/heldout2_rope.py", npu=Tol(rtol=8e-3, atol=0),
                          float_ranges={"cos": (-1.0, 1.0), "sin": (-1.0, 1.0)}),
    "heldout2_cross_entropy": Case("harness/cases/heldout2_cross_entropy.py", npu=Tol(rtol=1e-5, atol=0),
                                   int_ranges={"target": (0, 1000)}),
    "heldout2_softmax_bwd": Case("harness/cases/heldout2_softmax_bwd.py", npu=Tol(rtol=1e-3, atol=0),
                                 float_ranges={"y": (0.0, 1.0)}),
    "heldout2_group_norm": Case("harness/cases/heldout2_group_norm.py", npu=Tol(rtol=1e-5, atol=0)),
    "heldout2_quant_int8": Case("harness/cases/heldout2_quant_int8.py", npu=Tol(rtol=1e-5, atol=0)),
    "heldout2_adamw": Case("harness/cases/heldout2_adamw.py", exact=Tol(rtol=1e-5, atol=0), npu=Tol(rtol=1e-5, atol=0)),
    "heldout2_linear_bias_residual": Case("harness/cases/heldout2_linear_bias_residual.py", npu=Tol(rtol=8e-3, atol=0)),
    "heldout2_conv2d": Case("harness/cases/heldout2_conv2d.py", npu=Tol(rtol=4e-3, atol=0)),
    # Third held-out batch, by the same rules (ten standard ops written from their math after 1.4.0, without
    # reading the compiler). The attention's contractions allow twice the measured drift of rounding q, k, v and p
    # to bf16 (0.32% of max|o|); e4m3 values are exact in bf16, so the fp8 matmul allows one bf16 ulp; integers are
    # exact.
    "heldout3_sliding_window_attention": Case("harness/cases/heldout3_sliding_window_attention.py",
                                              exact=Tol(rtol=2e-3, atol=2e-3), npu=Tol(rtol=6.4e-3, atol=0)),
    "heldout3_scatter_add": Case("harness/cases/heldout3_scatter_add.py", npu=Tol(rtol=1e-5, atol=0),
                                 int_ranges={"ids": (0, 40)}),
    "heldout3_batch_norm": Case("harness/cases/heldout3_batch_norm.py", npu=Tol(rtol=1e-5, atol=0)),
    "heldout3_avgpool": Case("harness/cases/heldout3_avgpool.py", npu=Tol(rtol=1e-3, atol=0)),
    "heldout3_upsample_bilinear": Case("harness/cases/heldout3_upsample_bilinear.py", npu=Tol(rtol=8e-3, atol=0)),
    "heldout3_moe_topk_gating": Case("harness/cases/heldout3_moe_topk_gating.py", npu=Tol(rtol=1e-5, atol=0)),
    "heldout3_segment_sum": Case("harness/cases/heldout3_segment_sum.py", npu=Tol(rtol=1e-5, atol=0),
                                 int_ranges={"seg_len": (0, 16)}),
    "heldout3_fp8_matmul": Case("harness/cases/heldout3_fp8_matmul.py", npu=Tol(rtol=8e-3, atol=0),
                                float_ranges={"a": (-8.0, 8.0), "b": (-8.0, 8.0)}),
    "heldout3_add_layernorm": Case("harness/cases/heldout3_add_layernorm.py", npu=Tol(rtol=1e-3, atol=0)),
    "heldout3_cumsum": Case("harness/cases/heldout3_cumsum.py", npu=Tol(rtol=0, atol=0),
                            int_ranges={"x": (-100, 100)}),
    # fp32 models: the NPU contracts in bf16, so L3/L4 allow twice the logit drift of the same Triton
    # graph run in bf16 on the GPU (measured: lenet5 0.16%, alexnet 0.37%, vgg16 0.66%, resnet101 2.4%,
    # mobilenetv3 6.5%, vit_b16 0.29% of max|logit|), and the top-1 class must not change.
    "lenet5": Case("models/lenet5.py", npu=Tol(rtol=3.2e-3, atol=0), top1="logits"),
    "alexnet": Case("models/alexnet.py", weights="torchvision", npu=Tol(rtol=7.4e-3, atol=0), top1="logits"),
    "vgg16": Case("models/vgg16.py", weights="torchvision", npu=Tol(rtol=1.3e-2, atol=0), top1="logits"),
    "resnet101": Case("models/resnet101.py", weights="torchvision", npu=Tol(rtol=4.8e-2, atol=0), top1="logits"),
    "mobilenetv3_large": Case("models/mobilenetv3_large.py", weights="torchvision", npu=Tol(rtol=1.3e-1, atol=0),
                              top1="logits"),
    "vit_b16": Case("models/vit_b16.py", weights="torchvision", npu=Tol(rtol=5.8e-3, atol=0), top1="logits"),
    # Held-out vision models (written from torchvision's definitions without reading the compiler), by the same
    # rule: twice the bf16 logit drift (efficientnet_b0 3.7%, convnext_tiny 0.74%, swin_t 1.13%,
    # densenet121 1.94% of max|logit|).
    "efficientnet_b0": Case("models/efficientnet_b0.py", weights="torchvision", npu=Tol(rtol=7.4e-2, atol=0),
                            top1="logits"),
    "convnext_tiny": Case("models/convnext_tiny.py", weights="torchvision", npu=Tol(rtol=1.5e-2, atol=0),
                          top1="logits"),
    "swin_t": Case("models/swin_t.py", weights="torchvision", npu=Tol(rtol=2.3e-2, atol=0), top1="logits"),
    "densenet121": Case("models/densenet121.py", weights="torchvision", npu=Tol(rtol=3.9e-2, atol=0), top1="logits"),
    # 28 bf16 layers: each kernel is held to one ulp per launch, but one-ulp differences compound over
    # the depth of a whole graph. NPU: the models' rule, twice the drift between two correct
    # implementations of the graph (L1's CPU JIT vs the GPU golden: 2.7% at kv_cache.27.v); the greedy
    # next_token is an integer output, so it must match exactly.
    "qwen3_0_6b": Case("models/qwen3_0_6b.py", weights="huggingface", int_ranges={"input_ids": (0, 151936)},
                       graph=Tol(rtol=3e-2, atol=0), npu=Tol(rtol=5.4e-2, atol=0)),
    # Larger Qwen3 sizes (models/qwen3_0_6b.py at their configs), by the same rule; the two implementations are
    # the GPU graph and transformers' Qwen3 on the same weights and prompt, at the worst output (that measure
    # gives 3.05% for 0.6B): graph = the drift, npu = twice it. A seed other than 0 is the first whose golden
    # greedy tokens both lead the runner-up by more than the two implementations' logits differ.
    "qwen3_1_7b": Case("models/qwen3_1_7b.py", weights="huggingface", int_ranges={"input_ids": (0, 151936)},
                       graph=Tol(rtol=2.5e-2, atol=0), npu=Tol(rtol=5e-2, atol=0), seed=2),  # 2.47% kv_cache.14.v
    "qwen3_4b": Case("models/qwen3_4b.py", weights="huggingface", int_ranges={"input_ids": (0, 151936)},
                     graph=Tol(rtol=1.8e-2, atol=0), npu=Tol(rtol=3.6e-2, atol=0), seed=2),  # 1.79% kv_cache.13.v
    "qwen3_8b": Case("models/qwen3_8b.py", weights="huggingface", int_ranges={"input_ids": (0, 151936)},
                     graph=Tol(rtol=2.3e-2, atol=0), npu=Tol(rtol=4.5e-2, atol=0)),  # 2.20% kv_cache.35.v
    # Held-out Hugging Face models (fp32 checkpoints), by the fp32 models' rule: twice the drift of the same
    # Triton graph run in bf16 on the GPU, at the worst output (measured: bert_base pooler_output 3.2%,
    # last_hidden_state 1.0% of max|out|); integer outputs are exact.
    "bert_base": Case("models/bert_base.py", weights="huggingface", npu=Tol(rtol=6.4e-2, atol=0),
                      int_ranges={"input_ids": (0, 30522), "token_type_ids": (0, 2), "lengths": (1, 129)}),
    # gpt2: kv_cache.11.k 2.1% after prefill, logits 0.73% / 0.47% (prefill / decode), next_token unchanged.
    "gpt2": Case("models/gpt2.py", weights="huggingface", npu=Tol(rtol=4.2e-2, atol=0), int_ranges={"input_ids": (0, 50257)}),
    # whisper_tiny: decoder logits 2.7%, encoder_hidden_states 2.2%, next_token unchanged; input_features in
    # [-1, 1), the range of Whisper's normalized log-mel features.
    "whisper_tiny": Case("models/whisper_tiny.py", weights="huggingface", npu=Tol(rtol=5.3e-2, atol=0),
                         int_ranges={"decoder_input_ids": (0, 51865)}, float_ranges={"input_features": (-1.0, 1.0)}),
}
