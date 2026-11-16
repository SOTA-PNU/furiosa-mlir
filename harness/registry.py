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
    # fp32 models: the NPU contracts in bf16, so L3/L4 allow twice the logit drift of the same Triton
    # graph run in bf16 on the GPU (measured: lenet5 0.16%, alexnet 0.37%, vgg16 0.66%, resnet101 2.4%,
    # mobilenetv3 6.5% of max|logit|), and the top-1 class must not change.
    "lenet5": Case("models/lenet5.py", npu=Tol(rtol=3.2e-3, atol=0), top1="logits"),
    "alexnet": Case("models/alexnet.py", weights="torchvision", npu=Tol(rtol=7.4e-3, atol=0), top1="logits"),
    "vgg16": Case("models/vgg16.py", weights="torchvision", npu=Tol(rtol=1.3e-2, atol=0), top1="logits"),
    "resnet101": Case("models/resnet101.py", weights="torchvision", npu=Tol(rtol=4.8e-2, atol=0), top1="logits"),
    "mobilenetv3_large": Case("models/mobilenetv3_large.py", weights="torchvision", npu=Tol(rtol=1.3e-1, atol=0),
                              top1="logits"),
    # 28 bf16 layers: each kernel is held to one ulp per launch, but one-ulp differences compound over
    # the depth of a whole graph (logits drift ~2% of their scale; the greedy token still agrees).
    "qwen3_0_6b": Case("models/qwen3_0_6b.py", weights="huggingface", int_ranges={"input_ids": (0, 151936)},
                       graph=Tol(rtol=3e-2, atol=0), npu=Tol(rtol=3e-2, atol=0)),
}
