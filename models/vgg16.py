"""VGG-16 (torchvision layout) in pure Triton.

Weights are the raw torchvision ``state_dict`` tensors by name: conv weights stay ``[Cout, Cin, KH, KW]``
and are read through strides, each conv's bias and ReLU run in its epilogue. Activations are NHWC; the
adaptive average pool writes the (C, H, W)-flattened rows the classifier reads.
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
# Kernels
# ---------------------------------------------------------------------------------------------------


@triton.jit
def conv2d_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, H, W, C, OH, OW, CO,
    sxn, sxh, sxw, sxc,
    STRIDE: tl.constexpr, PAD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, CO] = relu(conv(x, w) + b) as an implicit GEMM: rows are output pixels (NHWC order), columns
    output channels. x is read through its strides; w is the raw ``[CO, C, KH, KW]`` tensor."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < CO
    n = offs_m // (OH * OW)
    oh = (offs_m // OW) % OH
    ow = offs_m % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in range(KH):
        ih = oh * STRIDE - PAD + kh
        for kw in range(KW):
            iw = ow * STRIDE - PAD + kw
            ok = m_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_rows = x_ptr + n * sxn + ih * sxh + iw * sxw
            w_cols = w_ptr + offs_n * (C * KH * KW) + kh * KW + kw
            for c0 in range(0, C, BLOCK_K):
                c = c0 + offs_k
                c_mask = c < C
                a = tl.load(x_rows[:, None] + c[None, :] * sxc, mask=ok[:, None] & c_mask[None, :], other=0.0)
                b = tl.load(w_cols[None, :] + c[:, None] * (KH * KW), mask=c_mask[:, None] & n_mask[None, :], other=0.0)
                acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = tl.maximum(acc + tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :], 0.0)
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * CO + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def maxpool2d_kernel(
    x_ptr, y_ptr,
    P, H, W, C, OH, OW,
    K: tl.constexpr, STRIDE: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Max pooling over unpadded K x K windows, NHWC to NHWC."""
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_p < P)[:, None] & (offs_c < C)[None, :]
    n = offs_p // (OH * OW)
    oh = (offs_p // OW) % OH
    ow = offs_p % OW
    acc = tl.full((BLOCK_P, BLOCK_C), float("-inf"), tl.float32)
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            rows = ((n * H + oh * STRIDE + kh) * W + ow * STRIDE + kw) * C
            acc = tl.maximum(acc, tl.load(x_ptr + rows[:, None] + offs_c[None, :], mask=mask, other=float("-inf")).to(tl.float32))
    tl.store(y_ptr + offs_p[:, None] * C + offs_c[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def adaptive_avgpool2d_kernel(x_ptr, y_ptr, H, W, C, OH, OW, BLOCK_C: tl.constexpr):
    """torch.nn.AdaptiveAvgPool2d on an NHWC x: output pixel (oh, ow) averages rows
    [oh*H // OH, ceil((oh+1)*H / OH)) and the matching columns. One program per output pixel and channel
    block; it writes the (C, H, W)-flattened ``[N, C*OH*OW]`` rows the classifier reads."""
    pid = tl.program_id(0)
    n = pid // (OH * OW)
    oh = (pid // OW) % OH
    ow = pid % OW
    h0 = (oh * H) // OH
    h1 = ((oh + 1) * H + OH - 1) // OH
    w0 = (ow * W) // OW
    w1 = ((ow + 1) * W + OW - 1) // OW
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for h in range(h0, h1):
        for w in range(w0, w1):
            acc += tl.load(x_ptr + ((n * H + h) * W + w) * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    acc = acc / ((h1 - h0) * (w1 - w0))
    tl.store(y_ptr + n * (C * OH * OW) + offs_c * (OH * OW) + oh * OW + ow, acc.to(y_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    RELU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ w[N, K]^T + b, then ReLU if ``RELU``; w is the raw ``nn.Linear`` weight."""
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
    if RELU:
        acc = tl.maximum(acc, 0.0)
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------------------------------
# Layers: each declares its buffers and appends its launch to the graph
# ---------------------------------------------------------------------------------------------------


def conv2d_relu(g, x, prefix, cout, k, stride, pad):
    """``relu(conv(x) + bias)`` with the checkpoint's ``<prefix>.weight`` / ``<prefix>.bias``; x is the NCHW
    input or an NHWC activation, the result is NHWC."""
    if x.kind == "input":
        n, c, h, w = x.shape
        sx = (c * h * w, w, 1, h * w)
    else:
        n, h, w, c = x.shape
        sx = (h * w * c, w * c, c, 1)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    y = g.buffer(f"{prefix}.out", x.dtype, (n, oh, ow, cout), "activation")
    m = n * oh * ow
    g.launch(
        conv2d_relu_kernel, (triton.cdiv(m, 64), triton.cdiv(cout, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (cout, c, k, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (cout,), "weight"), y_ptr=y,
        M=m, H=h, W=w, C=c, OH=oh, OW=ow, CO=cout,
        sxn=sx[0], sxh=sx[1], sxw=sx[2], sxc=sx[3],
        STRIDE=stride, PAD=pad, KH=k, KW=k, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 if c >= 32 else 16,
    )
    return y


def maxpool2d(g, x, name, k, stride):
    """Unpadded K x K max pooling, NHWC to NHWC."""
    n, h, w, c = x.shape
    oh = (h - k) // stride + 1
    ow = (w - k) // stride + 1
    y = g.buffer(f"{name}.out", x.dtype, (n, oh, ow, c), "activation")
    p = n * oh * ow
    g.launch(
        maxpool2d_kernel, (triton.cdiv(p, 32), triton.cdiv(c, 64)),
        x_ptr=x, y_ptr=y, P=p, H=h, W=w, C=c, OH=oh, OW=ow, K=k, STRIDE=stride, BLOCK_P=32, BLOCK_C=64,
    )
    return y


def adaptive_avgpool2d_flatten(g, x, name, out_hw):
    """torch.nn.AdaptiveAvgPool2d(out_hw) then torch.flatten(1): NHWC to ``[N, C*OH*OW]``."""
    n, h, w, c = x.shape
    oh, ow = out_hw
    y = g.buffer(f"{name}.out", x.dtype, (n, c * oh * ow), "activation")
    g.launch(
        adaptive_avgpool2d_kernel, (n * oh * ow, triton.cdiv(c, 128)),
        x_ptr=x, y_ptr=y, H=h, W=w, C=c, OH=oh, OW=ow, BLOCK_C=128,
    )
    return y


def linear(g, x, prefix, n_out, relu, name=None, kind="activation"):
    """``x @ weight.T + bias`` (then ReLU) with the checkpoint's ``<prefix>.weight`` / ``<prefix>.bias``."""
    m, k = x.shape
    y = g.buffer(name or f"{prefix}.out", x.dtype, (m, n_out), kind)
    g.launch(
        linear_kernel, (triton.cdiv(m, 16), triton.cdiv(n_out, 64)),
        x_ptr=x, w_ptr=g.buffer(f"{prefix}.weight", x.dtype, (n_out, k), "weight"),
        b_ptr=g.buffer(f"{prefix}.bias", x.dtype, (n_out,), "weight"), y_ptr=y,
        M=m, N=n_out, K=k, RELU=relu, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )
    return y


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

# torchvision's configuration "D": output channels per conv, "M" for a 2x2 max pool.
CONFIG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def forward(batch=1, dtype="fp32", image_size=224):
    """VGG-16 on ``image`` [N, 3, S, S] -> ``logits`` [N, 1000]. Dropout is the identity in eval."""
    g = Graph(f"vgg16.forward(batch={batch}, dtype={dtype}, image_size={image_size})")
    x = g.buffer("image", dtype, (batch, 3, image_size, image_size), "input")
    index = 0  # position in torchvision's ``features`` Sequential, which names the weights
    for v in CONFIG:
        if v == "M":
            x = maxpool2d(g, x, f"features.{index}", k=2, stride=2)
            index += 1
        else:
            x = conv2d_relu(g, x, f"features.{index}", v, k=3, stride=1, pad=1)
            index += 2  # the conv and its ReLU
    x = adaptive_avgpool2d_flatten(g, x, "avgpool", (7, 7))
    x = linear(g, x, "classifier.0", 4096, relu=True)
    x = linear(g, x, "classifier.3", 4096, relu=True)
    linear(g, x, "classifier.6", 1000, relu=False, name="logits", kind="output")
    return g


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return [forward(batch=1, dtype="fp32")]
