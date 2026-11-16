"""The CNN model files' kernels on shapes and fusions the full models don't reach, as small graphs."""

import pytest
import torch
import torch.nn.functional as F
import triton

from harness import TORCH_DTYPES, load_model, run

resnet = load_model("models/resnet101.py")
mobilenet = load_model("models/mobilenetv3_large.py")

TOL = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 5e-2}


def bn_params(prefix: str, c: int, dtype: str) -> dict:
    td = TORCH_DTYPES[dtype]
    return {
        f"{prefix}.weight": (torch.rand(c, device="cuda") + 0.5).to(td),
        f"{prefix}.bias": torch.randn(c, device="cuda").to(td),
        f"{prefix}.running_mean": torch.randn(c, device="cuda").to(td),
        f"{prefix}.running_var": (torch.rand(c, device="cuda") + 0.1).to(td),
    }


def torch_bn(y: torch.Tensor, t: dict, prefix: str, eps: float) -> torch.Tensor:
    p = {k: t[f"{prefix}.{k}"].float() for k in ("weight", "bias", "running_mean", "running_var")}
    return F.batch_norm(y, p["running_mean"], p["running_var"], p["weight"], p["bias"], training=False, eps=eps)


def nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_resnet_conv_nchw_input_residual(dtype):
    """The stem's NCHW read, on odd sizes, with a residual and ReLU."""
    td = TORCH_DTYPES[dtype]
    N, Cin, H, W, Cout, k, s, p = 2, 3, 37, 29, 24, 7, 2, 3
    g = resnet.Graph("conv")
    x = g.buffer("x", dtype, (N, Cin, H, W), "input")
    OH, OW = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    r = g.buffer("r", dtype, (N, OH, OW, Cout), "input")
    resnet.conv_bn(g, x, "c", "b", Cout, k, s, p, resnet.ACT_RELU, "y", residual=r, nchw=True)
    t = {"x": torch.randn(N, Cin, H, W, device="cuda").to(td), "r": torch.randn(N, OH, OW, Cout, device="cuda").to(td),
         "c.weight": (torch.randn(Cout, Cin, k, k, device="cuda") / (Cin * k * k) ** 0.5).to(td), **bn_params("b", Cout, dtype)}
    run(g, t)
    want = F.conv2d(t["x"].float(), t["c.weight"].float(), stride=s, padding=p)
    want = F.relu(nhwc(torch_bn(want, t, "b", resnet.BN_EPS)) + t["r"].float())
    torch.testing.assert_close(t["y"].float(), want, atol=TOL[dtype], rtol=TOL[dtype])


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("act", ["ACT_NONE", "ACT_RELU", "ACT_HARDSWISH"])
def test_mobilenet_conv_scale_residual(dtype, act):
    """SE input scale and residual together, on sizes no block uses."""
    td = TORCH_DTYPES[dtype]
    N, H, W, Cin, Cout = 3, 9, 11, 72, 40
    g = mobilenet.Graph("conv")
    x = g.buffer("x", dtype, (N, H, W, Cin), "input")
    s = g.buffer("s", dtype, (N, Cin), "input")
    r = g.buffer("r", dtype, (N, H, W, Cout), "input")
    mobilenet.conv_bn(g, x, "p", Cout, 1, 1, getattr(mobilenet, act), "y", residual=r, scale=s)
    t = {"x": torch.randn(N, H, W, Cin, device="cuda").to(td), "s": torch.rand(N, Cin, device="cuda").to(td),
         "r": torch.randn(N, H, W, Cout, device="cuda").to(td),
         "p.0.weight": (torch.randn(Cout, Cin, 1, 1, device="cuda") / Cin**0.5).to(td), **bn_params("p.1", Cout, dtype)}
    run(g, t)
    scaled = (t["x"].float() * t["s"].float()[:, None, None, :]).to(td).float()
    want = nhwc(torch_bn(F.conv2d(scaled.permute(0, 3, 1, 2), t["p.0.weight"].float()), t, "p.1", mobilenet.BN_EPS))
    want = {"ACT_NONE": lambda v: v, "ACT_RELU": F.relu, "ACT_HARDSWISH": F.hardswish}[act](want + t["r"].float())
    torch.testing.assert_close(t["y"].float(), want, atol=TOL[dtype], rtol=TOL[dtype])


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("k,stride", [(3, 1), (3, 2), (5, 1), (5, 2)])
def test_mobilenet_depthwise(dtype, k, stride):
    td = TORCH_DTYPES[dtype]
    N, H, W, C = 2, 15, 10, 100
    g = mobilenet.Graph("dw")
    x = g.buffer("x", dtype, (N, H, W, C), "input")
    mobilenet.depthwise_bn(g, x, "d", k, stride, mobilenet.ACT_HARDSWISH, "y")
    t = {"x": torch.randn(N, H, W, C, device="cuda").to(td),
         "d.0.weight": (torch.randn(C, 1, k, k, device="cuda") / k).to(td), **bn_params("d.1", C, dtype)}
    run(g, t)
    want = F.conv2d(t["x"].float().permute(0, 3, 1, 2), t["d.0.weight"].float(), stride=stride, padding=(k - 1) // 2, groups=C)
    want = F.hardswish(nhwc(torch_bn(want, t, "d.1", mobilenet.BN_EPS)))
    torch.testing.assert_close(t["y"].float(), want, atol=TOL[dtype], rtol=TOL[dtype])


@pytest.mark.parametrize("act", ["ACT_NONE", "ACT_RELU", "ACT_HARDSWISH", "ACT_HARDSIGMOID"])
def test_mobilenet_linear(act):
    M, K, N = 5, 200, 130
    g = mobilenet.Graph("linear")
    x = g.buffer("x", "fp32", (M, K), "input")
    mobilenet.linear(g, x, "w", "b", (N, K, 1, 1), getattr(mobilenet, act), "y")
    t = {"x": torch.randn(M, K, device="cuda"), "w": torch.randn(N, K, 1, 1, device="cuda") / K**0.5, "b": torch.randn(N, device="cuda")}
    run(g, t)
    f = {"ACT_NONE": lambda v: v, "ACT_RELU": F.relu, "ACT_HARDSWISH": F.hardswish, "ACT_HARDSIGMOID": F.hardsigmoid}[act]
    torch.testing.assert_close(t["y"], f(t["x"] @ t["w"][:, :, 0, 0].T + t["b"]), atol=1e-4, rtol=1e-4)


def test_pools():
    N, H, W, C = 2, 13, 9, 70
    x = torch.randn(N, H, W, C, device="cuda")
    g = resnet.Graph("pools")
    xb = g.buffer("x", "fp32", (N, H, W, C), "input")
    OH, OW = (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1
    yb = g.buffer("max", "fp32", (N, OH, OW, C), "output")
    g.launch(resnet.maxpool2d_kernel, (triton.cdiv(N * OH * OW, 32), triton.cdiv(C, 64)), x_ptr=xb, y_ptr=yb,
             M=N * OH * OW, H=H, W=W, C=C, OH=OH, OW=OW, stride=2, pad=1, K=3, BLOCK_P=32, BLOCK_C=64)
    mobilenet.avgpool(g, xb, "mean")
    t = run(g, {"x": x})
    torch.testing.assert_close(t["max"], nhwc(F.max_pool2d(x.permute(0, 3, 1, 2), 3, 2, 1)), atol=0, rtol=0)
    torch.testing.assert_close(t["mean"], x.mean(dim=(1, 2)), atol=1e-5, rtol=1e-5)

