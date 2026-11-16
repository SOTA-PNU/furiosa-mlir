import copy

import pytest
import torch
import torchvision

from harness import TORCH_DTYPES, load_model, run

alexnet = load_model("models/alexnet.py")


@pytest.fixture(scope="module")
def model():
    return torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).cuda().eval()


def run_alexnet(model, x, dtype):
    tensors = {name: t.to(TORCH_DTYPES[dtype]).contiguous() for name, t in model.state_dict().items()}
    tensors["image"] = x.to(TORCH_DTYPES[dtype]).contiguous()
    graph = alexnet.forward(batch=x.shape[0], dtype=dtype, image_size=x.shape[-1])
    return run(graph, tensors)["logits"].float()


@pytest.mark.parametrize("batch,size", [(1, 224), (3, 224), (2, 256)])
def test_alexnet_fp32(model, batch, size):
    """224 makes the adaptive pool an identity; 256 gives it overlapping 2x2 windows (7x7 -> 6x6)."""
    x = torch.randn(batch, 3, size, size, device="cuda")
    with torch.no_grad():
        want = model(x)
        exact = copy.deepcopy(model).double()(x.double())
    got = run_alexnet(model, x, "fp32")
    # cuDNN's fp32 AlexNet is itself ~5e-4 off; compare against fp64 instead.
    torch.testing.assert_close(got.double(), exact, atol=1e-4, rtol=0)
    assert torch.equal(got.argmax(1), want.argmax(1))


def test_alexnet_fp16(model):
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        want = model(x)
        torch_fp16 = copy.deepcopy(model).half()(x.half()).float()
    got = run_alexnet(model, x, "fp16")
    # Triton in fp16 may differ from fp32 by at most about as much as torch's own fp16 run does.
    assert (got - want).abs().max() <= 2 * (torch_fp16 - want).abs().max() + 1e-3
    assert torch.equal(got.argmax(1), want.argmax(1))
