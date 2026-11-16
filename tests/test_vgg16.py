import copy

import pytest
import torch
import torchvision

from harness import TORCH_DTYPES, load_model, run

vgg16 = load_model("models/vgg16.py")


@pytest.fixture(scope="module")
def model():
    return torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().eval()


def run_vgg16(model, x, dtype):
    tensors = {name: t.to(TORCH_DTYPES[dtype]).contiguous() for name, t in model.state_dict().items()}
    tensors["image"] = x.to(TORCH_DTYPES[dtype]).contiguous()
    graph = vgg16.forward(batch=x.shape[0], dtype=dtype, image_size=x.shape[-1])
    return run(graph, tensors)["logits"].float()


@pytest.mark.parametrize("batch,size", [(1, 224), (3, 224), (2, 256)])
def test_vgg16_fp32(model, batch, size):
    """224 makes the adaptive pool an identity; 256 gives it overlapping 2x2 windows (8x8 -> 7x7)."""
    x = torch.randn(batch, 3, size, size, device="cuda")
    with torch.no_grad():
        want = model(x)
        exact = copy.deepcopy(model).double()(x.double())
    got = run_vgg16(model, x, "fp32")
    torch.testing.assert_close(got.double(), exact, atol=1e-4, rtol=0)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0)
    assert torch.equal(got.argmax(1), want.argmax(1))


def test_vgg16_fp16(model):
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        want = model(x)
        torch_fp16 = copy.deepcopy(model).half()(x.half()).float()
    got = run_vgg16(model, x, "fp16")
    # Triton in fp16 may differ from fp32 by at most about as much as torch's own fp16 run does.
    assert (got - want).abs().max() <= 2 * (torch_fp16 - want).abs().max() + 1e-3
    assert torch.equal(got.argmax(1), want.argmax(1))
