import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from harness import TORCH_DTYPES, load_model, run

lenet5 = load_model("models/lenet5.py")


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def run_lenet5(model, x, dtype):
    tensors = {name: t.detach().to(TORCH_DTYPES[dtype]).contiguous() for name, t in model.state_dict().items()}
    tensors["image"] = x.to(TORCH_DTYPES[dtype]).contiguous()
    return run(lenet5.forward(batch=x.shape[0], dtype=dtype), tensors)["logits"].float()


@pytest.mark.parametrize("batch", [1, 5])
def test_lenet5_fp32(batch):
    model = LeNet5().cuda().eval()
    x = torch.randn(batch, 1, 28, 28, device="cuda")
    with torch.no_grad():
        want = model(x)
        exact = copy.deepcopy(model).double()(x.double())
    got = run_lenet5(model, x, "fp32")
    torch.testing.assert_close(got.double(), exact, atol=1e-5, rtol=0)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
    assert torch.equal(got.argmax(1), want.argmax(1))


def test_lenet5_fp16():
    model = LeNet5().cuda().eval()
    x = torch.randn(4, 1, 28, 28, device="cuda")
    with torch.no_grad():
        want = model(x)
        torch_fp16 = copy.deepcopy(model).half()(x.half()).float()
    got = run_lenet5(model, x, "fp16")
    # Triton in fp16 may differ from fp32 by at most about as much as torch's own fp16 run does.
    assert (got - want).abs().max() <= 2 * (torch_fp16 - want).abs().max() + 1e-3
    assert torch.equal(got.argmax(1), want.argmax(1))
