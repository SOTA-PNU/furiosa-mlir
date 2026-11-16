import pytest
import torch


@pytest.fixture(autouse=True)
def _exact_reference():
    """Reference ops run in exact fp32 so they compare against the kernels' "ieee" fp32 dots."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    yield
