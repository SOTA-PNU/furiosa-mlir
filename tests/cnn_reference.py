"""Shared checks for the CNN model files against their torchvision originals."""

import copy

import torch

from harness import TORCH_DTYPES, run


def checkpoint(reference: torch.nn.Module) -> dict:
    """The torchvision state_dict tensors a graph declares as weights (num_batches_tracked is unused)."""
    return {k: v for k, v in reference.state_dict().items() if not k.endswith("num_batches_tracked")}


def run_graph(model, reference, x: torch.Tensor, dtype: str) -> torch.Tensor:
    """The graph's fp32 logits for an NCHW batch, with the checkpoint cast to ``dtype``."""
    td = TORCH_DTYPES[dtype]
    g = model.forward(batch=x.shape[0], dtype=dtype)
    weights = {k: v.to(td).contiguous() for k, v in checkpoint(reference).items()}
    return run(g, {"image": x.to(td).contiguous(), **weights})["logits"].float()


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).norm() / want.norm()).item()


def assert_declares_checkpoint(model, reference):
    """The graph's weights are exactly the checkpoint's tensors, by name and shape."""
    g = model.forward(batch=1, dtype="fp32")
    declared = {b.name: b.shape for b in g.buffers.values() if b.kind == "weight"}
    expected = {k: tuple(v.shape) for k, v in checkpoint(reference).items()}
    assert declared == expected


def assert_fp32_matches(model, reference):
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        want = reference(x)
    got = run_graph(model, reference, x, "fp32")
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    assert torch.equal(got.argmax(1), want.argmax(1))


def assert_reduced_precision_tracks(model, reference, dtype: str):
    """In fp16/bf16 the graph's arithmetic error is no worse than torchvision's own in that dtype.

    Both are measured against exact fp32 math on the same rounded weights and input, which separates
    rounding in the arithmetic from the weight quantization they share.
    """
    x = torch.randn(2, 3, 224, 224, device="cuda")
    td = TORCH_DTYPES[dtype]
    rounded = copy.deepcopy(reference)
    rounded.load_state_dict({k: v.to(td).float() if v.is_floating_point() else v for k, v in reference.state_dict().items()})
    with torch.no_grad():
        exact = rounded(x.to(td).float())
        own = rounded.to(td)(x.to(td)).float()
    got = run_graph(model, reference, x, dtype)
    assert torch.isfinite(got).all()
    torch_error = relative_error(own, exact)
    triton_error = relative_error(got, exact)
    assert triton_error <= 1.5 * torch_error, f"{dtype}: triton {triton_error:.3e} vs torchvision {torch_error:.3e}"
