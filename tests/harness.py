"""Runs a model file's launch graph on the GPU. Test-only: this is the one place torch appears, as the
memory allocator for Triton's launcher and as the reference the tests compare against."""

import importlib.util
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

ROOT = Path(__file__).resolve().parent.parent
TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "i32": torch.int32, "i64": torch.int64}


def load_model(path: str | Path):
    """Imports a model file (e.g. "models/resnet101.py") as a module."""
    path = ROOT / path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def run(graph, tensors: dict, after_launch=None) -> dict:
    """Runs every launch of ``graph`` in order. ``tensors`` maps buffer names to CUDA tensors; inputs
    and weights must be there, and it gains the activations, outputs and state it allocates.
    ``after_launch(index, launch, tensors)`` is called after each launch."""
    for buf in graph.buffers.values():
        tensor = tensors.get(buf.name)
        dtype = TORCH_DTYPES[buf.dtype]
        if tensor is None:
            if buf.kind in ("input", "weight"):
                raise KeyError(f"missing {buf.kind} {buf.name}")
            # Zeros, as L1 starts them: an element no kernel writes must not be leftover GPU memory.
            tensors[buf.name] = torch.zeros(buf.shape, dtype=dtype, device="cuda")
        elif tuple(tensor.shape) != buf.shape or tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(
                f"{buf.name}: graph wants contiguous cuda {buf.dtype}{list(buf.shape)}, "
                f"got {tensor.dtype}{list(tensor.shape)} on {tensor.device}, contiguous={tensor.is_contiguous()}"
            )
    for index, launch in enumerate(graph.launches):
        args = {name: tensors[value.name] if hasattr(value, "kind") else value for name, value in launch.args.items()}
        # As in Triton, a kernel parameter shadows a launch option of the same name (e.g. num_stages).
        options = {k: v for k, v in launch.options.items() if k not in launch.args}
        launch.kernel[launch.grid](**args, **options)
        if after_launch:
            after_launch(index, launch, tensors)
    return tensors
