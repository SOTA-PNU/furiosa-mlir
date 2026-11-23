"""Qwen3-4B (Qwen/Qwen3-4B): the Qwen3 graphs of models/qwen3_0_6b.py at its config.json."""

import importlib.util
from pathlib import Path

# The Qwen3 kernels and builders, loaded by path as the harness loads a model file.
_spec = importlib.util.spec_from_file_location("qwen3_0_6b", Path(__file__).with_name("qwen3_0_6b.py"))
qwen3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qwen3)

CONFIG = qwen3.Config(hidden=2560, layers=36, q_heads=32, kv_heads=8, head_dim=128, inter=9728, vocab=151936, tied=True)


def graphs():
    """The graphs scripts/dump_mlir.py writes to the .triton.mlir file."""
    return qwen3.graphs(CONFIG)
