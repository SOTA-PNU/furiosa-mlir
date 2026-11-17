"""Playground: write a Triton kernel, launch it, and it compiles and runs on the NPU (npu3).

    compiler/.venv/bin/python playground/vector_add.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)



n = 3000
x, y = torch.randn(n), torch.randn(n)  # host (CPU) tensors: the launcher copies them to the NPU and back
out = torch.zeros(n)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
print("max abs error:", (out - (x + y)).abs().max().item())

# Every stage of this launch: ttir, linalg, structured, steps, tiled .mlir, and the furiosa-opt crate.
from furiosa_compiler.locks import CACHE  # importable once the furiosa driver has loaded

work = max(CACHE.glob(f"{add_kernel.__name__}-*"), key=lambda p: p.stat().st_mtime)
for path in sorted((work / "out" / "kernels").glob("*/*.mlir")) + [work / "crate"]:
    print(path)
