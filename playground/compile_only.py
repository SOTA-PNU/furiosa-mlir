#%%
"""Compile a Triton kernel through the furiosa stages with triton.compile, without running it.

    compiler/.venv/bin/python playground/compile_only.py
"""

import json

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

#%%
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


n, BLOCK = 3000, 1024
# Pointers typed, every scalar a constexpr (keys are parameter positions), as the furiosa driver specializes.
src = ASTSource(
    add_kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "constexpr", "BLOCK": "constexpr"},
    constexprs={(3,): n, (4,): BLOCK},
    attrs={(i,): [["tt.divisibility", 16]] for i in range(3)},
)
# The launch the raising pass needs: the grid, and the buffer behind each pointer argument in order.
buffers = [{"arg": i, "kind": "state", "dtype": "fp32", "shape": [n]} for i in range(3)]
compiled = triton.compile(src, target=GPUTarget("furiosa", "rngd", 1),
                          options={"grid": (triton.cdiv(n, BLOCK), 1, 1), "buffers": json.dumps(buffers)})

for stage in ("ttir", "linalg", "structured", "fopt", "rust"):
    text = compiled.asm[stage]
    print(f"===== {stage} =====\n{text.decode() if isinstance(text, bytes) else text}")  # ttir comes as bytes

# %%
