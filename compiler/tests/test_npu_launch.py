"""kernel[grid](...) on host tensors runs on the NPU (npu3) through the furiosa driver.

    PYTHONPATH=compiler compiler/.venv/bin/python compiler/tests/test_npu_launch.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_add(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, alpha * tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


@triton.jit
def relu(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask), 0.0), mask=mask)


def main():
    assert triton.runtime.driver.active.get_current_target().backend == "furiosa"
    n = 3000
    x, y = torch.randn(n), torch.randn(n)
    out = torch.zeros(n)
    scale_add[(triton.cdiv(n, 1024),)](x, y, out, n, 2.0, BLOCK=1024)
    assert torch.equal(out, 2.0 * x + y), (out - (2.0 * x + y)).abs().max()
    relu[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)  # another crate in the same target dir
    assert torch.equal(out, torch.relu(x))
    out.zero_()
    scale_add[(triton.cdiv(n, 1024),)](x, y, out, n, 2.0, BLOCK=1024)  # cached: still its own binary
    assert torch.equal(out, 2.0 * x + y)
    print("ok")


if __name__ == "__main__":
    main()
