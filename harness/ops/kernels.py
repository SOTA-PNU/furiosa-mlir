"""One Triton kernel per operator group; OP (a constexpr) picks the operator, so each test exercises one
`tl` operator on a [B, R, C] tensor flattened to rows = B*R, cols = C. Plain pointer arithmetic keeps
the operator under test the only feature in play; block pointers have their own memory test."""

import triton
import triton.language as tl

UNARY = ["abs", "ceil", "cos", "erf", "exp", "exp2", "floor", "log", "log2", "rsqrt", "sigmoid", "sin", "sqrt",
         "sqrt_rn", "neg"]
BINARY = ["add", "sub", "mul", "fdiv", "div_rn", "maximum", "minimum", "where"]
REDUCE = ["sum", "max", "min", "argmax", "argmin", "xor_sum"]
SCAN = ["cumsum", "cumprod", "flip", "sort", "softmax"]
MISC = ["cast", "clamp", "fma", "full", "zeros_like", "arange"]
SHAPE = ["trans", "reshape", "expand_dims", "broadcast_to", "permute"]


@triton.jit
def unary_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                 OP: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    m = (r < R) & (c < C)
    x = tl.load(x_ptr + r * C + c, mask=m, other=1)
    if OP == 0:
        y = tl.abs(x)
    elif OP == 1:
        y = tl.ceil(x)
    elif OP == 2:
        y = tl.cos(x)
    elif OP == 3:
        y = tl.erf(x)
    elif OP == 4:
        y = tl.exp(x)
    elif OP == 5:
        y = tl.exp2(x)
    elif OP == 6:
        y = tl.floor(x)
    elif OP == 7:
        y = tl.log(x)
    elif OP == 8:
        y = tl.log2(x)
    elif OP == 9:
        y = tl.rsqrt(x)
    elif OP == 10:
        y = tl.sigmoid(x)
    elif OP == 11:
        y = tl.sin(x)
    elif OP == 12:
        y = tl.sqrt(x)
    elif OP == 13:
        y = tl.sqrt_rn(x)
    else:
        y = -x
    tl.store(out_ptr + r * C + c, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def binary_kernel(x_ptr, y_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                  OP: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    m = (r < R) & (c < C)
    x = tl.load(x_ptr + r * C + c, mask=m, other=1)
    y = tl.load(y_ptr + r * C + c, mask=m, other=1)
    if OP == 0:
        z = x + y
    elif OP == 1:
        z = x - y
    elif OP == 2:
        z = x * y
    elif OP == 3:
        z = tl.fdiv(x, y)
    elif OP == 4:
        z = tl.div_rn(x, y)
    elif OP == 5:
        z = tl.maximum(x, y)
    elif OP == 6:
        z = tl.minimum(x, y)
    else:
        z = tl.where(x > y, x, y)
    tl.store(out_ptr + r * C + c, z.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def reduce_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                  OP: tl.constexpr):
    """Reduces each row (along C); masked lanes hold the reduction's identity."""
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    m = (r < R) & (c < C)
    if OP == 0:
        x = tl.load(x_ptr + r * C + c, mask=m, other=0)
        y = tl.sum(x, axis=1)
    elif OP == 1:
        x = tl.load(x_ptr + r * C + c, mask=m, other=-float("inf"))
        y = tl.max(x, axis=1)
    elif OP == 2:
        x = tl.load(x_ptr + r * C + c, mask=m, other=float("inf"))
        y = tl.min(x, axis=1)
    elif OP == 3:
        x = tl.load(x_ptr + r * C + c, mask=m, other=-float("inf"))
        y = tl.argmax(x, axis=1)
    elif OP == 4:
        x = tl.load(x_ptr + r * C + c, mask=m, other=float("inf"))
        y = tl.argmin(x, axis=1)
    else:
        x = tl.load(x_ptr + r * C + c, mask=m, other=0)
        y = tl.xor_sum(x, axis=1)
    rows = tl.arange(0, BR)
    tl.store(out_ptr + rows, y.to(out_ptr.dtype.element_ty), mask=rows < R)


@triton.jit
def scan_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, OP: tl.constexpr):
    """Row-wise scans and orderings; C must be a power of two (tl.sort/flip have no masked form)."""
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, C)[None, :]
    m = r < R
    x = tl.load(x_ptr + r * C + c, mask=m, other=0)
    if OP == 0:
        y = tl.cumsum(x, axis=1)
    elif OP == 1:
        y = tl.cumprod(x, axis=1)
    elif OP == 2:
        y = tl.flip(x, dim=1)
    elif OP == 3:
        y = tl.sort(x, dim=1)
    else:
        y = tl.softmax(x)
    tl.store(out_ptr + r * C + c, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def misc_kernel(x_ptr, y_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                OP: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    m = (r < R) & (c < C)
    x = tl.load(x_ptr + r * C + c, mask=m, other=0)
    if OP == 0:
        z = x.to(tl.float32)
    elif OP == 1:
        z = tl.clamp(x, -0.5, 0.5)
    elif OP == 2:
        y = tl.load(y_ptr + r * C + c, mask=m, other=0)
        z = tl.fma(x, y, x)
    elif OP == 3:
        z = tl.full((BR, BC), 3, x.dtype) + x
    elif OP == 4:
        z = tl.zeros_like(x) + x
    else:
        z = (r * C + c).to(x.dtype) + x
    tl.store(out_ptr + r * C + c, z.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def shape_kernel(x_ptr, out_ptr, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr, OP: tl.constexpr):
    """Shape and layout operators on a [B, R, C] block (powers of two)."""
    b = tl.arange(0, B)[:, None, None]
    r = tl.arange(0, R)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    x = tl.load(x_ptr + b * R * C + r * C + c)
    if OP == 0:  # trans of the last two dims: out [B, C, R]
        y = tl.trans(x, (0, 2, 1))
        tl.store(out_ptr + b * R * C + tl.arange(0, C)[None, :, None] * R + tl.arange(0, R)[None, None, :], y)
    elif OP == 1:  # reshape to [B, R*C]
        y = tl.reshape(x, (B, R * C))
        tl.store(out_ptr + tl.arange(0, B)[:, None] * R * C + tl.arange(0, R * C)[None, :], y)
    elif OP == 2:  # expand_dims then sum over the new axis: identity
        y = tl.sum(tl.expand_dims(x, 1), axis=1)
        tl.store(out_ptr + b * R * C + r * C + c, y)
    elif OP == 3:  # broadcast_to: row 0 of each batch broadcast over R
        row = tl.load(x_ptr + b * R * C + c)
        y = tl.broadcast_to(row, (B, R, C))
        tl.store(out_ptr + b * R * C + r * C + c, y)
    else:  # permute (2, 0, 1): out [C, B, R]
        y = tl.permute(x, (2, 0, 1))
        cc = tl.arange(0, C)[:, None, None]
        bb = tl.arange(0, B)[None, :, None]
        rr = tl.arange(0, R)[None, None, :]
        tl.store(out_ptr + cc * B * R + bb * R + rr, y)


@triton.jit
def dot_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    k = tl.arange(0, K)
    a = tl.load(a_ptr + m * K + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n)
    acc = tl.dot(a, b)
    tl.store(out_ptr + m * N + n, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def descriptor_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """Tensor descriptors (Triton 3.9 removed block pointers in their favor): copies x a row tile at a time."""
    src = tl.make_tensor_descriptor(x_ptr, [R, C], [C, 1], [BR, BC])
    dst = tl.make_tensor_descriptor(out_ptr, [R, C], [C, 1], [BR, BC])
    for r0 in tl.static_range(0, R, BR):
        dst.store([r0, 0], src.load([r0, 0]))


@triton.jit
def program_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    """One program per row: program_id and num_programs."""
    row = tl.program_id(0)
    c = tl.arange(0, C)
    x = tl.load(x_ptr + row * C + c)
    tl.store(out_ptr + row * C + c, x + tl.num_programs(0).to(x.dtype))


@triton.jit
def atomic_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, C)
    tl.atomic_add(out_ptr + c, tl.load(x_ptr + row * C + c))
