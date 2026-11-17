"""The operator test matrix: every (group, op, dtype, shape) case with its launch, inputs, torch
reference and tolerance. Criteria follow SOTA-PNU/lowlevel-api-test so results compare across backends:
floats `allclose(rtol=1e-2, atol=1e-2, equal_nan)`, integers exact."""

from dataclasses import dataclass

import numpy as np
import torch
import triton

from harness.ops import kernels as K

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "int32": torch.int32,
          "int8": torch.int8, "float8_e4m3fn": torch.float8_e4m3fn}
FLOATS = ["float32", "float16", "bfloat16"]
ALL = FLOATS + ["int32", "int8"]
# [B, R, C]: the default of lowlevel-api-test, a single row, several batches, an odd (masked) shape, a large tile.
SHAPES = [(1, 64, 64), (1, 1, 1024), (4, 32, 128), (2, 7, 33), (1, 256, 256)]
POW2 = [s for s in SHAPES if all(d & (d - 1) == 0 for d in s)]
DOTS = [(64, 64, 64), (16, 64, 32), (128, 256, 64)]
EXTRA_DTYPES = {"reduce_or": ["int32"], "squeeze_ravel": ALL, "rand": ["float32"], "randn": ["float32"],
                "randint": ["int32"], "rand4x": ["float32"], "randn4x": ["float32"], "randint4x": ["int32"],
                "join": ALL, "split": ALL, "interleave": ALL, "cat": ALL, "swizzle2d": ["int32"],
                "umulhi": ["int32"], "gather_data": ["int32"], "histogram": ["int32"], "maximum_nan": ["float32"], "atomic_cas_old": ["int32"],
                "device_print": ["float32"], "device_assert": ["float32"], "while_loop": ["float32"],
                "range_loop": ["float32"]}


def _philox(seed: int, offs: np.ndarray) -> list:
    """triton.language.random.randint4x for int32 offsets: 10 Philox-4x32 rounds, as uint32 arrays."""
    m = np.uint64(0xFFFFFFFF)
    c0 = offs.astype(np.int64).astype(np.uint64) & m
    c1, c2, c3 = np.zeros_like(c0), np.zeros_like(c0), np.zeros_like(c0)
    k0, k1 = np.uint64(seed & 0xFFFFFFFF), np.uint64((seed >> 32) & 0xFFFFFFFF)
    a, b = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    for _ in range(10):
        p0, p2 = b * c2, a * c0
        c0, c2, c1, c3 = ((p0 >> np.uint64(32)) ^ c1 ^ k0) & m, ((p2 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m, p2 & m
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & m, (k1 + np.uint64(0xBB67AE85)) & m
    return [v.astype(np.uint32) for v in (c0, c1, c2, c3)]


def _uniform(i: np.ndarray) -> np.ndarray:
    x = i.view(np.int32)
    return np.where(x < 0, ~x, x).astype(np.float32) * np.float32(4.6566127342e-10)


def _normal(u1: np.ndarray, u2: np.ndarray):
    r = np.sqrt(np.float32(-2.0) * np.log(np.maximum(np.float32(1e-7), u1)))
    th = np.float32(6.283185307179586) * u2
    return r * np.cos(th), r * np.sin(th)


@dataclass(frozen=True)
class Case:
    group: str
    op: str
    dtype: str
    shape: tuple

    @property
    def key(self) -> str:
        return f"{self.group}.{self.op}.{self.dtype}.{'x'.join(map(str, self.shape))}"

    @staticmethod
    def parse(key: str) -> "Case":
        group, op, dtype, shape = key.split(".")
        return Case(group, op, dtype, tuple(int(d) for d in shape.split("x")))


def all_cases() -> list[Case]:
    cases = []
    for op in K.UNARY:
        for dtype in ALL if op in ("abs", "neg") else FLOATS:
            cases += [Case("unary", op, dtype, s) for s in SHAPES]
    for op in K.BINARY:
        for dtype in FLOATS if op in ("fdiv", "div_rn") else ALL:
            cases += [Case("binary", op, dtype, s) for s in SHAPES]
    for op in K.REDUCE:
        for dtype in ["int32", "int8"] if op == "xor_sum" else ALL:
            cases += [Case("reduce", op, dtype, s) for s in SHAPES]
    for op in K.SCAN:
        for dtype in FLOATS if op in ("cumprod", "softmax") else ALL:
            cases += [Case("scan", op, dtype, s) for s in POW2]
    for op in K.MISC:
        for dtype in FLOATS if op in ("clamp", "fma") else ALL if op != "cast" else ["float16", "bfloat16", "int32", "int8"]:
            cases += [Case("misc", op, dtype, s) for s in [(1, 64, 64), (2, 7, 33)]]
    for op in K.SHAPE:
        cases += [Case("shape", op, dtype, s) for dtype in ALL for s in [(1, 64, 64), (2, 32, 64)]]
    for dtype in FLOATS + ["int8", "float8_e4m3fn"]:
        cases += [Case("dot", "dot", dtype, s) for s in DOTS]
    cases += [Case("memory", "descriptor", dtype, s) for dtype in ALL for s in [(1, 64, 64), (1, 50, 64)]]
    cases += [Case("memory", "masked_store", dtype, s) for dtype in ALL for s in [(1, 64, 64), (2, 7, 33)]]
    for op in K.EXTRA:
        cases += [Case("extra", op, dtype, s) for dtype in EXTRA_DTYPES.get(op, ["float32", "int32"])
                  for s in [(1, 64, 64), (4, 32, 128)]]
    cases += [Case("program", "program_id", "float32", (1, 64, 64))]
    for op in K.ATOMIC:
        cases += [Case("atomic", op, dtype, (1, 8, 64)) for dtype in (["int32"] if op in ("atomic_and", "atomic_or",
                                                                                         "atomic_xor") else ["float32", "int32"])]
    return cases


def _input(shape, dtype, low=-2.0, high=2.0, generator=None):
    if dtype in ("int32", "int8"):
        return torch.randint(-50, 50, shape, generator=generator).to(DTYPES[dtype])
    return (torch.rand(shape, generator=generator) * (high - low) + low).to(DTYPES[dtype])


def build(case: Case):
    """(kernel, grid, args by parameter name, output names, reference by output name, tolerance kind)."""
    g = torch.Generator().manual_seed(0)
    b, r, c = case.shape
    rows, dt = b * r, case.dtype
    blk = dict(R=rows, C=c, BR=triton.next_power_of_2(rows), BC=triton.next_power_of_2(c))
    wide = (lambda t: t.double()) if dt in FLOATS else (lambda t: t.long())
    if case.group == "unary":
        positive = case.op in ("log", "log2", "rsqrt", "sqrt", "sqrt_rn")
        x = _input((rows, c), dt, 0.5 if positive else -2.0, 3.0 if positive else 2.0, g)
        ref = {"abs": torch.abs, "ceil": torch.ceil, "cos": torch.cos, "erf": torch.erf, "exp": torch.exp,
               "exp2": torch.exp2, "floor": torch.floor, "log": torch.log, "log2": torch.log2, "rsqrt": torch.rsqrt,
               "sigmoid": torch.sigmoid, "sin": torch.sin, "sqrt": torch.sqrt, "sqrt_rn": torch.sqrt,
               "neg": torch.neg}[case.op](wide(x))
        out = torch.empty_like(x)
        return K.unary_kernel, (1,), dict(x_ptr=x, out_ptr=out, OP=K.UNARY.index(case.op), **blk), {"out_ptr": ref}, "float"
    if case.group == "binary":
        x = _input((rows, c), dt, generator=g)
        y = _input((rows, c), dt, 0.5, 2.0, g) if case.op in ("fdiv", "div_rn") else _input((rows, c), dt, generator=g)
        xs, ys = wide(x), wide(y)
        ref = {"add": xs + ys, "sub": xs - ys, "mul": xs * ys, "fdiv": xs / ys, "div_rn": xs / ys,
               "maximum": torch.maximum(xs, ys), "minimum": torch.minimum(xs, ys),
               "where": torch.where(xs > ys, xs, ys)}[case.op]
        out = torch.empty_like(x)
        return (K.binary_kernel, (1,), dict(x_ptr=x, y_ptr=y, out_ptr=out, OP=K.BINARY.index(case.op), **blk),
                {"out_ptr": ref}, "float")
    if case.group == "reduce":
        x = _input((rows, c), dt, generator=g)
        xs = wide(x)
        if case.op == "xor_sum":
            ref = xs[:, 0].clone()
            for j in range(1, c):
                ref ^= xs[:, j]
        else:
            ref = {"sum": lambda: xs.sum(-1), "max": lambda: xs.amax(-1), "min": lambda: xs.amin(-1),
                   "argmax": lambda: xs.argmax(-1), "argmin": lambda: xs.argmin(-1)}[case.op]()
        out = torch.empty(rows, dtype=torch.int32 if case.op in ("argmax", "argmin") else x.dtype)
        if x.dtype.is_floating_point:
            lo, hi = -float("inf"), float("inf")
        else:
            lo, hi = torch.iinfo(x.dtype).min, torch.iinfo(x.dtype).max
        return (K.reduce_kernel, (1,), dict(x_ptr=x, out_ptr=out, OP=K.REDUCE.index(case.op), LO=lo, HI=hi, **blk),
                {"out_ptr": (ref, xs.abs().sum(-1))}, "sum")
    if case.group == "scan":
        if case.op == "cumprod":
            x = (_input((rows, c), "float32", 0.5, 1.5, g) * torch.where(torch.rand(rows, c, generator=g) < 0.5, -1, 1)).to(DTYPES[dt])
        else:
            x = _input((rows, c), dt, generator=g)
        xs = wide(x)
        ref = {"cumsum": lambda: xs.cumsum(-1), "cumprod": lambda: xs.cumprod(-1), "flip": lambda: xs.flip(-1),
               "sort": lambda: xs.sort(-1).values, "softmax": lambda: torch.softmax(xs, -1)}[case.op]()
        out = torch.empty_like(x)
        return (K.scan_kernel, (1,), dict(x_ptr=x, out_ptr=out, R=rows, C=c, BR=blk["BR"], OP=K.SCAN.index(case.op)),
                {"out_ptr": ref}, "scan" if case.op in ("cumsum", "cumprod") else "float")
    if case.group == "misc":
        x = _input((rows, c), dt, generator=g)
        y = _input((rows, c), dt, generator=g)
        xs, ys = wide(x), wide(y)
        idx = torch.arange(rows * c).reshape(rows, c)
        ref = {"cast": xs, "clamp": xs.clamp(-0.5, 0.5), "fma": xs * ys + xs, "full": xs + 3, "zeros_like": xs,
               "arange": idx.to(x.dtype).to(xs.dtype) + xs}[case.op]
        out = torch.empty(rows, c, dtype=torch.float32 if case.op == "cast" else x.dtype)
        return (K.misc_kernel, (1,), dict(x_ptr=x, y_ptr=y, out_ptr=out, OP=K.MISC.index(case.op), **blk),
                {"out_ptr": ref}, "float")
    if case.group == "shape":
        x = _input(case.shape, dt, generator=g)
        ref = {"trans": x.transpose(1, 2), "reshape": x.reshape(b, r * c), "expand_dims": x,
               "broadcast_to": x[:, :1, :].expand(b, r, c), "permute": x.permute(2, 0, 1)}[case.op].contiguous()
        out = torch.empty_like(ref)
        return (K.shape_kernel, (1,), dict(x_ptr=x, out_ptr=out, B=b, R=r, C=c, OP=K.SHAPE.index(case.op)),
                {"out_ptr": wide(ref)}, "float")
    if case.group == "dot":
        m, n, k = case.shape
        a = _input((m, k), "float32", -1.0, 1.0, g)
        bb = _input((k, n), "float32", -1.0, 1.0, g)
        if dt == "int8":
            a, bb = (a * 50).to(torch.int8), (bb * 50).to(torch.int8)
        else:
            a, bb = a.to(DTYPES[dt]), bb.to(DTYPES[dt])
        out_dtype = {"int8": torch.int32, "float8_e4m3fn": torch.float16}.get(dt, DTYPES[dt])
        ref = a.long() @ bb.long() if dt == "int8" else a.double() @ bb.double()
        bound = a.double().abs() @ bb.double().abs()
        out = torch.empty(m, n, dtype=out_dtype)
        return K.dot_kernel, (1,), dict(a_ptr=a, b_ptr=bb, out_ptr=out, M=m, N=n, K=k), {"out_ptr": (ref, bound)}, "dot"
    if case.group == "memory" and case.op == "masked_store":
        x = _input((rows, c), dt, generator=g)
        out = _input((rows, c), dt, generator=g)
        return (K.masked_store_kernel, (1,), dict(x_ptr=x, out_ptr=out, **blk),
                {"out_ptr": torch.where(wide(x) > 0, wide(x), wide(out))}, "float")
    if case.group == "memory":
        x = _input((rows, c), dt, generator=g)
        out = torch.empty_like(x)
        return (K.descriptor_kernel, (1,), dict(x_ptr=x, out_ptr=out, R=rows, C=c, BR=16, BC=blk["BC"]),
                {"out_ptr": wide(x)}, "float")
    if case.group == "extra":
        return _build_extra(case, rows, c, dt, g)
    if case.group == "program":
        x = _input((rows, c), dt, generator=g)
        return K.program_kernel, (rows,), dict(x_ptr=x, out_ptr=torch.empty_like(x), R=rows, C=c), {"out_ptr": wide(x) + rows}, "float"
    if case.group == "atomic":
        x = _input((rows, c), dt, generator=g)
        rowwise = case.op == "atomic_add_rows"
        out = _input((rows, c) if rowwise else (c,), dt, generator=g)  # the old value takes part
        xs, os_ = wide(x), wide(out)
        if rowwise:
            ref = os_ + xs
        elif case.op == "atomic_add":
            ref = (os_ + xs.sum(0), os_.abs() + xs.abs().sum(0))
        elif case.op in ("atomic_max", "atomic_min"):
            ref = (torch.maximum if case.op == "atomic_max" else torch.minimum)(os_, xs.amax(0) if case.op == "atomic_max" else xs.amin(0))
        else:
            ref = out.clone()
            for r in range(rows):
                ref = {"atomic_and": torch.bitwise_and, "atomic_or": torch.bitwise_or, "atomic_xor": torch.bitwise_xor}[case.op](ref, x[r])
        kind = "sum" if case.op == "atomic_add" else "float"
        return (K.atomic_kernel, (rows,), dict(x_ptr=x, out_ptr=out, R=rows, C=c, OP=K.ATOMIC.index(case.op)),
                {"out_ptr": ref}, kind)
    raise ValueError(case.group)


def _build_extra(case: Case, rows: int, c: int, dt: str, g):
    op = case.op
    x = _input((rows, c), dt, generator=g)
    y = _input((rows, c), dt, generator=g)
    if op == "histogram":
        x = torch.randint(0, c, (rows, c), generator=g, dtype=torch.int32)
    if op in ("umulhi", "reduce_or"):
        x = torch.randint(-2**31, 2**31 - 1, (rows, c), generator=g, dtype=torch.int32)
        y = torch.randint(-2**31, 2**31 - 1, (rows, c), generator=g, dtype=torch.int32)
    if op == "maximum_nan":
        x[torch.rand(rows, c, generator=g) < 0.1] = float("nan")
        y[torch.rand(rows, c, generator=g) < 0.1] = float("nan")
    xs, ys = x.double() if x.is_floating_point() else x.long(), y.double() if y.is_floating_point() else y.long()
    offs = (np.arange(rows)[:, None] * c + np.arange(c)[None, :]).astype(np.int32)
    out_dtype = {"rand": torch.float32, "randn": torch.float32, "rand4x": torch.float32,
                 "randn4x": torch.float32}.get(op, x.dtype)
    out = torch.zeros(rows, c, dtype=out_dtype)
    refs, kind = {}, "float"
    if op == "reduce_or":
        ref = x.clone()
        acc = x[:, 0].clone()
        for j in range(1, c):
            acc |= x[:, j]
        ref[:] = acc[:, None]
    elif op == "topk":
        ref = xs.sort(-1, descending=True).values
    elif op == "map_elementwise":
        ref = torch.sign(xs - ys)
    elif op in ("squeeze_ravel", "device_print", "device_assert"):
        ref = xs
    elif op == "broadcast":
        ref, kind = (xs.sum(1, keepdim=True) + ys, xs.abs().sum(1, keepdim=True) + ys.abs()), "sum"
    elif op in ("rand", "randn", "randint", "rand4x", "randn4x", "randint4x"):
        i1, i2, i3, i4 = _philox(1234, offs)
        ref = torch.from_numpy({
            "randint": lambda: i1.view(np.int32),
            "rand": lambda: _uniform(i1),
            "randn": lambda: _normal(_uniform(i1), _uniform(i2))[0],
            "rand4x": lambda: _uniform(i1) + _uniform(i2) + _uniform(i3) + _uniform(i4),
            "randn4x": lambda: sum(_normal(_uniform(i1), _uniform(i2)) + _normal(_uniform(i3), _uniform(i4))),
            "randint4x": lambda: (i1 ^ i2 ^ i3 ^ i4).view(np.int32),
        }[op]().copy())
    elif op == "associative_scan":
        ref, kind = xs.cumsum(-1), "scan"
    elif op == "reduce_keep_dims":
        ref, kind = (xs * 0 + xs.sum(1, keepdim=True), xs * 0 + xs.abs().sum(1, keepdim=True)), "sum"
    elif op in ("join", "interleave"):
        ref = torch.empty_like(xs)
        ref[:, 0::2], ref[:, 1::2] = xs[:, : c // 2], ys[:, : c // 2]
    elif op == "split":
        ref = torch.cat([xs[:, 0::2], xs[:, 1::2]], 1)
    elif op == "cat":
        ref, kind = xs, "sorted"
    elif op == "swizzle2d":
        ij = torch.from_numpy(offs.astype(np.int64))
        size_g = 4
        off_i = ij // (size_g * c) * size_g
        g_rows = torch.clamp(rows - off_i, max=size_g)
        ij = ij % (size_g * c)
        ref = (off_i + ij % g_rows) * c + ij // g_rows
    elif op == "umulhi":
        a = x.numpy().view(np.uint32).astype(np.uint64)
        b = y.numpy().view(np.uint32).astype(np.uint64)
        ref = torch.from_numpy(((a * b) >> np.uint64(32)).astype(np.uint32).view(np.int32).copy())
    elif op == "gather":
        ref = xs[:, (torch.arange(c) + 1) % c]
    elif op == "gather_data":
        ref = torch.gather(xs, 1, xs.abs() % c)
    elif op == "histogram":
        ref = torch.bincount(x.flatten().long(), minlength=c)[None, :].expand(rows, c)
    elif op == "maximum_nan":
        ref = torch.where(torch.arange(c)[None, :] < c // 2, torch.maximum(xs, ys), torch.fmax(xs, ys))
    elif op in ("atomic_xchg_old", "atomic_add_old", "atomic_cas_old"):
        even = torch.from_numpy(offs % 2 == 0)
        refs["x_ptr"] = {"atomic_xchg_old": ys, "atomic_add_old": xs + ys,
                         "atomic_cas_old": torch.where(xs == xs * even, ys, xs)}[op]  # cmp: x on even offsets, else 0
        ref = xs
    elif op == "while_loop":
        ref = xs * 4
    else:  # range_loop
        ref = xs * 0 + 6
    refs["out_ptr"] = ref
    args = dict(x_ptr=x, y_ptr=y, out_ptr=out, R=rows, C=c, OP=K.EXTRA.index(op))
    return K.extra_kernel, (1,), args, refs, kind


def check(actual: torch.Tensor, expected, kind: str) -> tuple[bool, str]:
    """lowlevel-api-test's rule: the reference cast to the output dtype (so integers wrap as the kernel's
    do), then floats allclose(1e-2, 1e-2, equal_nan) and integers exact. Running sums and reductions
    allow log2(n)*eps*scale more; dots allow the NPU contraction's bf16 operand rounding, 2^-8*(|a|@|b|)."""
    scale = None
    if kind in ("dot", "sum"):
        expected, scale = expected
    expected = expected.to(actual.dtype)
    if kind == "sorted":  # any order (tl.cat with can_reorder)
        actual, expected, kind = actual.flatten().sort().values, expected.flatten().sort().values, "float"
    if not actual.is_floating_point():
        ok = torch.equal(actual, expected)
        return ok, "" if ok else f"{int((actual != expected).sum())} integer mismatches"
    exp = expected.to(torch.float64)
    act = actual.to(torch.float64)
    tol = 1e-2 + 1e-2 * exp.abs()
    eps = torch.finfo(actual.dtype).eps
    if kind == "scan":
        tol = tol + torch.log2(torch.tensor(float(exp.shape[-1]))) * eps * exp.abs().cummax(-1).values
    if kind == "sum":
        tol = tol + torch.log2(torch.tensor(1024.0)) * eps * scale.to(torch.float64)
    if kind == "dot":
        tol = tol + 2.0**-8 * scale
    err = (act - exp).abs()
    nan_ok = torch.isnan(act) & torch.isnan(exp)
    ok = bool(((err <= tol) | nan_ok).all())
    rel = float((err / exp.abs().clamp_min(1e-30)).nan_to_num(0).max()) if err.numel() else 0.0
    return ok, f"max_abs {float(err.nan_to_num(0).max()):.3g} max_rel {rel:.3g}"
