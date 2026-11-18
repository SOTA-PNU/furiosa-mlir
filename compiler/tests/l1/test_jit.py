"""Checks the L1 JIT on the constructs raised kernels use, against numpy.

    PYTHONPATH=compiler compiler/.venv/bin/python compiler/tests/l1/test_jit.py
"""

import ml_dtypes
import numpy as np
from triton._C.libtriton import furiosa

from furiosa_compiler import tensors

BF16 = ml_dtypes.bfloat16

MODULE = r"""
#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  // bf16 in/out, f32 math: y = x * rsqrt(mean(x^2) + eps) * w, one row per d0.
  func.func public @rmsnorm(%x: tensor<4x64xbf16>, %w: tensor<64xbf16>, %y: tensor<4x64xbf16>) -> tensor<4x64xbf16> {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<4xf32>
    %acc = linalg.fill ins(%zero : f32) outs(%e : tensor<4xf32>) -> tensor<4xf32>
    %ss = linalg.generic {indexing_maps = [#id2, #row], iterator_types = ["parallel", "reduction"]}
        ins(%x : tensor<4x64xbf16>) outs(%acc : tensor<4xf32>) {
    ^bb0(%a: bf16, %s: f32):
      %f = arith.extf %a : bf16 to f32
      %sq = arith.mulf %f, %f : f32
      %r = arith.addf %s, %sq : f32
      linalg.yield %r : f32
    } -> tensor<4xf32>
    %out = linalg.generic {indexing_maps = [#id2, #row, #col, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%x, %ss, %w : tensor<4x64xbf16>, tensor<4xf32>, tensor<64xbf16>) outs(%y : tensor<4x64xbf16>) {
    ^bb0(%a: bf16, %s: f32, %b: bf16, %o: bf16):
      %n = arith.constant 64.0 : f32
      %eps = arith.constant 1.0e-6 : f32
      %m = arith.divf %s, %n : f32
      %me = arith.addf %m, %eps : f32
      %rs = math.rsqrt %me : f32
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %t = arith.mulf %af, %rs : f32
      %u = arith.mulf %t, %bf : f32
      %v = arith.truncf %u : f32 to bf16
      linalg.yield %v : bf16
    } -> tensor<4x64xbf16>
    return %out : tensor<4x64xbf16>
  }

  // matmul with a K loop kept as scf.for over 16-wide tiles (iter_args carry the f32 accumulator).
  func.func public @matmul(%a: tensor<8x64xbf16>, %b: tensor<32x64xbf16>, %c: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<8x32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%e : tensor<8x32xf32>) -> tensor<8x32xf32>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %acc = scf.for %k = %c0 to %c64 step %c16 iter_args(%t = %init) -> tensor<8x32xf32> {
      %at = tensor.extract_slice %a[0, %k] [8, 16] [1, 1] : tensor<8x64xbf16> to tensor<8x16xbf16>
      %bt = tensor.extract_slice %b[0, %k] [32, 16] [1, 1] : tensor<32x64xbf16> to tensor<32x16xbf16>
      %r = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>]
          ins(%at, %bt : tensor<8x16xbf16>, tensor<32x16xbf16>) outs(%t : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.yield %r : tensor<8x32xf32>
    }
    %out = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%acc : tensor<8x32xf32>) outs(%c : tensor<8x32xbf16>) {
    ^bb0(%v: f32, %o: bf16):
      %t = arith.truncf %v : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<8x32xbf16>
    return %out : tensor<8x32xbf16>
  }

  // A partial in-place write: rows [2, 5) of a cache get new values, the rest keep the argument's.
  // Also a second result and a padded read: out2[i] = xpad[i] where xpad pads x by 3 zeros on both sides.
  func.func public @partial(%rows: tensor<3x16xf32>, %cache: tensor<8x16xf32>, %x: tensor<10xf32>, %out2: tensor<16xf32>)
      -> (tensor<8x16xf32>, tensor<16xf32>) {
    %0 = tensor.insert_slice %rows into %cache[2, 0] [3, 16] [1, 1] : tensor<3x16xf32> into tensor<8x16xf32>
    %z = arith.constant 0.0 : f32
    %p = tensor.pad %x low[3] high[3] {
    ^bb0(%i: index):
      tensor.yield %z : f32
    } : tensor<10xf32> to tensor<16xf32>
    %1 = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%p : tensor<16xf32>) outs(%out2 : tensor<16xf32>) {
    ^bb0(%a: f32, %o: f32):
      %s = math.sin %a : f32
      %c = math.cos %a : f32
      %t = math.tanh %a : f32
      %e = math.exp %a : f32
      %r0 = arith.addf %s, %c : f32
      %r1 = arith.addf %r0, %t : f32
      %r2 = arith.addf %r1, %e : f32
      linalg.yield %r2 : f32
    } -> tensor<16xf32>
    return %0, %1 : tensor<8x16xf32>, tensor<16xf32>
  }

  // i32 index math and a reshape: out[i] = (ids[i] * 3) mod 7 over a 2x4 view of 8 ids; f16 output.
  func.func public @ints(%ids: tensor<8xi32>, %out: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %v = tensor.expand_shape %ids [[0, 1]] output_shape [2, 4] : tensor<8xi32> into tensor<2x4xi32>
    %r = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%v : tensor<2x4xi32>) outs(%out : tensor<2x4xf16>) {
    ^bb0(%a: i32, %o: f16):
      %c3 = arith.constant 3 : i32
      %c7 = arith.constant 7 : i32
      %m = arith.muli %a, %c3 : i32
      %q = arith.remsi %m, %c7 : i32
      %f = arith.sitofp %q : i32 to f16
      linalg.yield %f : f16
    } -> tensor<2x4xf16>
    return %r : tensor<2x4xf16>
  }
  // A round trip through fp8 e4m3 (tl.float8e4nv): truncf encodes, extf decodes.
  func.func public @fp8(%x: tensor<65544xf32>, %out: tensor<65544xf32>) -> tensor<65544xf32> {
    %r = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%x : tensor<65544xf32>) outs(%out : tensor<65544xf32>) {
    ^bb0(%a: f32, %o: f32):
      %q = arith.truncf %a : f32 to f8E4M3FN
      %w = arith.extf %q : f8E4M3FN to f32
      linalg.yield %w : f32
    } -> tensor<65544xf32>
    return %r : tensor<65544xf32>
  }
}
"""


# Gather form (compiler/README.md): bodies read buffers with tensor.extract at addresses computed from
# linalg.index; masked reads select over a clamped address; reduction extents ride on an unread
# tensor.empty input.
GATHER = r"""
module {
  // out[i] = (x[i - 3] if 0 <= i - 3 < 10 else 0) + y[i]
  func.func public @shifted(%x: tensor<10xf32>, %y: tensor<16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%out : tensor<16xf32>) {
    ^bb0(%o: f32):
      %i = linalg.index 0 : index
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c9 = arith.constant 9 : index
      %j = arith.subi %i, %c3 : index
      %lo = arith.cmpi sge, %j, %c0 : index
      %hi = arith.cmpi sle, %j, %c9 : index
      %ok = arith.andi %lo, %hi : i1
      %j0 = arith.maxsi %j, %c0 : index
      %jc = arith.minsi %j0, %c9 : index
      %v = tensor.extract %x[%jc] : tensor<10xf32>
      %z = arith.constant 0.0 : f32
      %m = arith.select %ok, %v, %z : f32
      %w = tensor.extract %y[%i] : tensor<16xf32>
      %s = arith.addf %m, %w : f32
      linalg.yield %s : f32
    } -> tensor<16xf32>
    return %r : tensor<16xf32>
  }

  // s[r] = sum_c x[r * 64 + c] over a flat bf16 buffer; the reduction extent rides on %carrier.
  // Then t[r, c] = x[r * 64 + c] * s[r]: an extract from the intermediate tensor.
  func.func public @rowsum(%x: tensor<256xbf16>, %s: tensor<4xf32>, %t: tensor<4x64xf32>) -> (tensor<4xf32>, tensor<4x64xf32>) {
    %z = arith.constant 0.0 : f32
    %init = linalg.fill ins(%z : f32) outs(%s : tensor<4xf32>) -> tensor<4xf32>
    %carrier = tensor.empty() : tensor<64xf32>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]}
        ins(%carrier : tensor<64xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%unused: f32, %acc: f32):
      %r = linalg.index 0 : index
      %c = linalg.index 1 : index
      %c64 = arith.constant 64 : index
      %rb = arith.muli %r, %c64 : index
      %a = arith.addi %rb, %c : index
      %v = tensor.extract %x[%a] : tensor<256xbf16>
      %f = arith.extf %v : bf16 to f32
      %n = arith.addf %acc, %f : f32
      linalg.yield %n : f32
    } -> tensor<4xf32>
    %scaled = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%t : tensor<4x64xf32>) {
    ^bb0(%o: f32):
      %r = linalg.index 0 : index
      %c = linalg.index 1 : index
      %c64 = arith.constant 64 : index
      %rb = arith.muli %r, %c64 : index
      %a = arith.addi %rb, %c : index
      %v = tensor.extract %x[%a] : tensor<256xbf16>
      %f = arith.extf %v : bf16 to f32
      %w = tensor.extract %sum[%r] : tensor<4xf32>
      %p = arith.mulf %f, %w : f32
      linalg.yield %p : f32
    } -> tensor<4x64xf32>
    return %sum, %scaled : tensor<4xf32>, tensor<4x64xf32>
  }

  // out[t, h] = table[ids[t], h]: a data-dependent address is just more arithmetic.
  func.func public @gather(%ids: tensor<5xi32>, %table: tensor<32x8xbf16>, %out: tensor<5x8xbf16>) -> tensor<5x8xbf16> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%out : tensor<5x8xbf16>) {
    ^bb0(%o: bf16):
      %t = linalg.index 0 : index
      %h = linalg.index 1 : index
      %id = tensor.extract %ids[%t] : tensor<5xi32>
      %row = arith.index_cast %id : i32 to index
      %v = tensor.extract %table[%row, %h] : tensor<32x8xbf16>
      linalg.yield %v : bf16
    } -> tensor<5x8xbf16>
    return %r : tensor<5x8xbf16>
  }

  // out[m, n] = x[(n * 6 + m) floordiv 2 ... ] style flat addressing: out[m, n] = x[(m * 8 + n) mod 24 floordiv 3]
  func.func public @flat(%x: tensor<8xi32>, %out: tensor<3x8xi32>) -> tensor<3x8xi32> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%out : tensor<3x8xi32>) {
    ^bb0(%o: i32):
      %m = linalg.index 0 : index
      %n = linalg.index 1 : index
      %c8 = arith.constant 8 : index
      %c24 = arith.constant 24 : index
      %c3 = arith.constant 3 : index
      %mb = arith.muli %m, %c8 : index
      %a = arith.addi %mb, %n : index
      %w = arith.remsi %a, %c24 : index
      %q = arith.divsi %w, %c3 : index
      %v = tensor.extract %x[%q] : tensor<8xi32>
      linalg.yield %v : i32
    } -> tensor<3x8xi32>
    return %r : tensor<3x8xi32>
  }
}
"""


def gather_form(rng):
    engine = furiosa.runtime.compile(GATHER)

    x = rng.standard_normal(10).astype(np.float32)
    y = rng.standard_normal(16).astype(np.float32)
    out = np.empty(16, np.float32)
    tensors.call(engine.lookup("shifted"), [x, y, np.zeros_like(out), out])
    assert np.array_equal(out, np.pad(x, (3, 3)) + y), "shifted masked read"

    xb = rng.standard_normal(256).astype(BF16)
    s, t = np.empty(4, np.float32), np.empty((4, 64), np.float32)
    tensors.call(engine.lookup("rowsum"), [xb, np.zeros_like(s), np.zeros_like(t), s, t])
    rows = xb.astype(np.float32).reshape(4, 64)
    assert np.allclose(s, rows.sum(1), rtol=1e-6, atol=1e-5), "carrier reduction"
    assert np.allclose(t, rows * s[:, None], rtol=1e-6), "extract from an intermediate"

    ids = rng.integers(0, 32, 5).astype(np.int32)
    table = rng.standard_normal((32, 8)).astype(BF16)
    out = np.empty((5, 8), BF16)
    tensors.call(engine.lookup("gather"), [ids, table, np.zeros_like(out), out])
    assert np.array_equal(out, table[ids]), "data-dependent gather"

    xi = rng.integers(0, 1000, 8).astype(np.int32)
    out = np.empty((3, 8), np.int32)
    tensors.call(engine.lookup("flat"), [xi, np.zeros_like(out), out])
    m, n = np.meshgrid(np.arange(3), np.arange(8), indexing="ij")
    assert np.array_equal(out, xi[((m * 8 + n) % 24) // 3]), "flat div/mod addressing"
    print("gather form: masked shifted read, carrier reduction, intermediate extract, id gather, div/mod: exact")


def bf16_round(x):
    return x.astype(BF16).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    engine = furiosa.runtime.compile(MODULE)

    x = rng.standard_normal((4, 64)).astype(BF16)
    w = rng.standard_normal(64).astype(BF16)
    y = np.empty((4, 64), BF16)
    tensors.call(engine.lookup("rmsnorm"), [x, w, np.zeros_like(y), y])
    xf, wf = x.astype(np.float32), w.astype(np.float32)
    want = xf / np.sqrt((xf * xf).mean(1, keepdims=True) + 1e-6) * wf
    err = np.abs(y.astype(np.float32) - want).max() / np.abs(want).max()
    assert err < 8e-3, err
    print(f"rmsnorm bf16: rel err {err:.2g}")

    a = rng.standard_normal((8, 64)).astype(BF16)
    b = rng.standard_normal((32, 64)).astype(BF16)
    c = np.empty((8, 32), BF16)
    tensors.call(engine.lookup("matmul"), [a, b, np.zeros_like(c), c])
    want = a.astype(np.float32) @ b.astype(np.float32).T
    err = np.abs(c.astype(np.float32) - want).max() / np.abs(want).max()
    assert err < 8e-3, err
    print(f"matmul scf.for + linalg.matmul (transposed B): rel err {err:.2g}")

    rows = rng.standard_normal((3, 16)).astype(np.float32)
    cache = rng.standard_normal((8, 16)).astype(np.float32)
    xs = rng.standard_normal(10).astype(np.float32)
    cache_before = cache.copy()
    new_cache, out2 = np.empty_like(cache), np.empty(16, np.float32)
    tensors.call(engine.lookup("partial"), [rows, cache, xs, np.zeros(16, np.float32), new_cache, out2])
    want_cache = cache.copy()
    want_cache[2:5] = rows
    assert np.array_equal(new_cache, want_cache), "partial write"
    assert np.array_equal(cache, cache_before), "an argument was written"
    p = np.pad(xs, 3)
    want = np.sin(p) + np.cos(p) + np.tanh(p) + np.exp(p)
    err = np.abs(out2 - want).max()
    assert err < 1e-5, err
    print(f"insert_slice partial write, two results, tensor.pad, sin/cos/tanh/exp: err {err:.2g}; args untouched")

    ids = rng.integers(0, 100, 8).astype(np.int32)
    out = np.empty((2, 4), np.float16)
    tensors.call(engine.lookup("ints"), [ids, np.zeros_like(out), out])
    assert np.array_equal(out, ((ids * 3) % 7).reshape(2, 4).astype(np.float16)), "ints"
    print("i32 muli/remsi, expand_shape, sitofp to f16: exact")

    # Every bf16, then infinities, overflow and NaN: nearest even, saturating to +-448 as the GPU converts.
    xs = np.concatenate([np.arange(1 << 16, dtype=np.uint16).view(BF16).astype(np.float32),
                         np.float32([np.inf, -np.inf, 463.9, 464.0, -1e30, 0.01171875, 0.009765625, np.nan])])
    out = np.empty_like(xs)
    tensors.call(engine.lookup("fp8"), [xs, np.zeros_like(out), out])
    want = np.clip(np.nan_to_num(xs, nan=np.nan, posinf=448, neginf=-448), -448, 448).astype(ml_dtypes.float8_e4m3fn)
    assert np.array_equal(out, want.astype(np.float32), equal_nan=True), "fp8"
    print("f32 -> f8E4M3FN -> f32 (every bf16, saturating): exact")

    x, y = rng.standard_normal(16).astype(np.float32), np.empty(16, np.float32)
    path = "/tmp/claude-2006/-home-chacha-git-furiosa-mlir/d735c3dd-63aa-4a2f-bd7a-bbb5a499ef73/scratchpad/roundtrip.safetensors"
    tensors.save(path, {"a": x, "b": a, "c": ids})
    back = tensors.load(path)
    assert np.array_equal(back["a"], x) and np.array_equal(back["b"], a) and np.array_equal(back["c"], ids)
    print("safetensors round trip (f32, bf16, i32): exact")

    gather_form(rng)


if __name__ == "__main__":
    main()
