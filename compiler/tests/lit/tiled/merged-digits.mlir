// RUN: env FURIOSA_TILED_DEBUG=1 furiosa-mlir-opt --furiosa-linalg-to-fopt %s 2>&1 | FileCheck %s

// A contraction whose rows (256) and columns (128) are read from buffers that split them (2 x 128, 4 x 32): the
// reshapes take each loop apart into digits, which every index reads together (adjacent buffer dims, the lower
// one whole), so they merge back into one row axis and one column axis and the contraction maps. 2^x folds as
// e^(x ln 2).

// CHECK: tiled: rows:
// CHECK-NEXT: axes: a0=256 a1=128 a2=64r
// CHECK-NEXT: out: a0 a1
// CHECK-NEXT: store arg2[a0, a1]
// CHECK: fopt.vector_intra_slice_reduce
#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#y = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @rows(%a: tensor<2x128x64xf32>, %b: tensor<4x32x64xf32>, %o: tensor<2x128x4x32xf32>) -> tensor<2x128x4x32xf32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0.0 : f32
    %x = tensor.collapse_shape %a [[0, 1], [2]] : tensor<2x128x64xf32> into tensor<256x64xf32>
    %w = tensor.collapse_shape %b [[0, 1], [2]] : tensor<4x32x64xf32> into tensor<128x64xf32>
    %e = tensor.empty() : tensor<256x128xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<256x128xf32>) -> tensor<256x128xf32>
    %r = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<256x64xf32>, tensor<128x64xf32>) outs(%f : tensor<256x128xf32>) {
    ^bb0(%p: f32, %q: f32, %acc: f32):
      %t = math.exp2 %p : f32
      %m = arith.mulf %t, %q : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<256x128xf32>
    %out = tensor.expand_shape %r [[0, 1], [2, 3]] output_shape [2, 128, 4, 32] : tensor<256x128xf32> into tensor<2x128x4x32xf32>
    return %out : tensor<2x128x4x32xf32>
  }
}
