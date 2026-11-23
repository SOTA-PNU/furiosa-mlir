// RUN: furiosa-mlir-opt --furiosa-tile %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=FOPT

// A contraction whose K (2048) takes two K tiles at N = 6144 returns its K loop's result directly (a linear
// layer's step before its epilogue). The tiled class maps it; the row-wise class, which folds row tile loops,
// declines a K loop (not a row tile nest) instead of reading a nest that is not there.

// CHECK-LABEL: func.func @linear_k
// CHECK: %[[K:.*]] = scf.for
// CHECK: } {furiosa.tile_dim = 2 : i64}
// CHECK-NEXT: return %[[K]]

// FOPT-LABEL: func.func @linear_k
// FOPT: fopt.
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d1, d2)>
#y = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @linear_k(%x: tensor<32x2048xbf16>, %w: tensor<6144x2048xbf16>, %o: tensor<32x6144xf32>) -> tensor<32x6144xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x6144xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x6144xf32>) -> tensor<32x6144xf32>
    %2 = linalg.generic {indexing_maps = [#a, #b, #y], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<32x2048xbf16>, tensor<6144x2048xbf16>) outs(%1 : tensor<32x6144xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<32x6144xf32>
    return %2 : tensor<32x6144xf32>
  }
}
