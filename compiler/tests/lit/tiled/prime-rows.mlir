// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// y[r, c] = x[r, c] * s[r] + w[c] over 1151 (prime) rows. Row-wise refuses it (the NPU cannot window 256 of
// 1151 rows), and the tiled class pads the rows to 1152 (RP0), which split into slices and local rows; the
// padded row is computed and never stored.

// CHECK-LABEL: func.func @scale_rows
// CHECK-SAME: ["RP0", 1152]
// CHECK: fopt.hbm<f32, <"1">, <"A0{1151}, A1{8192}">>

#rc = affine_map<(d0, d1) -> (d0, d1)>
#r = affine_map<(d0, d1) -> (d0)>
#c = affine_map<(d0, d1) -> (d1)>
module {
  func.func @scale_rows(%x: tensor<1151x8192xf32>, %s: tensor<1151xf32>, %w: tensor<8192xf32>, %y: tensor<1151x8192xf32>) -> tensor<1151x8192xf32> attributes {furiosa.written = [3]} {
    %e = tensor.empty() : tensor<1151x8192xf32>
    %r = linalg.generic {indexing_maps = [#rc, #r, #c, #rc], iterator_types = ["parallel", "parallel"]} ins(%x, %s, %w : tensor<1151x8192xf32>, tensor<1151xf32>, tensor<8192xf32>) outs(%e : tensor<1151x8192xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      %t = arith.addf %m, %c : f32
      linalg.yield %t : f32
    } -> tensor<1151x8192xf32>
    return %r : tensor<1151x8192xf32>
  }
}
