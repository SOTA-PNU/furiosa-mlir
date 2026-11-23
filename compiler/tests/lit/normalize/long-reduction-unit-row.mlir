// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A full reduction longer than a row splits in two levels (upstream splitReduction, which reads every map result as
// a dim): an input read at 0 in a unit dim (one row of a [1, n] slice, a grid-1 norm of the last token) is
// reshaped to drop that dim first, so the split sees dims only.

// CHECK-LABEL: func.func @last_row_sum
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]] : tensor<1x2048xf32> into tensor<2048xf32>
// CHECK: tensor.expand_shape {{.*}} : tensor<2048xf32> into tensor<16x128xf32>
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]
// CHECK: linalg.generic {{.*}}iterator_types = ["reduction"]
#row = affine_map<(d0) -> (0, d0)>
#out = affine_map<(d0) -> ()>
module {
  func.func @last_row_sum(%x: tensor<32x2048xf32>, %o: tensor<f32>) -> tensor<f32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0.0 : f32
    %s = tensor.extract_slice %x[31, 0] [1, 2048] [1, 1] : tensor<32x2048xf32> to tensor<1x2048xf32>
    %e = tensor.empty() : tensor<f32>
    %z = linalg.fill ins(%c0 : f32) outs(%e : tensor<f32>) -> tensor<f32>
    %r = linalg.generic {indexing_maps = [#row, #out], iterator_types = ["reduction"]} ins(%s : tensor<1x2048xf32>) outs(%z : tensor<f32>) {
    ^bb0(%v: f32, %acc: f32):
      %sq = arith.mulf %v, %v : f32
      %a = arith.addf %acc, %sq : f32
      linalg.yield %a : f32
    } -> tensor<f32>
    return %r : tensor<f32>
  }
}
