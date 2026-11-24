// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// Two reduction loops read only as one value, k = d2 * 4 + d3 (an implicit GEMM's K stepped in tiles of 4), that a
// floordiv takes apart where the tiles have no boundary (k // 3: the channel of 3 taps) merge into one loop k, which
// then splits at 3 into the channel and the tap: the weight becomes a projection of [N, C, 3], the input a window
// linear in the loops.

// CHECK-DAG: #[[X:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2 + d3 * 4)>
// CHECK-DAG: #[[W:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-LABEL: func.func @taps
// CHECK: %[[W3:.*]] = tensor.expand_shape %{{.*}} : tensor<24xf32> into tensor<2x4x3xf32>
// CHECK: linalg.generic {indexing_maps = [#[[X]], #[[W]], #{{.*}}], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.*}}, %[[W3]] : tensor<24xf32>, tensor<2x4x3xf32>)
#x = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2 * 16 + d3 * 4 - ((d2 * 4 + d3) floordiv 3) * 11)>
#w = affine_map<(d0, d1, d2, d3) -> (d1 * 12 + d2 * 4 + d3)>
#k = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#o = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @taps(%x: tensor<24xf32>, %w: tensor<24xf32>, %y: tensor<4x2xf32>) -> tensor<4x2xf32> attributes {furiosa.written = [2]} {
    %c0 = arith.constant 0.0 : f32
    %k = tensor.empty() : tensor<3x4xi8>
    %e = tensor.empty() : tensor<4x2xf32>
    %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
    %r = linalg.generic {indexing_maps = [#k, #x, #w, #o], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%k, %x, %w : tensor<3x4xi8>, tensor<24xf32>, tensor<24xf32>) outs(%f : tensor<4x2xf32>) {
    ^bb0(%c: i8, %a: f32, %b: f32, %out: f32):
      %m = arith.mulf %a, %b : f32
      %s = arith.addf %out, %m : f32
      linalg.yield %s : f32
    } -> tensor<4x2xf32>
    return %r : tensor<4x2xf32>
  }
}
