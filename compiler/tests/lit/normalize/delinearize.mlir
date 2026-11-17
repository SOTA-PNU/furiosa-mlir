// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A read of a flat buffer at a mixed-radix address (here the head-split q[t, h*128 + d] of an attention, as
// raised) becomes a projection of the buffer expanded by the address's digits: q as [t, h, d] read at (t, h, d).

// CHECK-LABEL: func.func @heads
// CHECK: %[[Q:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [4, 2, 128] : tensor<4x256xf32> into tensor<4x2x128xf32>
// CHECK: linalg.generic {indexing_maps = [#[[ID:map[0-9]*]], #[[ID]]], {{.*}} ins(%[[Q]] : tensor<4x2x128xf32>)
#flat = affine_map<(d0, d1, d2) -> (d0, d1 * 128 + d2)>
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @heads(%q: tensor<4x256xf32>, %o: tensor<4x2x128xf32>) -> tensor<4x2x128xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<4x2x128xf32>
    %r = linalg.generic {indexing_maps = [#flat, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%q : tensor<4x256xf32>) outs(%e : tensor<4x2x128xf32>) {
    ^bb0(%a: f32, %out: f32):
      %s = arith.mulf %a, %a : f32
      linalg.yield %s : f32
    } -> tensor<4x2x128xf32>
    return %r : tensor<4x2x128xf32>
  }
}
