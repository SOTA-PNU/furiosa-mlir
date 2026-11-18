// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A loop that reads take apart with floordiv and mod (tl.sort's last stage: its rows, iterated as [4, 2], read
// back as the bits of the [2, 2, 2] stages) splits into those digits, and every read becomes a projection; the
// result is reshaped back to the loop's extent.

// CHECK-LABEL: func.func @digits
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%arg0 : tensor<2x2x2xf32>) outs(%{{.*}} : tensor<2x2x2xf32>)
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1], [2]] : tensor<2x2x2xf32> into tensor<4x2xf32>
#bits = affine_map<(d0, d1) -> (d0 floordiv 2, d0 mod 2, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @digits(%x: tensor<2x2x2xf32>, %o: tensor<4x2xf32>) -> tensor<4x2xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<4x2xf32>
    %r = linalg.generic {indexing_maps = [#bits, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<2x2x2xf32>) outs(%e : tensor<4x2xf32>) {
    ^bb0(%a: f32, %out: f32):
      %s = arith.negf %a : f32
      linalg.yield %s : f32
    } -> tensor<4x2xf32>
    return %r : tensor<4x2xf32>
  }
}
