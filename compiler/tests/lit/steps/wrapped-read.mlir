// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// tl.gather by (j + 1) % C reads x[i, (j + 1) mod 64]. The read wraps at j = 63: split there, each run's read is
// linear (columns 1..63, then column 0), and each run is a box copy inserted into the result.

// CHECK-LABEL: func.func @rotate(
// CHECK-COUNT-4: call @rotate__s
// CHECK-LABEL: func.func private @rotate__s0
// CHECK: tensor.extract_slice %{{.*}}[0, 1] [64, 63] [1, 1]
// CHECK-LABEL: func.func private @rotate__s2
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [64, 1] [1, 1]
#wrap = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 64)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @rotate(%x: tensor<64x64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#wrap, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%v: f32, %out: f32):
      linalg.yield %v : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
