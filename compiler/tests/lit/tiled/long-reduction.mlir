// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A per-row sum over 8192 columns of 1151 rows (layer norm's mean). The output has no even axis, so its odd
// column pads to whole packets (CP = 1152) and spreads over slices; the reduction, too long for one DM
// tile, reduces in four chunks of 2048 whose partial sums add on the Vector Engine.

// CHECK-LABEL: func.func @row_sum
// CHECK-SAME: ["CP", 1152]
// CHECK-SAME: ["RF0", 2048], ["RC0", 4]
// CHECK-COUNT-4: fopt.vector_intra_slice_reduce
// CHECK: fopt.ve "vector_clip_zip" "ClipBinaryOpF32::Add" {{.*}} zip

#x = affine_map<(d0, d1) -> (d0, d1)>
#y = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_sum(%x: tensor<1151x8192xf32>, %y: tensor<1151xf32>) -> tensor<1151xf32> attributes {furiosa.written = [1]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<1151xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1151xf32>) -> tensor<1151xf32>
    %r = linalg.generic {indexing_maps = [#x, #y], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<1151x8192xf32>) outs(%f : tensor<1151xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %s = arith.addf %acc, %in : f32
      linalg.yield %s : f32
    } -> tensor<1151xf32>
    return %r : tensor<1151xf32>
  }
}
