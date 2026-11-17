// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=arith.select

// A generic that maps no other way, whose select between computed values is keyed by a comparison one loop
// decides (columns < 32: tl.maximum's NaN modes by column), splits there: each run computes its own side.

// CHECK-LABEL: func.func private @by_column__s0
// CHECK: arith.maximumf
// CHECK-LABEL: func.func private @by_column__s{{[0-9]+}}
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [64, 32] [1, 1]
// CHECK-LABEL: func.func private @by_column__s{{[0-9]+}}
// CHECK: arith.maxnumf
// CHECK-LABEL: func.func private @by_column__s{{[0-9]+}}
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 32] [64, 32] [1, 1]
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @by_column(%x: tensor<64x64xf32>, %y: tensor<64x64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %c31 = arith.constant 31 : index
    %c0 = arith.constant 0 : index
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %j = linalg.index 1 : index
      %d = arith.subi %c31, %j : index
      %left = arith.cmpi sge, %d, %c0 : index
      %all = arith.maximumf %a, %b : f32
      %none = arith.maxnumf %a, %b : f32
      %s = arith.select %left, %all, %none : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
