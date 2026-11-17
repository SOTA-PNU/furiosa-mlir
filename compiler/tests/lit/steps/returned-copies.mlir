// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// tl.atomic_cas returning the old values: a buffer the kernel returns into another written buffer is a copy (the
// old values, s0, run before anything overwrites them), and the step that reads %arg0 cannot also take %arg0 as
// its destination (the runtime refuses the aliasing): its result goes to a temporary, copied into %arg0 last.

// CHECK-LABEL: func.func @cas(
// CHECK: call @cas__s0(%arg0, %arg2)
// CHECK: %[[T:.*]] = call @cas__s1(
// CHECK: call @cas__s2(%[[T]], %{{.*}})
// CHECK-LABEL: func.func private @cas__s2
// CHECK: linalg.yield %in
#map = affine_map<(d0, d1, d2) -> (d0, d1 * 2 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @cas(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi32>) -> (tensor<64x64xi32>, tensor<64x64xi32>) attributes {furiosa.written = [0, 2]} {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [64, 32, 2] : tensor<64x64xi32> into tensor<64x32x2xi32>
    %0 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%expanded : tensor<64x32x2xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %1 = linalg.index 2 : index
      %2 = arith.cmpi sge, %1, %c0 : index
      %3 = arith.muli %1, %c-1 : index
      %4 = arith.cmpi sge, %3, %c0 : index
      %5 = arith.andi %2, %4 : i1
      %6 = arith.extui %5 : i1 to i32
      %7 = arith.muli %6, %in_0 : i32
      %8 = arith.cmpi eq, %out, %7 : i32
      %9 = arith.select %8, %in, %out : i32
      linalg.yield %9 : i32
    } -> tensor<64x32x2xi32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<64x32x2xi32> into tensor<64x64xi32>
    return %collapsed, %arg0 : tensor<64x64xi32>, tensor<64x64xi32>
  }
}
