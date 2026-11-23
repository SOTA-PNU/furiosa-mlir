// RUN: furiosa-mlir-opt --furiosa-hoist-index-only %s | FileCheck %s --implicit-check-not=linalg.index

// An index compared with data (a loop counter against the bounds a kernel loads, u - 2 < ub, and the step test
// (u - 2 - lb) & 3 == 0) addresses nothing: it is a table too, read as an index, so the Vector Engine compares
// data with data.

// CHECK-LABEL: func.func @bounds
// CHECK: %[[T:.*]] = arith.constant dense<[-2, -1, 0, 1, 2, 3, 4, 5]> : tensor<8xi32>
// CHECK: linalg.generic {{.*}} ins(%arg0, %arg1, %arg2, %[[T]] : tensor<4x8xf32>, tensor<4xi32>, tensor<4xi32>, tensor<8xi32>)
// CHECK: %[[U:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: arith.cmpi slt, %[[U]],
// CHECK: arith.subi %[[U]],
#x = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_map<(d0, d1) -> (d0)>
module {
  func.func @bounds(%x: tensor<4x8xf32>, %lb: tensor<4xi32>, %ub: tensor<4xi32>, %o: tensor<4xf32>) -> tensor<4xf32> attributes {furiosa.written = [3]} {
    %c-2 = arith.constant -2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %z = arith.constant -0.000000e+00 : f32
    %r = linalg.generic {indexing_maps = [#x, #s, #s, #s], iterator_types = ["parallel", "reduction"]} ins(%x, %lb, %ub : tensor<4x8xf32>, tensor<4xi32>, tensor<4xi32>) outs(%o : tensor<4xf32>) {
    ^bb0(%v: f32, %l: i32, %h: i32, %acc: f32):
      %u = linalg.index 1 : index
      %r0 = arith.addi %u, %c-2 : index
      %li = arith.index_cast %l : i32 to index
      %hi = arith.index_cast %h : i32 to index
      %in = arith.cmpi slt, %r0, %hi : index
      %off = arith.subi %r0, %li : index
      %low = arith.andi %off, %c3 : index
      %step = arith.cmpi eq, %low, %c0 : index
      %both = arith.andi %in, %step : i1
      %t = arith.select %both, %v, %z : f32
      %s = arith.addf %acc, %t : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
