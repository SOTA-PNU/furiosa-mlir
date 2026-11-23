// RUN: furiosa-mlir-opt --furiosa-hoist-index-only %s | FileCheck %s --implicit-check-not=arith.select --implicit-check-not=arith.mulf --implicit-check-not=dense

// A mask true at every point of its range (tl.where(offs < N, x - mean, 0) once the row runs over its N live
// columns) has one value: it becomes a constant, not a table input, and the select it guards folds away.

// CHECK-LABEL: func.func @live
// CHECK: linalg.generic {{.*}} ins(%arg0 : tensor<8xf32>)
// CHECK-NEXT: ^bb0(%[[A:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[N:.*]] = arith.negf %[[A]] : f32
// CHECK-NEXT: linalg.yield %[[N]] : f32
#id = affine_map<(d0) -> (d0)>
module {
  func.func @live(%x: tensor<8xf32>, %o: tensor<8xf32>) -> tensor<8xf32> attributes {furiosa.written = [1]} {
    %c7 = arith.constant 7 : index
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<8xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%x : tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%a: f32, %out: f32):
      %d = linalg.index 0 : index
      %left = arith.subi %c7, %d : index
      %live = arith.cmpi sge, %left, %c0 : index
      %s = arith.select %live, %a, %zero : f32
      %n = arith.negf %s : f32
      linalg.yield %n : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
