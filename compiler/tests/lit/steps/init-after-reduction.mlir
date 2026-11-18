// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A reduction that starts from a computed value (the buffer's own, as an atomic that every program applies to the
// same elements raises) and maps no other way reduces from its combiner's identity, as the row-wise class does, and
// the value joins the total after.

// CHECK-LABEL: func.func @atomic_max
// CHECK: %[[F:.*]] = linalg.fill ins(%c-2147483648_i32 : i32)
// CHECK: %[[R:.*]] = linalg.generic {{.*}}["parallel", "reduction"]{{.*}} outs(%[[F]] : tensor<64xi32>)
// CHECK: arith.maxsi
// CHECK: linalg.generic {{.*}}["parallel"]{{.*}} ins(%arg1, %[[R]] : tensor<64xi32>, tensor<64xi32>)
// CHECK-NEXT: ^bb0(%[[OLD:.*]]: i32, %[[RED:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT: arith.maxsi %[[OLD]], %[[RED]] : i32
#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @atomic_max(%x: tensor<64x8xi32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [1]} {
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<64x8xi32>) outs(%o : tensor<64xi32>) {
    ^bb0(%a: i32, %acc: i32):
      %m = arith.maxsi %acc, %a : i32
      linalg.yield %m : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
