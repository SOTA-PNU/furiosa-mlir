// RUN: furiosa-mlir-opt --furiosa-hoist-index-only %s | FileCheck %s

// Philox-style counters (tl.rand): an index-only chain through mului_extended's high half is evaluated at compile
// time like any index-only value, so the Vector Engine never needs a 32x32->64 multiply. Here (i * 2^31) >> 32 over
// i in 0..3 gives 0, 0, 1, 1.

// CHECK-LABEL: func.func @umulhi
// CHECK: arith.constant dense<[0, 0, 1, 1]> : tensor<4xi32>
// CHECK-NOT: arith.mului_extended
#id = affine_map<(d0) -> (d0)>
module {
  func.func @umulhi(%x: tensor<4xi32>, %o: tensor<4xi32>) -> tensor<4xi32> attributes {furiosa.written = [1]} {
    %c = arith.constant -2147483648 : i32
    %e = tensor.empty() : tensor<4xi32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%x : tensor<4xi32>) outs(%e : tensor<4xi32>) {
    ^bb0(%a: i32, %out: i32):
      %i = linalg.index 0 : index
      %ii = arith.index_cast %i : index to i32
      %lo, %hi = arith.mului_extended %ii, %c : i32
      %s = arith.addi %a, %hi : i32
      linalg.yield %s : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
