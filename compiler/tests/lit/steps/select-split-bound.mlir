// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=tensor.insert_slice

// A split only the oracle asks for stands when each piece maps: here the left columns' piece computes a remainder
// no class maps, so the generic stays whole (one diagnostic, at mapping) instead of splitting into pieces, which
// split again by any further loop would multiply.

// CHECK-LABEL: func.func @by_column
// CHECK: arith.remsi
// CHECK: arith.select
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @by_column(%x: tensor<64x64xi32>, %y: tensor<64x64xi32>, %o: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {furiosa.written = [2]} {
    %c31 = arith.constant 31 : index
    %c0 = arith.constant 0 : index
    %e = tensor.empty() : tensor<64x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xi32>, tensor<64x64xi32>) outs(%e : tensor<64x64xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %j = linalg.index 1 : index
      %d = arith.subi %c31, %j : index
      %left = arith.cmpi sge, %d, %c0 : index
      %q = arith.remsi %a, %b : i32
      %s = arith.select %left, %q, %b : i32
      linalg.yield %s : i32
    } -> tensor<64x64xi32>
    return %r : tensor<64x64xi32>
  }
}
