// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=tensor.collapse_shape

// Loops collapse only when the collapsed generic maps: this int8 select (a masked store's), which the oracle reports
// unmapped as it is, keeps its two loops; one loop of 4096 elements would map no better.

// CHECK-LABEL: func.func @keep
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @keep(%x: tensor<64x64xi8>, %y: tensor<64x64xi8>, %o: tensor<64x64xi8>) -> tensor<64x64xi8> attributes {furiosa.written = [2]} {
    %c0 = arith.constant 0 : i8
    %e = tensor.empty() : tensor<64x64xi8>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xi8>, tensor<64x64xi8>) outs(%e : tensor<64x64xi8>) {
    ^bb0(%a: i8, %b: i8, %out: i8):
      %p = arith.cmpi sgt, %a, %c0 : i8
      %s = arith.select %p, %a, %b : i8
      linalg.yield %s : i8
    } -> tensor<64x64xi8>
    return %r : tensor<64x64xi8>
  }
}
