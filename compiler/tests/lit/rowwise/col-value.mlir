// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A column value computed from a column input (a token's row offset ids[t] * 96) against a row value (each row's
// offset): the key streams from the column input's DM tile (its load broadcasts it over the rows) through the
// computation, the row value (negated: the difference is an add) read from the VRF, as a column input itself does.

// CHECK-LABEL: func.func @col_value
// CHECK: %[[C:.*]] = fopt.view_to_dm %{{.*}} : !fopt.hbmv<i32, <"1">, <"H{32}">, false> -> !fopt.dm<i32, <"1">, <"1 # 2">, <"R{8} # 256">, <"H{32}">>
// CHECK: fopt.begin main %[[C]]
// CHECK: "FxpBinaryOp::MulInt"
// CHECK: "FxpBinaryOp::AddFxp" %{{.*}} vrf
// CHECK: fopt.vector_intra_slice_reduce "Add"
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @col_value(%dy: tensor<8x32xf32>, %ids: tensor<32xi32>, %off: tensor<8xi32>, %o: tensor<8xf32>) -> tensor<8xf32> attributes {furiosa.written = [3]} {
    %c96 = arith.constant 96 : i32
    %z = arith.constant -0.000000e+00 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<8xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
    %r = linalg.generic {indexing_maps = [#id, #col, #row, #row], iterator_types = ["parallel", "reduction"]} ins(%dy, %ids, %off : tensor<8x32xf32>, tensor<32xi32>, tensor<8xi32>) outs(%f : tensor<8xf32>) {
    ^bb0(%x: f32, %id: i32, %v: i32, %acc: f32):
      %a = arith.muli %id, %c96 : i32
      %hit = arith.cmpi eq, %a, %v : i32
      %t = arith.select %hit, %x, %z : f32
      %s = arith.addf %acc, %t : f32
      linalg.yield %s : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
