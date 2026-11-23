// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt --split-input-file %s | FileCheck %s

// @masked_sum (segment offsets: lens[j] summed where a table selects it, over 13 columns padded to 16): a term of
// several inputs pads the one whose identity padding alone folds it to the identity: lens with 0, whatever the
// table holds past it (a select whose sides fold alike is that value).

// CHECK-LABEL: func.func @masked_sum
// CHECK: fopt.memset %{{.*}}, 0.000000e+00 : !fopt.dm<i32
// CHECK: fopt.vector_intra_slice_reduce "AddSat"

// @col_key (top-2 gating's second pick: -1 where the column is the row's first pick): cols == pick[r] combines a
// column input and a row value into a full key. The column input's DM tile holds its columns in every row's slice
// (its load broadcasts it over the rows), so it streams, and pick comes from the VRF.

// CHECK-LABEL: func.func @col_key
// CHECK: %[[C:.*]] = fopt.view_to_dm %{{.*}} : !fopt.hbmv<i32, <"1">, <"H{16}">, false> -> !fopt.dm<i32, <"1">, <"1 # 2">, <"R{8} # 256">, <"H{16}">>
// CHECK: fopt.begin main %[[C]]
// CHECK: "FxpBinaryOp::AddFxp" %{{.*}} vrf

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @masked_sum(%tab: tensor<14x13xi32>, %lens: tensor<13xi32>, %o: tensor<14xi32>) -> tensor<14xi32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<14xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<14xi32>) -> tensor<14xi32>
    %r = linalg.generic {indexing_maps = [#id, #col, #row], iterator_types = ["parallel", "reduction"]} ins(%tab, %lens : tensor<14x13xi32>, tensor<13xi32>) outs(%f : tensor<14xi32>) {
    ^bb0(%t: i32, %l: i32, %acc: i32):
      %on = arith.cmpi ne, %t, %zero : i32
      %v = arith.select %on, %l, %zero : i32
      %s = arith.addi %acc, %v : i32
      linalg.yield %s : i32
    } -> tensor<14xi32>
    return %r : tensor<14xi32>
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @col_key(%x: tensor<8x16xf32>, %cols: tensor<16xi32>, %pick: tensor<8xi32>, %o: tensor<8xf32>) -> tensor<8xf32> attributes {furiosa.written = [3]} {
    %ninf = arith.constant 0xFF800000 : f32
    %neg = arith.constant -1.000000e+00 : f32
    %e = tensor.empty() : tensor<8xf32>
    %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
    %r = linalg.generic {indexing_maps = [#id, #col, #row, #row], iterator_types = ["parallel", "reduction"]} ins(%x, %cols, %pick : tensor<8x16xf32>, tensor<16xi32>, tensor<8xi32>) outs(%f : tensor<8xf32>) {
    ^bb0(%a: f32, %c: i32, %p: i32, %acc: f32):
      %was = arith.cmpi eq, %c, %p : i32
      %v = arith.select %was, %neg, %a : f32
      %m = arith.maximumf %acc, %v : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
