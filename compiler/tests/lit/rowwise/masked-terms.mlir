// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// @masked_sum (segment offsets: lens[j] summed where a table selects it, over 13 columns padded to 16): a term of
// several inputs pads the one whose identity padding alone folds it to the identity: lens with 0, whatever the
// table holds past it (a select whose sides fold alike is that value).

// CHECK-LABEL: func.func @masked_sum
// CHECK: fopt.memset %{{.*}}, 0.000000e+00 : !fopt.dm<i32
// CHECK: fopt.vector_intra_slice_reduce "AddSat"

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
