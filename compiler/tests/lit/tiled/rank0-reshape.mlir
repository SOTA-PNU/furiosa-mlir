// RUN: env FURIOSA_TILED_DEBUG=1 furiosa-mlir-opt --furiosa-linalg-to-fopt %s 2>&1 | FileCheck %s

// A row sum stored through a reshape of its rank-0 result (one row, as a split step of a 1-row reduction gets
// it): the expand_shape's unit dim, which no reassociation group covers, indexes 0. The analysis used to leave
// that index null and crash; now the program folds, and the class declines a scalar output cleanly. The row-wise
// class maps the kernel (one row's sum, stored through the reshape).

// CHECK: tiled: row_sum:
// CHECK: store arg1[0]
// CHECK: tiled: row_sum: a scalar output
// CHECK: func.func @row_sum(%arg0: !fopt.hbm<f32, <"1">, <"H{1024}">>

#x = affine_map<(d0) -> (0, d0)>
#s = affine_map<(d0) -> ()>
module {
  func.func @row_sum(%x: tensor<1x1024xf32>, %y: tensor<1xf32>) -> tensor<1xf32> attributes {furiosa.written = [1]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<f32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<f32>) -> tensor<f32>
    %r = linalg.generic {indexing_maps = [#x, #s], iterator_types = ["reduction"]} ins(%x : tensor<1x1024xf32>) outs(%f : tensor<f32>) {
    ^bb0(%in: f32, %acc: f32):
      %a = arith.addf %acc, %in : f32
      linalg.yield %a : f32
    } -> tensor<f32>
    %o = tensor.expand_shape %r [] output_shape [1] : tensor<f32> into tensor<1xf32>
    return %o : tensor<1xf32>
  }
}
