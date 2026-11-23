// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A masked sum, sum over [i, k] of x * m with m a 0/1 table (Steps' hoisted index-only values, marked
// furiosa.table: a mask selectsToArithmetic made a factor), is no contraction: the Contraction Engine would round x
// to bf16, where Triton sums f32. The Vector Engine multiplies and reduces it in f32. (With m a buffer, the same
// kernel is a contraction of x's columns with m.)

// CHECK-LABEL: func.func @masked_sum
// CHECK-NOT: fopt.contract_outer
// CHECK: fopt.vector_intra_slice_reduce
#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#p = affine_map<(d0, d1, d2) -> (d0, d1)>
#pp = affine_map<(d0, d1) -> (d0, d1)>
#pr = affine_map<(d0, d1) -> (d0)>
module {
  func.func @masked_sum(%x: tensor<6x128x3xf32>, %o: tensor<6xf32>, %m: tensor<128x3xf32> {furiosa.table}) -> tensor<6xf32> attributes {furiosa.written = [1]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<6xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<6xf32>) -> tensor<6xf32>
    %e2 = tensor.empty() : tensor<6x128xf32>
    %f2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %p = linalg.generic {indexing_maps = [#x, #w, #p], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %m : tensor<6x128x3xf32>, tensor<128x3xf32>) outs(%f2 : tensor<6x128xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %v = arith.mulf %a, %b : f32
      %r = arith.addf %acc, %v : f32
      linalg.yield %r : f32
    } -> tensor<6x128xf32>
    %s = linalg.generic {indexing_maps = [#pp, #pr], iterator_types = ["parallel", "reduction"]} ins(%p : tensor<6x128xf32>) outs(%f : tensor<6xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %r = arith.addf %acc, %a : f32
      linalg.yield %r : f32
    } -> tensor<6xf32>
    return %s : tensor<6xf32>
  }
}
