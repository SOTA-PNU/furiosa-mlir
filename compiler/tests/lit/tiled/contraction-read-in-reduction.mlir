// RUN: env FURIOSA_TILED_DEBUG=1 not furiosa-mlir-opt --furiosa-linalg-to-fopt %s 2>&1 | FileCheck %s

// A contraction (s, the sum of x * w over [i, k]) that an enclosing reduction reads (a variance of x - s, as a
// batch norm recomputes its mean) is refused: that reduction's body runs over its own axes in time, which the
// contraction's stream does not have (furiosa-opt rejected the commit of its [columns] result into the tile of
// [i, k] x columns). No class maps the kernel whole; the steps splitter maps s as a step of its own.

// CHECK: tiled: a contraction an enclosing reduction reads

#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#c = affine_map<(d0, d1, d2) -> (d0)>
#p = affine_map<(d0, d1, d2) -> (d0, d1)>
#pp = affine_map<(d0, d1) -> (d0, d1)>
#pr = affine_map<(d0, d1) -> (d0)>
module {
  func.func @meanvar(%x: tensor<6x128x3xf32>, %w: tensor<128x3xf32>, %o: tensor<6xf32>) -> tensor<6xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<6xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<6xf32>) -> tensor<6xf32>
    %e2 = tensor.empty() : tensor<6x128xf32>
    %f2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %p = linalg.generic {indexing_maps = [#x, #w, #p], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<6x128x3xf32>, tensor<128x3xf32>) outs(%f2 : tensor<6x128xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %m = arith.mulf %a, %b : f32
      %r = arith.addf %acc, %m : f32
      linalg.yield %r : f32
    } -> tensor<6x128xf32>
    %s = linalg.generic {indexing_maps = [#pp, #pr], iterator_types = ["parallel", "reduction"]} ins(%p : tensor<6x128xf32>) outs(%f : tensor<6xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %r = arith.addf %acc, %a : f32
      linalg.yield %r : f32
    } -> tensor<6xf32>
    %q = linalg.generic {indexing_maps = [#x, #c, #p], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %s : tensor<6x128x3xf32>, tensor<6xf32>) outs(%f2 : tensor<6x128xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %d = arith.subf %a, %b : f32
      %d2 = arith.mulf %d, %d : f32
      %r = arith.addf %acc, %d2 : f32
      linalg.yield %r : f32
    } -> tensor<6x128xf32>
    %v = linalg.generic {indexing_maps = [#pp, #pr], iterator_types = ["parallel", "reduction"]} ins(%q : tensor<6x128xf32>) outs(%f : tensor<6xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %r = arith.addf %acc, %a : f32
      linalg.yield %r : f32
    } -> tensor<6xf32>
    return %v : tensor<6xf32>
  }
}
