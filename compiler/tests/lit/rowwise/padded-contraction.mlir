// RUN: not furiosa-mlir-opt --furiosa-linalg-to-fopt %s 2>&1 | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --check-prefix=STEPS

// Rows the slices hold only padded or windowed along an inner dim (here [5, 300]) are no reason for the row-wise
// class to take contractions (x[r, k] * w[c, k]) from the Tensor Unit: two of them gated together map whole in no
// class, and split into a step per contraction, each the tiled class's.

// CHECK: error: row-wise: a contraction over rows the slices hold only padded (the tiled class's)

// STEPS-LABEL: func.func @gated
// STEPS: call @gated__s0
// STEPS: call @gated__s1
#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gated(%x: tensor<5x16xf32>, %w: tensor<300x16xf32>, %v: tensor<300x16xf32>, %o: tensor<5x300xf32>) -> tensor<5x300xf32> attributes {furiosa.written = [3]} {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<5x300xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<5x300xf32>) -> tensor<5x300xf32>
    %a = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<5x16xf32>, tensor<300x16xf32>) outs(%f : tensor<5x300xf32>) {
    ^bb0(%p: f32, %q: f32, %acc: f32):
      %m = arith.mulf %p, %q : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<5x300xf32>
    %b = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %v : tensor<5x16xf32>, tensor<300x16xf32>) outs(%f : tensor<5x300xf32>) {
    ^bb0(%p: f32, %q: f32, %acc: f32):
      %m = arith.mulf %p, %q : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<5x300xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<5x300xf32>, tensor<5x300xf32>) outs(%e : tensor<5x300xf32>) {
    ^bb0(%p: f32, %q: f32, %out: f32):
      %m = arith.mulf %p, %q : f32
      linalg.yield %m : f32
    } -> tensor<5x300xf32>
    return %r : tensor<5x300xf32>
  }
}
