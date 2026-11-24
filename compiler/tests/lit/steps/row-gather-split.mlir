// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not='tensor<64x2xf32>'

// An embedding read inside a sum (h[t] = tok[ids[t]] + pos[t]) moves into a generic of its own reading whole rows of
// the table at the data-dependent row ids[t]. Tried as a step of its own (with the table it reads from outside its
// body as a buffer, and its empty init kept), the gather class maps it: one row gather, no element gathers of the
// table seen as [64, 2]. The sum is the next step.

// CHECK-LABEL: func.func private @embed__s0(%arg0: tensor<128xf32>, %arg1: tensor<4xi32>
// CHECK: tensor.extract %arg0[%{{.*}}] : tensor<128xf32>
// CHECK-LABEL: func.func private @embed__s1
// CHECK: arith.addf
#row = affine_map<(d0, d1) -> (d0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @embed(%ids: tensor<4xi32>, %tok: tensor<16x8xf32>, %pos: tensor<4x8xf32>, %h: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {furiosa.written = [3]} {
    %c8 = arith.constant 8 : i32
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    %flat = tensor.collapse_shape %tok [[0, 1]] : tensor<16x8xf32> into tensor<128xf32>
    %e = tensor.empty() : tensor<4x8xf32>
    %r = linalg.generic {indexing_maps = [#row, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%ids, %pos : tensor<4xi32>, tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
    ^bb0(%i: i32, %p: f32, %out: f32):
      %col = linalg.index 1 : index
      %rs = arith.muli %i, %c8 : i32
      %r0 = arith.index_cast %rs : i32 to index
      %a = arith.addi %col, %r0 : index
      %lo = arith.maxsi %a, %c0 : index
      %hi = arith.minsi %lo, %c127 : index
      %x = tensor.extract %flat[%hi] : tensor<128xf32>
      %s = arith.addf %x, %p : f32
      linalg.yield %s : f32
    } -> tensor<4x8xf32>
    return %r : tensor<4x8xf32>
  }
}
