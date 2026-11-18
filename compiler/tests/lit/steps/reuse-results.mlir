// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A generic no class maps as it is, whose body recomputes what an earlier generic yields, reads that result instead
// (tl.sort's stages, each fused with every stage before it). @second (a remainder no class maps) iterates @first's
// loops swapped: its reads of %x and %q agree with @first's under that correspondence, so it reads @first's result
// (relaid out to its loop order), viewing the yielded bits as the f32 they were, and no longer reads %x or %q.

// CHECK-LABEL: func.func @chain
// CHECK: %[[FIRST:.*]] = call @chain__s0(%arg0, %arg1, %arg3)
// CHECK: %[[T:.*]] = call @chain__s1(%[[FIRST]], %{{.*}})
// CHECK: call @chain__s2(%arg2, %[[T]], %arg4)
// CHECK-LABEL: func.func private @chain__s1
// CHECK: linalg.transpose
// CHECK-LABEL: func.func private @chain__s2(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x4xf32>)
// CHECK-NOT: arith.mulf
// CHECK: %[[W:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT: arith.remf %[[W]], %{{.*}} : f32
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#swap = affine_map<(d0, d1) -> (d1, d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @chain(%x: tensor<4x2xf32>, %q: tensor<4xf32>, %r: tensor<2x4xf32>, %p: tensor<4x2xi32>, %o: tensor<2x4xf32>) -> (tensor<4x2xi32>, tensor<2x4xf32>) attributes {furiosa.written = [3, 4]} {
    %e = tensor.empty() : tensor<4x2xi32>
    %first = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %q : tensor<4x2xf32>, tensor<4xf32>) outs(%e : tensor<4x2xi32>) {
    ^bb0(%a: f32, %b: f32, %out: i32):
      %v = arith.mulf %a, %b : f32
      %bits = arith.bitcast %v : f32 to i32
      linalg.yield %bits : i32
    } -> tensor<4x2xi32>
    %f = tensor.empty() : tensor<2x4xf32>
    %second = linalg.generic {indexing_maps = [#swap, #col, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %q, %r : tensor<4x2xf32>, tensor<4xf32>, tensor<2x4xf32>) outs(%f : tensor<2x4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      %s = arith.remf %v, %c : f32
      linalg.yield %s : f32
    } -> tensor<2x4xf32>
    return %first, %second : tensor<4x2xi32>, tensor<2x4xf32>
  }
}
