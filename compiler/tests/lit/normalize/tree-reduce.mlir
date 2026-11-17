// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A reduction whose combiner the Vector Engine cannot reduce (here a product, as tl.cumprod raises) becomes a tree
// of elementwise ops: the terms over the whole space, the halves along the reduced dim combined until one element
// is left, then the init.

// CHECK-LABEL: func.func @prod
// CHECK: %[[T:.*]] = linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x4xf32>)
// CHECK: %[[A:.*]] = tensor.extract_slice %[[T]][0, 0] [8, 2] [1, 1]
// CHECK: %[[B:.*]] = tensor.extract_slice %[[T]][0, 2] [8, 2] [1, 1]
// CHECK: %[[H:.*]] = linalg.generic {{.*}} ins(%[[A]], %[[B]] : tensor<8x2xf32>, tensor<8x2xf32>)
// CHECK: arith.mulf
// CHECK: tensor.extract_slice %[[H]][0, 1] [8, 1] [1, 1]
// CHECK: %[[L:.*]] = linalg.generic {{.*}} : tensor<8x1xf32>, tensor<8x1xf32>) outs(%{{.*}} : tensor<8x1xf32>)
// CHECK: linalg.generic {{.*}} ins(%[[L]], %{{.*}} : tensor<8x1xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>)
// CHECK-NOT: "reduction"
#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @prod(%x: tensor<8x4xf32>, %o: tensor<8xf32>) -> tensor<8xf32> attributes {furiosa.written = [1]} {
    %one = arith.constant 1.0 : f32
    %e = tensor.empty() : tensor<8xf32>
    %f = linalg.fill ins(%one : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<8x4xf32>) outs(%f : tensor<8xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.mulf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
