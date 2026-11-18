// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A reduction whose combiner the Vector Engine cannot reduce (here a product, as tl.cumprod raises) becomes a tree
// of elementwise ops: the terms over the whole space, the halves along the reduced dim combined until one element
// is left, then the init (here 2; an init of the combiner's neutral element, as xor's 0, is left out). Terms over
// an extent the halves do not divide are padded with the combiner's identity (and's -1) to a power of two.

// CHECK-LABEL: func.func @prod
// CHECK: %[[T:.*]] = linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x4xf32>)
// CHECK: %[[A:.*]] = tensor.extract_slice %[[T]][0, 0] [8, 2] [1, 1]
// CHECK: %[[B:.*]] = tensor.extract_slice %[[T]][0, 2] [8, 2] [1, 1]
// CHECK: %[[H:.*]] = linalg.generic {{.*}} ins(%[[A]], %[[B]] : tensor<8x2xf32>, tensor<8x2xf32>)
// CHECK: arith.mulf
// CHECK: tensor.extract_slice %[[H]][0, 1] [8, 1] [1, 1]
// CHECK: %[[L:.*]] = linalg.generic {{.*}} : tensor<8x1xf32>, tensor<8x1xf32>) outs(%{{.*}} : tensor<8x1xf32>)
// CHECK: linalg.generic {{.*}} ins(%[[L]], %{{.*}} : tensor<8x1xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>)
// CHECK-LABEL: func.func @xor
// CHECK: %[[X:.*]] = linalg.generic {{.*}} : tensor<8x1xi32>, tensor<8x1xi32>) outs(%{{.*}} : tensor<8x1xi32>)
// CHECK: linalg.generic {{.*}} ins(%[[X]] : tensor<8x1xi32>) outs(%{{.*}} : tensor<8xi32>)
// CHECK-NOT: "reduction"
// CHECK-LABEL: func.func @and3
// CHECK-DAG: %[[ONES:.*]] = arith.constant dense<-1> : tensor<8x4xi32>
// CHECK-DAG: %[[T3:.*]] = linalg.generic {{.*}} ins(%arg0 : tensor<8x3xi32>)
// CHECK: %[[P:.*]] = tensor.insert_slice %[[T3]] into %[[ONES]][0, 0] [8, 3] [1, 1]
// CHECK: tensor.extract_slice %[[P]][0, 0] [8, 2] [1, 1]
// CHECK: tensor.extract_slice %[[P]][0, 2] [8, 2] [1, 1]
// CHECK: arith.andi
// CHECK-NOT: "reduction"
#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @prod(%x: tensor<8x4xf32>, %o: tensor<8xf32>) -> tensor<8xf32> attributes {furiosa.written = [1]} {
    %one = arith.constant 2.0 : f32
    %e = tensor.empty() : tensor<8xf32>
    %f = linalg.fill ins(%one : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<8x4xf32>) outs(%f : tensor<8xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.mulf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
  func.func @xor(%x: tensor<8x4xi32>, %o: tensor<8xi32>) -> tensor<8xi32> attributes {furiosa.written = [1]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<8x4xi32>) outs(%f : tensor<8xi32>) {
    ^bb0(%a: i32, %acc: i32):
      %m = arith.xori %acc, %a : i32
      linalg.yield %m : i32
    } -> tensor<8xi32>
    return %r : tensor<8xi32>
  }
  func.func @and3(%x: tensor<8x3xi32>, %o: tensor<8xi32>) -> tensor<8xi32> attributes {furiosa.written = [1]} {
    %ones = arith.constant -1 : i32
    %e = tensor.empty() : tensor<8xi32>
    %f = linalg.fill ins(%ones : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<8x3xi32>) outs(%f : tensor<8xi32>) {
    ^bb0(%a: i32, %acc: i32):
      %m = arith.andi %acc, %a : i32
      linalg.yield %m : i32
    } -> tensor<8xi32>
    return %r : tensor<8xi32>
  }
}
