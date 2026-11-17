// RUN: furiosa-mlir-opt --furiosa-hoist-index-only %s | FileCheck %s --implicit-check-not=math.cos

// A body value that depends only on the iteration indices (a rotary angle's cosine, cos(d * 0.5)) is evaluated
// at compile time over the dims it reads and becomes a constant tensor input: the Vector Engine never sees an
// angle it would have to range-reduce.

// CHECK-LABEL: func.func @rope
// CHECK: %[[T:.*]] = arith.constant dense<[1.000000e+00, 0.87758255, 0.540302277, 7.073720e-02]> : tensor<4xf32>
// CHECK: linalg.generic {{.*}} ins(%arg0, %[[T]] : tensor<4xf32>, tensor<4xf32>)
// CHECK: arith.mulf
#id = affine_map<(d0) -> (d0)>
module {
  func.func @rope(%x: tensor<4xf32>, %o: tensor<4xf32>) -> tensor<4xf32> attributes {furiosa.written = [1]} {
    %half = arith.constant 5.000000e-01 : f32
    %e = tensor.empty() : tensor<4xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%x : tensor<4xf32>) outs(%e : tensor<4xf32>) {
    ^bb0(%a: f32, %out: f32):
      %d = linalg.index 0 : index
      %di = arith.index_cast %d : index to i32
      %df = arith.sitofp %di : i32 to f32
      %ang = arith.mulf %df, %half : f32
      %c = math.cos %ang : f32
      %y = arith.mulf %a, %c : f32
      linalg.yield %y : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
