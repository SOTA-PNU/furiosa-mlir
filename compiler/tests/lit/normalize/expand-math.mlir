// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=math.fma --implicit-check-not=math.log2

// Math ops the Vector Engine lacks become ones it has: fma(a, b, c) = a*b + c, log2(x) = ln(x) * log2(e).

// CHECK-LABEL: func.func @fma_log2
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[M:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: %[[S:.*]] = arith.addf %[[M]], %[[C]] : f32
// CHECK-DAG: %[[E:.*]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[L:.*]] = math.log %[[S]] : f32
// CHECK: arith.mulf %[[L]], %[[E]] : f32
#id = affine_map<(d0) -> (d0)>
module {
  func.func @fma_log2(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [3]} {
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b, %c : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32, %out: f32):
      %f = math.fma %x, %y, %z : f32
      %l = math.log2 %f : f32
      linalg.yield %l : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
