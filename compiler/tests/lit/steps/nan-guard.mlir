// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=arith.maxnumf

// maxnumf (relu, tl.max) takes the other operand where one is NaN, as on the GPU; the Vector Engine's max
// propagates NaN. Before any kernel maps whole, a value cleans to x > -inf ? x : -inf (false exactly for NaN and
// -inf, which lose to anything) and the engine's maximumf takes over: relu = maximumf(clean(x), 0), and a max
// reduction's combiner maximumf(acc, clean(v)).

// CHECK-LABEL: func.func @relu
// CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[K:.*]] = arith.cmpf ogt, %[[X:.*]], %[[NINF]] : f32
// CHECK: %[[C:.*]] = arith.select %[[K]], %[[X]], %[[NINF]] : f32
// CHECK: arith.maximumf %[[C]], %{{.*}} : f32
// CHECK-LABEL: func.func @rowmax
// CHECK: %[[RK:.*]] = arith.cmpf ogt, %[[V:.*]], %{{.*}} : f32
// CHECK: %[[RC:.*]] = arith.select %[[RK]], %[[V]], %{{.*}} : f32
// CHECK: arith.maximumf %[[RC]], %{{.*}} : f32
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @relu(%x: tensor<64x256xf32>, %o: tensor<64x256xf32>) -> tensor<64x256xf32> attributes {furiosa.written = [1]} {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x256xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x256xf32>) outs(%e : tensor<64x256xf32>) {
    ^bb0(%a: f32, %out: f32):
      %m = arith.maxnumf %a, %zero : f32
      linalg.yield %m : f32
    } -> tensor<64x256xf32>
    return %r : tensor<64x256xf32>
  }
  func.func @rowmax(%x: tensor<64x256xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<64xf32>
    %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<64x256xf32>) outs(%f : tensor<64xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.maxnumf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
