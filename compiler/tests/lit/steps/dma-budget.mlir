// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A product over 512 columns raises to a halving tree (the Vector Engine cannot reduce products), which the tiled
// class would map as one kernel staging every leaf (over 500 DMA statements: furiosa-opt's compile time grows
// superlinearly in them). Beyond the step budget of 256 the kernel splits into steps instead.

// CHECK-LABEL: func.func @prod
// CHECK: call @prod__s0
// CHECK: call @prod__s1
#x = affine_map<(d0, d1) -> (d0, d1)>
#r = affine_map<(d0, d1) -> (d0)>
module {
  func.func @prod(%x: tensor<16x512xf32>, %o: tensor<16xf32>) -> tensor<16xf32> attributes {furiosa.written = [1]} {
    %one = arith.constant 1.0 : f32
    %e = tensor.empty() : tensor<16xf32>
    %f = linalg.fill ins(%one : f32) outs(%e : tensor<16xf32>) -> tensor<16xf32>
    %p = linalg.generic {indexing_maps = [#x, #r], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<16x512xf32>) outs(%f : tensor<16xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.mulf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<16xf32>
    return %p : tensor<16xf32>
  }
}
