// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A windowed contraction operand (a 3x3 convolution's input) is materialized tap-major: each tap's DMA
// writes one contiguous block of a [taps, rows, channels] HBM buffer (the NPU compiler's cost grows with the
// contiguous runs an HBM destination has). Taps pad 9 -> 12 so K is whole 32-element packets.

// CHECK-LABEL: func.func @conv
// CHECK: fopt.alloc_hbm : !fopt.hbm<f32, <"1">, <"[[S:S[0-9]+_]]TQ{12}, [[S]]D1_0{8} = 6, [[S]]D2_0{8} = 6, [[S]]CQ{8}">>
// CHECK: "fopt.for"() <{count = 3 : i64}>
// CHECK: "fopt.for"() <{count = 3 : i64}>

#x = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 + d3, d1 + d4, d5)>
#w = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d5, d3, d4)>
#y = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#i = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
module {
  func.func @conv(%arg0: tensor<1x6x6x8xf32>, %arg1: tensor<16x8x3x3xf32>, %arg2: tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<6x6x16xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<6x6x16xf32>) -> tensor<6x6x16xf32>
    %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x6x6x8xf32> to tensor<1x8x8x8xf32>
    %r = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<1x8x8x8xf32>, tensor<16x8x3x3xf32>) outs(%f : tensor<6x6x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %m = arith.mulf %in, %in_0 : f32
      %s = arith.addf %out, %m : f32
      linalg.yield %s : f32
    } -> tensor<6x6x16xf32>
    %o = tensor.empty() : tensor<1x6x6x16xf32>
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, #i], iterator_types = ["parallel", "parallel", "parallel"]} ins(%r : tensor<6x6x16xf32>) outs(%o : tensor<1x6x6x16xf32>) {
    ^bb0(%in: f32, %out_1: f32):
      linalg.yield %in : f32
    } -> tensor<1x6x6x16xf32>
    return %out : tensor<1x6x6x16xf32>
  }
}
