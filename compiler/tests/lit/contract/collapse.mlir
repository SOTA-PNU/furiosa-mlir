// RUN: furiosa-mlir-opt --furiosa-normalize-contractions %s | FileCheck %s

// A contraction becomes y[m, n] = sum_k x[m, k] w[n, k] over collapsed operands, the form the contraction
// classes map. In P @ V the second operand V is laid out [k, n], so it is transposed (linalg.transpose, which
// the copy class maps as one DMA). The operand whose dims lead the output is x, whatever its position.

// CHECK-LABEL: func.func @pv
// CHECK: %[[VT:.*]] = linalg.transpose ins(%{{.*}} : tensor<64x128xbf16>) outs(%{{.*}} : tensor<128x64xbf16>) permutation = [1, 0]
// CHECK: linalg.generic {indexing_maps = [#[[X:map[0-9]*]], #[[W:map[0-9]*]], #[[Y:map[0-9]*]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}}, %[[VT]] : tensor<2x64xbf16>, tensor<128x64xbf16>)

// A square (x * x, a sum of squares) is a reduction of one operand, not a contraction: it stays as it is.

// CHECK-LABEL: func.func @squares
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "reduction"]}
#v = affine_map<(d0, d1, d2) -> (d2, d1)>
#p = affine_map<(d0, d1, d2) -> (d0, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#x = affine_map<(d0, d1) -> (d0, d1)>
#r = affine_map<(d0, d1) -> (d0)>
module {
  func.func @pv(%vv: tensor<64x128xbf16>, %pp: tensor<2x64xbf16>, %out: tensor<2x128xf32>) -> tensor<2x128xf32> attributes {furiosa.written = [2]} {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<2x128xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x128xf32>) -> tensor<2x128xf32>
    %r = linalg.generic {indexing_maps = [#v, #p, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%vv, %pp : tensor<64x128xbf16>, tensor<2x64xbf16>) outs(%f : tensor<2x128xf32>) {
    ^bb0(%a: bf16, %b: bf16, %acc: f32):
      %x = arith.extf %b : bf16 to f32
      %y = arith.extf %a : bf16 to f32
      %m = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<2x128xf32>
    return %r : tensor<2x128xf32>
  }
  func.func @squares(%xx: tensor<4x64xf32>, %out: tensor<4xf32>) -> tensor<4xf32> attributes {furiosa.written = [1]} {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<4xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<4xf32>) -> tensor<4xf32>
    %r = linalg.generic {indexing_maps = [#x, #r], iterator_types = ["parallel", "reduction"]} ins(%xx : tensor<4x64xf32>) outs(%f : tensor<4xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.mulf %a, %a : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
