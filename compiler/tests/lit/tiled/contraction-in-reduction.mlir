// RUN: env FURIOSA_TILED_DEBUG=1 not furiosa-mlir-opt --furiosa-linalg-to-fopt %s 2>&1 | FileCheck %s

// A contraction (QK^T) whose column an enclosing reduction reduces (the max over keys of a softmax in the
// same kernel) is refused: the contraction would need its rows in that reduction's time. The steps splitter
// then maps the two as separate steps.

// CHECK: tiled: a contraction whose output axis an enclosing reduction reduces

#q = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#k = affine_map<(d0, d1, d2, d3, d4) -> (d3, d1, d4)>
#s = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#n4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#r3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @qkmax(%q: tensor<1x8x2x128xbf16>, %k: tensor<64x8x128xbf16>, %mask: tensor<64xf32>, %o: tensor<1x8x2xf32>) -> tensor<1x8x2xf32> attributes {furiosa.written = [3]} {
    %cst = arith.constant 0.000000e+00 : f32
    %ninf = arith.constant 0xFF800000 : f32
    %scale = arith.constant 0.125 : f32
    %e = tensor.empty() : tensor<1x8x2x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1x8x2x64xf32>) -> tensor<1x8x2x64xf32>
    %r = linalg.generic {indexing_maps = [#q, #k, #s], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%q, %k : tensor<1x8x2x128xbf16>, tensor<64x8x128xbf16>) outs(%f : tensor<1x8x2x64xf32>) {
    ^bb0(%a: bf16, %b: bf16, %acc: f32):
      %x = arith.extf %a : bf16 to f32
      %y = arith.extf %b : bf16 to f32
      %m = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<1x8x2x64xf32>
    %e3 = tensor.empty() : tensor<1x8x2xf32>
    %f3 = linalg.fill ins(%ninf : f32) outs(%e3 : tensor<1x8x2xf32>) -> tensor<1x8x2xf32>
    %mx = linalg.generic {indexing_maps = [#a4, #n4, #r3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%r, %mask : tensor<1x8x2x64xf32>, tensor<64xf32>) outs(%f3 : tensor<1x8x2xf32>) {
    ^bb0(%v: f32, %mk: f32, %acc: f32):
      %t = arith.mulf %v, %scale : f32
      %u = arith.addf %t, %mk : f32
      %w = arith.maxnumf %acc, %u : f32
      linalg.yield %w : f32
    } -> tensor<1x8x2xf32>
    return %mx : tensor<1x8x2xf32>
  }
}
