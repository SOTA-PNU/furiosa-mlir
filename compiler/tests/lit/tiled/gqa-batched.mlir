// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// The tiled class maps a GQA attention contraction (QK^T) whole: kvh is a batch axis that both operands
// index, laid out over slices; the TRF operand's column is the two query heads of a kv head. bf16
// contraction operands keep K innermost, so two columns per slice need no 8-byte column rows.

// CHECK-LABEL: func.func @qk
// CHECK: fopt.to_trf {{.*}} -> !fopt.trf<bf16, <"1">, <"1 # 2">, <"A0{8}, A2{64} / 2">, <"A1{2} # 8">, <"KF{128}">>
// CHECK: fopt.contract_outer
#q = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#k = affine_map<(d0, d1, d2, d3, d4) -> (d3, d1, d4)>
#s = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @qk(%q: tensor<1x8x2x128xbf16>, %k: tensor<64x8x128xbf16>, %o: tensor<1x8x2x64xf32>) -> tensor<1x8x2x64xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
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
    return %r : tensor<1x8x2x64xf32>
  }
}
