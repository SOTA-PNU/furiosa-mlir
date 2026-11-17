// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// One row-wise kernel stores several outputs of one DAG: a residual add h + r (a full row, here in pair mode)
// and the RMS norm of that sum times a weight, whose row reduction and normalization passes follow in the same
// kernel. Both outputs come back from the one kernel.

// CHECK-LABEL: func.func @add_norm
// CHECK-SAME: -> (!fopt.hbm<bf16, <"1">, <"R{16}, H{256}">> {fopt.arg = 3 : i64{{.*}}}, !fopt.hbm<bf16, <"1">, <"R{16}, H{256}">> {fopt.arg = 4 : i64
// CHECK: fopt.vector_intra_slice_reduce "Add"
// CHECK: return %{{.*}}, %{{.*}} :
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @add_norm(%h: tensor<16x256xbf16>, %r: tensor<16x256xbf16>, %w: tensor<256xbf16>, %oh: tensor<16x256xbf16>, %oy: tensor<16x256xbf16>) -> (tensor<16x256xbf16>, tensor<16x256xbf16>) attributes {furiosa.written = [3, 4]} {
    %zero = arith.constant 0.000000e+00 : f32
    %inv = arith.constant 3.906250e-03 : f32
    %eps = arith.constant 9.99999997E-7 : f32
    %e = tensor.empty() : tensor<16x256xbf16>
    %er = tensor.empty() : tensor<16xf32>
    %fr = linalg.fill ins(%zero : f32) outs(%er : tensor<16xf32>) -> tensor<16xf32>
    %s = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%h, %r : tensor<16x256xbf16>, tensor<16x256xbf16>) outs(%e : tensor<16x256xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %x = arith.extf %a : bf16 to f32
      %y = arith.extf %b : bf16 to f32
      %z = arith.addf %x, %y : f32
      %t = arith.truncf %z : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<16x256xbf16>
    %ss = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%s : tensor<16x256xbf16>) outs(%fr : tensor<16xf32>) {
    ^bb0(%a: bf16, %acc: f32):
      %x = arith.extf %a : bf16 to f32
      %q = arith.mulf %x, %x : f32
      %n = arith.addf %acc, %q : f32
      linalg.yield %n : f32
    } -> tensor<16xf32>
    %y = linalg.generic {indexing_maps = [#id, #row, #col, #id], iterator_types = ["parallel", "parallel"]} ins(%s, %ss, %w : tensor<16x256xbf16>, tensor<16xf32>, tensor<256xbf16>) outs(%e : tensor<16x256xbf16>) {
    ^bb0(%a: bf16, %q: f32, %g: bf16, %out: bf16):
      %x = arith.extf %a : bf16 to f32
      %m = arith.mulf %q, %inv : f32
      %v = arith.addf %m, %eps : f32
      %rs = math.rsqrt %v : f32
      %n = arith.mulf %x, %rs : f32
      %gw = arith.extf %g : bf16 to f32
      %o = arith.mulf %n, %gw : f32
      %t = arith.truncf %o : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<16x256xbf16>
    return %s, %y : tensor<16x256xbf16>, tensor<16x256xbf16>
  }
}
