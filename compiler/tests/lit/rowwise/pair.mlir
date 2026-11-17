// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// Two inputs of one dtype combined elementwise (swiglu's gate * up) stream interleaved through one pass (pair
// mode: begin_interleaved, then unzip into the two operands of a zip op) instead of filling the VRF with one.

// CHECK-LABEL: func.func @gate_up
// CHECK: fopt.begin_interleaved
// CHECK: fopt.vector_intra_slice_unzip
// CHECK: fopt.vector_fp_zip "MulF(FpMulAlu::Mul0)"
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gate_up(%g: tensor<16x256xbf16>, %u: tensor<16x256xbf16>, %o: tensor<16x256xbf16>) -> tensor<16x256xbf16> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<16x256xbf16>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%g, %u : tensor<16x256xbf16>, tensor<16x256xbf16>) outs(%e : tensor<16x256xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %x = arith.extf %a : bf16 to f32
      %y = arith.extf %b : bf16 to f32
      %z = arith.mulf %x, %y : f32
      %t = arith.truncf %z : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<16x256xbf16>
    return %r : tensor<16x256xbf16>
  }
}
