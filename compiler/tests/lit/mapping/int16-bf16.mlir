// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// int16 values live as i32 like int8. A bf16's bits as an i16 are those of the f32 it widens to, >> 16, and back
// << 16 (a bf16 is an f32's high half): tl.flip's bf16 xor swaps run in i32 lanes.

// CHECK-LABEL: func.func @swap
// CHECK-SAME: !fopt.hbm<i32
// CHECK: "FxpBinaryOp::ArithRightShift"
// CHECK: "LogicBinaryOpI32::BitXor"
// CHECK: "FxpBinaryOp::LeftShift"
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @swap(%x: tensor<64x64xbf16>, %k: tensor<64x64xi16>, %o: tensor<64x64xbf16>) -> tensor<64x64xbf16> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64x64xbf16>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %k : tensor<64x64xbf16>, tensor<64x64xi16>) outs(%e : tensor<64x64xbf16>) {
    ^bb0(%a: bf16, %b: i16, %out: bf16):
      %i = arith.bitcast %a : bf16 to i16
      %s = arith.xori %i, %b : i16
      %f = arith.bitcast %s : i16 to bf16
      linalg.yield %f : bf16
    } -> tensor<64x64xbf16>
    return %r : tensor<64x64xbf16>
  }
}
