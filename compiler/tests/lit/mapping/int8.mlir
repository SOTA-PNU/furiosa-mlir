// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// int8 buffers live as i32 on the device: the kernel's i8 tensors become i32 holding the same values, and an add
// that can leave the int8 range wraps back into it with (v << 24) >> 24.

// CHECK-LABEL: func.func @add
// CHECK-SAME: !fopt.hbm<i32
// CHECK: fopt.vector_alu "vector_fxp_zip" "FxpBinaryOp::AddFxp"
// CHECK: fopt.vector_alu "vector_fxp" "FxpBinaryOp::LeftShift"
// CHECK: fopt.vector_alu "vector_fxp" "FxpBinaryOp::ArithRightShift"
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add(%x: tensor<64x64xi8>, %y: tensor<64x64xi8>, %o: tensor<64x64xi8>) -> tensor<64x64xi8> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64x64xi8>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xi8>, tensor<64x64xi8>) outs(%e : tensor<64x64xi8>) {
    ^bb0(%a: i8, %b: i8, %out: i8):
      %s = arith.addi %a, %b : i8
      linalg.yield %s : i8
    } -> tensor<64x64xi8>
    return %r : tensor<64x64xi8>
  }
}
