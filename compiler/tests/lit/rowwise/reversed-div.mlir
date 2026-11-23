// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A division of a constant by the streamed value (a layer norm's rstd = 1 / sqrt(var + eps), a row value computed
// once per row) takes the divider with its operands swapped (BinaryArgMode::Mode10).

// CHECK-LABEL: func.func @rstd
// CHECK: fopt.vector_fp_unary "Sqrt"
// CHECK-NEXT: fopt.vector_alu "vector_fp_div" "" {{.*}} {imm = 1.000000e+00 : f32, mode = "Mode10"}
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @rstd(%x: tensor<16x256xf32>, %v: tensor<16xf32>, %o: tensor<16x256xf32>) -> tensor<16x256xf32> attributes {furiosa.written = [2]} {
    %one = arith.constant 1.0 : f32
    %e = tensor.empty() : tensor<16x256xf32>
    %r = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %v : tensor<16x256xf32>, tensor<16xf32>) outs(%e : tensor<16x256xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %s = math.sqrt %b : f32
      %q = arith.divf %one, %s : f32
      %y = arith.mulf %a, %q : f32
      linalg.yield %y : f32
    } -> tensor<16x256xf32>
    return %r : tensor<16x256xf32>
  }
}
