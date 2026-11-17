// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// i32 kernels in the row-wise class (and round, which converts to i32 and back): the stream carries an i32's bits and each i32 op rereads it as i32. Two i32
// inputs stream interleaved into the fixed-point ALU; a select on an i32 comparison tags the key read as i32 and
// selects with i32 logic ops; an i32 sum reduces with saturating adds.

// CHECK-LABEL: func.func @add
// CHECK: fopt.vector_reinterpret {{.*}} -> !fopt.tu<i32
// CHECK: fopt.vector_alu "vector_fxp_zip" "FxpBinaryOp::AddFxp"
// CHECK-LABEL: func.func @where
// CHECK: fopt.vector_tag_cmp ["Greater(0i32)", "False", "False", "False"] {{.*}} -> !fopt.tu<i32
// CHECK: fopt.vector_alu "vector_logic" "LogicBinaryOpI32::BitAnd"
// CHECK-LABEL: func.func @round
// CHECK: fopt.vector_fp_to_fxp {{.*}}, 31
// CHECK: fopt.vector_fxp_to_fp {{.*}}, 31
// CHECK-LABEL: func.func @sum
// CHECK: fopt.vector_intra_slice_reduce "AddSat" {{.*}} -> !fopt.tu<i32
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @add(%x: tensor<64x64xi32>, %y: tensor<64x64xi32>, %o: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xi32>, tensor<64x64xi32>) outs(%e : tensor<64x64xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<64x64xi32>
    return %r : tensor<64x64xi32>
  }
  func.func @where(%x: tensor<64x64xi32>, %y: tensor<64x64xi32>, %o: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xi32>, tensor<64x64xi32>) outs(%e : tensor<64x64xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %gt = arith.cmpi sgt, %a, %b : i32
      %s = arith.select %gt, %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<64x64xi32>
    return %r : tensor<64x64xi32>
  }
  func.func @round(%x: tensor<64x64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %out: f32):
      %s = math.round %a : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
  func.func @sum(%x: tensor<64x64xi32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<64xi32>
    %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<64xi32>) -> tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<64x64xi32>) outs(%f : tensor<64xi32>) {
    ^bb0(%a: i32, %out: i32):
      %s = arith.addi %out, %a : i32
      linalg.yield %s : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
