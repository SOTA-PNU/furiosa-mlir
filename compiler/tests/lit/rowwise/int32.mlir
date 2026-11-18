// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A float->int cast of an integral value converts exactly (@quantize: an int8 quantizer's clamped floor, whose
// expansion passes x through where |x| > 2^23, an integer there).
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
// CHECK-LABEL: func.func @quantize
// CHECK: fopt.vector_fp_to_fxp {{.*}}, 31
// CHECK: fopt.vector_fxp_to_fp {{.*}}, 31
// CHECK: fopt.vector_fp_to_fxp {{.*}}, 31
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
  func.func @quantize(%x: tensor<64x64xf32>, %o: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {furiosa.written = [1]} {
    %big = arith.constant 0x4B000000 : f32
    %nbig = arith.constant 0xCB000000 : f32
    %one = arith.constant 1.000000e+00 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %lo = arith.constant -1.270000e+02 : f32
    %hi = arith.constant 1.270000e+02 : f32
    %e = tensor.empty() : tensor<64x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x64xf32>) outs(%e : tensor<64x64xi32>) {
    ^bb0(%a: f32, %out: i32):
      %t = math.round %a : f32
      %past = arith.cmpf ogt, %t, %a : f32
      %step = arith.select %past, %one, %zero : f32
      %near = arith.subf %t, %step : f32
      %low = arith.cmpf olt, %a, %nbig : f32
      %high = arith.cmpf ogt, %a, %big : f32
      %f0 = arith.select %low, %a, %near : f32
      %f = arith.select %high, %a, %f0 : f32
      %m = arith.maximumf %f, %lo : f32
      %n = arith.minimumf %m, %hi : f32
      %i = arith.fptosi %n : f32 to i8
      %w = arith.extsi %i : i8 to i32
      linalg.yield %w : i32
    } -> tensor<64x64xi32>
    return %r : tensor<64x64xi32>
  }
}
