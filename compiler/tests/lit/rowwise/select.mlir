// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// Row-wise selects. select(x > y, x, y) keys on x - y > 0, which is exact: IEEE subtraction keeps the sign,
// and equal infinities give NaN, false either way. The value is replaced bit-exactly: clear the key, OR in the
// true value where the tag holds, XOR in the false value where it does not.

// CHECK-LABEL: func.func @strict
// CHECK: "SubF"
// CHECK: fopt.vector_tag_cmp ["Greater(0.0f32)", "False", "False", "False"]
// CHECK-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitAnd"
// CHECK-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitOr"
// CHECK-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitXor"

// select(x == top, i, N) under an i32 min: the equality key (x + 0) ^ (top + 0) leaves its pass as i32 and is
// tagged as i32, since the tag unit flushes subnormal f32s (npu3); the min reduces the i32 bits.

// CHECK-LABEL: func.func @first
// CHECK: fopt.vector_tag_cmp ["Less(f32::INFINITY)", "Equal(f32::INFINITY)", "False", "False"]
// CHECK: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitXor"
// CHECK: fopt.vector_reinterpret {{.*}} -> !fopt.tu<i32
// CHECK: fopt.vector_tag_cmp ["Equal(0i32)", "False", "False", "False"]
// CHECK: fopt.vector_intra_slice_reduce "Min" {{.*}} !fopt.tu<i32
// A key per row (a table along d0) selects bitwise: masks of all ones where it passes or fails, computed along
// the rows alone, then a & m (where(key, a, 0)) and (a & m) | (b & ~m) on the whole tile, the row mask from the VRF.

// CHECK-LABEL: func.func @per_row
// CHECK: fopt.vector_tag_cmp ["Equal(0i32)", "False", "False", "False"]
// CHECK: fopt.vector_tag_cmp ["Equal(0i32)", "False", "False", "False"]
// CHECK: "LogicBinaryOpI32::BitAnd" %{{[0-9]+}} vrf
// CHECK: "LogicBinaryOpI32::BitOr" %{{[0-9]+}} vrf
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#scalar = affine_map<(d0, d1) -> ()>
module {
  func.func @strict(%x: tensor<16x256xf32>, %y: tensor<16x256xf32>, %o: tensor<16x256xf32>) -> tensor<16x256xf32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<16x256xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<16x256xf32>, tensor<16x256xf32>) outs(%e : tensor<16x256xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %c = arith.cmpf ogt, %a, %b : f32
      %s = arith.select %c, %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<16x256xf32>
    return %r : tensor<16x256xf32>
  }
  func.func @first(%x: tensor<16x256xf32>, %top: tensor<f32>, %i: tensor<16x256xi32>, %o: tensor<16xi32>) -> tensor<16xi32> attributes {furiosa.written = [3]} {
    %n = arith.constant 151936 : i32
    %max = arith.constant 2147483647 : i32
    %e = tensor.empty() : tensor<16xi32>
    %f = linalg.fill ins(%max : i32) outs(%e : tensor<16xi32>) -> tensor<16xi32>
    %r = linalg.generic {indexing_maps = [#id, #scalar, #id, #row], iterator_types = ["parallel", "reduction"]} ins(%x, %top, %i : tensor<16x256xf32>, tensor<f32>, tensor<16x256xi32>) outs(%f : tensor<16xi32>) {
    ^bb0(%a: f32, %t: f32, %k: i32, %acc: i32):
      %c = arith.cmpf oeq, %a, %t : f32
      %s = arith.select %c, %k, %n : i32
      %m = arith.minsi %acc, %s : i32
      linalg.yield %m : i32
    } -> tensor<16xi32>
    return %r : tensor<16xi32>
  }
  func.func @per_row(%x: tensor<16x256xf32>, %k: tensor<16xi32>, %y: tensor<16x256xf32>, %o: tensor<16x256xf32>) -> tensor<16x256xf32> attributes {furiosa.written = [3]} {
    %c0 = arith.constant 0 : i32
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x256xf32>
    %r = linalg.generic {indexing_maps = [#id, #row, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %k, %y : tensor<16x256xf32>, tensor<16xi32>, tensor<16x256xf32>) outs(%e : tensor<16x256xf32>) {
    ^bb0(%a: f32, %key: i32, %b: f32, %out: f32):
      %c = arith.cmpi ne, %key, %c0 : i32
      %s = arith.select %c, %a, %zero : f32
      %t = arith.select %c, %a, %b : f32
      %u = arith.addf %s, %t : f32
      linalg.yield %u : f32
    } -> tensor<16x256xf32>
    return %r : tensor<16x256xf32>
  }
}
