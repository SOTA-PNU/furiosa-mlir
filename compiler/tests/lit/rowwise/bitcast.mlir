// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s --implicit-check-not=fopt.cast

// A bitcast between 32-bit types moves nothing: the f32 input streams on as bits into the i32 max (whose stream
// it rereads as i32), and the i32 result is committed as its bits, with no Cast Engine conversion.

// CHECK-LABEL: func.func @bits
// CHECK: fopt.to_vrf {{.*}} -> !fopt.vrf<i32
// CHECK: fopt.begin main {{.*}} : !fopt.dm<f32
// CHECK: fopt.vector_alu "vector_clip" "ClipBinaryOpI32::Max" {{.*}} vrf
// CHECK: fopt.commit {{.*}} -> !fopt.dm<i32
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @bits(%x: tensor<16x256xf32>, %k: tensor<16x256xi32>, %o: tensor<16x256xi32>) -> tensor<16x256xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<16x256xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %k : tensor<16x256xf32>, tensor<16x256xi32>) outs(%e : tensor<16x256xi32>) {
    ^bb0(%a: f32, %b: i32, %out: i32):
      %i = arith.bitcast %a : f32 to i32
      %y = arith.maxsi %b, %i : i32
      linalg.yield %y : i32
    } -> tensor<16x256xi32>
    return %r : tensor<16x256xi32>
  }
}
