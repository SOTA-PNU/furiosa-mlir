// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// npu3's Tensor Unit reads at most 5 VRF operands in one pass ("cannot load more than 5"; the CPU emulation takes
// more): (select(x > 0, a, b) * c + d) * e + f over columns a..f reads two for the select, then c, d and e; f, which
// the clip stage could still take, starts a pass of its own.

// CHECK-LABEL: func.func @six
// CHECK: "LogicBinaryOpF32::BitOr" %{{.*}} vrf
// CHECK: "LogicBinaryOpF32::BitXor" %{{.*}} vrf
// CHECK: fopt.vector_fp_binary "MulF(FpMulAlu::Mul0)" %{{.*}} vrf
// CHECK: fopt.vector_fp_binary "AddF" %{{.*}} vrf
// CHECK: fopt.vector_fp_binary "MulF(FpMulAlu::Mul1)" %{{.*}} vrf
// CHECK-NOT: vrf %
// CHECK: fopt.commit
// CHECK: fopt.begin main
// CHECK: fopt.vector_fp_binary "AddF" %{{.*}} vrf

#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @six(%x: tensor<64x64xf32>, %a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<64xf32>, %d: tensor<64xf32>, %e: tensor<64xf32>, %f: tensor<64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [7]} {
    %zero = arith.constant 0.0 : f32
    %t = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#id, #col, #col, #col, #col, #col, #col, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %a, %b, %c, %d, %e, %f : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%t : tensor<64x64xf32>) {
    ^bb0(%vx: f32, %va: f32, %vb: f32, %vc: f32, %vd: f32, %ve: f32, %vf: f32, %out: f32):
      %p = arith.cmpf ogt, %vx, %zero : f32
      %s = arith.select %p, %va, %vb : f32
      %m = arith.mulf %s, %vc : f32
      %n = arith.addf %m, %vd : f32
      %q = arith.mulf %n, %ve : f32
      %y = arith.addf %q, %vf : f32
      linalg.yield %y : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
