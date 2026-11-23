// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma %s | FileCheck %s

// A contraction operand that is a K window of a longer dim (attention's P x V reads V's rows of one key block):
// the window's dim splits into whole padded K blocks, an axis-less digit the window's offset selects (block 1
// of 4), and the padded K digit itself. A window of a dim the K blocks do not divide (32 keys of 100) reads a copy
// of the buffer padded to whole blocks (128 rows), block 1 of 4.

// CHECK-LABEL: func.func @pv
// CHECK: %[[ST:.*]] = fopt.alloc_hbm : !fopt.hbm<f32, <"1">, <"S0_D1_0{64}, S0_CQ{64}">>
// CHECK: fopt.view_reshape {{.*}} -> !fopt.hbmv<f32, <"1">, <"S0_D0_0{4}, S0_CQ{64}, S0_D1_0{64}">, false>
// CHECK: %[[T:.*]] = fopt.tile {{.*}} index <"S0_D0_0{4}"> len 1 start 1
// CHECK: %[[D:.*]] = fopt.view %[[ST]]
// CHECK: fopt.hbm_to_hbm_view %[[T]], %[[D]] : !fopt.hbmv<f32, <"1">, <"1 # 4, S0_CQ{64}, S0_D1_0{64}">, false>
// CHECK: fopt.contract_outer

// CHECK-LABEL: func.func @pv_pad
// CHECK: %[[PAD:.*]] = fopt.alloc_hbm : !fopt.hbm<f32, <"1">, <"S0_P0{128}, S0_P1{64}">>
// CHECK: fopt.tile {{.*}} index <"S0_P0{128}"> len 100 start 0
// CHECK: %[[PV:.*]] = fopt.view %[[PAD]]
// CHECK: fopt.view_reshape %[[PV]] {{.*}} -> !fopt.hbmv<f32, <"1">, <"S0_D0_0{4}, S0_CQ{32}, S0_D1_0{64}">, false>
// CHECK: fopt.tile {{.*}} index <"S0_D0_0{4}"> len 1 start 1
// CHECK: fopt.contract_outer
#v = affine_map<(d0, d1, d2) -> (d2, d1)>
#p = affine_map<(d0, d1, d2) -> (d0, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @pv(%v: tensor<256x64xf16>, %p: tensor<64x64xf16>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
    %vs = tensor.extract_slice %v[64, 0] [64, 64] [1, 1] : tensor<256x64xf16> to tensor<64x64xf16>
    %r = linalg.generic {indexing_maps = [#v, #p, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%vs, %p : tensor<64x64xf16>, tensor<64x64xf16>) outs(%f : tensor<64x64xf32>) {
    ^bb0(%a: f16, %b: f16, %acc: f32):
      %x = arith.extf %b : f16 to f32
      %y = arith.extf %a : f16 to f32
      %m = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
  func.func @pv_pad(%v: tensor<100x64xf16>, %p: tensor<64x32xf16>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
    %vs = tensor.extract_slice %v[32, 0] [32, 64] [1, 1] : tensor<100x64xf16> to tensor<32x64xf16>
    %r = linalg.generic {indexing_maps = [#v, #p, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%vs, %p : tensor<32x64xf16>, tensor<64x32xf16>) outs(%f : tensor<64x64xf32>) {
    ^bb0(%a: f16, %b: f16, %acc: f32):
      %x = arith.extf %b : f16 to f32
      %y = arith.extf %a : f16 to f32
      %m = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
