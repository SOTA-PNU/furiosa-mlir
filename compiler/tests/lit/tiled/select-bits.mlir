// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma --furiosa-ve-fuse %s | FileCheck %s --check-prefix=FUSED

// A select between two tiles beyond the 8 KB VRF (acc > -inf ? acc : exp(a), as NaN guards emit around running
// maxima) selects on the bits, exact for any NaN or infinity in the branch not taken: the mask m = c ? ~0 : 0 is
// two logic ops on the key ((x | m) & m), and b ^ ((a ^ b) & m) zips the tiles op by op.

// CHECK-LABEL: func.func @big
// CHECK: %[[B:[0-9]+]] = fopt.ve "vector_fp_unary" "FpUnaryOp::Exp"
// CHECK: %[[M0:[0-9]+]] = fopt.ve "vector_logic" "LogicBinaryOpF32::BitOr" %[[X:[0-9]+]] {branches = {{\[\["1---", 0xFFFFFFFF : f32\], \["all", 0.000000e\+00 : f32\]\]}}, tags = ["Greater(f32::NEG_INFINITY)"
// CHECK-NEXT: %[[M:[0-9]+]] = fopt.ve "vector_logic" "LogicBinaryOpF32::BitAnd" %[[M0]] {branches = {{\[\["1---", 0xFFFFFFFF : f32\], \["all", 0.000000e\+00 : f32\]\]}}}
// CHECK-NEXT: %[[D:[0-9]+]] = fopt.ve "vector_logic_zip" "LogicBinaryOpF32::BitXor" %[[X]] zip %[[B]]
// CHECK-NEXT: %[[E:[0-9]+]] = fopt.ve "vector_logic_zip" "LogicBinaryOpF32::BitAnd" %[[D]] zip %[[M]]
// CHECK-NEXT: fopt.ve "vector_logic_zip" "LogicBinaryOpF32::BitXor" %[[E]] zip %[[B]]

// FUSED: fopt.vector_tag_cmp ["Greater(f32::NEG_INFINITY)"
// FUSED-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitOr"
// FUSED-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitAnd"
// FUSED-COUNT-3: fopt.vector_alu "vector_logic_zip"
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [512, 64]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 4096]}, {\22arg\22: 2, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [512, 4096]}, {\22arg\22: 3, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [512, 4096]}]}"} {
  func.func @big(%arg0: tensor<512x64xf32>, %arg1: tensor<64x4096xf32>, %arg2: tensor<512x4096xf32>, %arg3: tensor<512x4096xf32>) -> tensor<512x4096xf32> attributes {furiosa.written = [3]} {
    %ninf = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<512x4096xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<512x4096xf32>) -> tensor<512x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<512x64xf32>, tensor<64x4096xf32>) outs(%2 : tensor<512x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.mulf %in, %in_1 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<512x4096xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %arg2 : tensor<512x4096xf32>, tensor<512x4096xf32>) outs(%1 : tensor<512x4096xf32>) {
    ^bb0(%in: f32, %a: f32, %out: f32):
      %5 = arith.cmpf ogt, %in, %ninf : f32
      %6 = math.exp %a : f32
      %7 = arith.select %5, %in, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<512x4096xf32>
    return %4 : tensor<512x4096xf32>
  }
}
