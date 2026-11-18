// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s

// A select in a contraction's epilogue (the tutorial's leaky ReLU, acc >= 0 ? acc : 0.01 * acc) takes each branch
// under the tag: Max in Mode11 of the branch from the VRF or an immediate (op(b, b), exact for any bits, so a NaN
// or infinity in the branch not taken cannot leak), one op per tile branch, and the two added. The immediate takes
// its side's guards first, the VRF the rest. acc >= 0 is ordered, Greater(0) or Equal(0) (false on NaN): it fails
// where both bits are clear and holds where either is set, a slot each.

// CHECK-LABEL: func.func @matmul_kernel
// CHECK: fopt.to_vrf
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" %[[X:[0-9]+]] vrf {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", "rf"\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]
// CHECK: fopt.to_vrf
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" %[[X]] vrf {{.*}} {branches = {{\[\["1---", 0.000000e\+00 : f32\], \["-1--", 0.000000e\+00 : f32\], \["all", "rf"\]\]}}, mode = "Mode11"
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {furiosa.launch = "{\22grid\22: [12, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [256, 512]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [512, 384]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22fp16\22, \22shape\22: [256, 384]}]}"} {
  func.func @matmul_kernel(%arg0: tensor<256x512xf16>, %arg1: tensor<512x384xf16>, %arg2: tensor<256x384xf16>) -> tensor<256x384xf16> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.00999999977 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x384xf16>
    %1 = tensor.empty() : tensor<256x384xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<256x384xf32>) -> tensor<256x384xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<256x512xf16>, tensor<512x384xf16>) outs(%2 : tensor<256x384xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %5 = arith.extf %in : f16 to f32
      %6 = arith.extf %in_1 : f16 to f32
      %7 = arith.mulf %5, %6 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<256x384xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<256x384xf32>) outs(%0 : tensor<256x384xf16>) {
    ^bb0(%in: f32, %out: f16):
      %5 = arith.cmpf oge, %in, %cst_0 : f32
      %6 = arith.mulf %in, %cst : f32
      %7 = arith.select %5, %in, %6 : f32
      %8 = arith.truncf %7 : f32 to f16
      linalg.yield %8 : f16
    } -> tensor<256x384xf16>
    return %4 : tensor<256x384xf16>
  }
}
