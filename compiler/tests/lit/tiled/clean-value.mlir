// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma --furiosa-ve-fuse %s | FileCheck %s --check-prefix=FUSED

// A max reduction's clean value (NaN skipped, as maxnumf does: x > -inf ? x : -inf, VGG16's 2x2 max pool) is two
// logic ops on the key, exact for any NaN: where the tag fails (x NaN or -inf), (x | -inf) & -inf = -inf; where it
// holds, (x | 0) & ~0 = x. The second reads the first's tags, in its pass, before the reduction's max.

// CHECK-LABEL: func.func @maxpool2d_kernel
// CHECK: fopt.ve "vector_logic" "LogicBinaryOpF32::BitOr" {{.*}} {branches = {{\[\["0---", 0xFF800000 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, tags = ["Greater(f32::NEG_INFINITY)"
// CHECK-NEXT: fopt.ve "vector_logic" "LogicBinaryOpF32::BitAnd" {{.*}} {branches = {{\[\["0---", 0xFF800000 : f32\], \["all", 0xFFFFFFFF : f32\]\]}}}

// FUSED: fopt.vector_tag_cmp ["Greater(f32::NEG_INFINITY)"
// FUSED-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitOr"
// FUSED-NEXT: fopt.vector_alu "vector_logic" "LogicBinaryOpF32::BitAnd"
// FUSED-NEXT: fopt.vector_alu "vector_clip" "ClipBinaryOpF32::Max"
#map = affine_map<(d0, d1, d2, d3) -> (0, d1 * 2, d2 * 2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {furiosa.launch = "{\22grid\22: [392, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22activation\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 224, 224, 64]}, {\22arg\22: 1, \22kind\22: \22activation\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 112, 112, 64]}]}"} {
  func.func @maxpool2d_kernel(%arg0: tensor<1x224x224x64xf32>, %arg1: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> attributes {furiosa.written = [1]} {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x112x112x64xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 223, 223, 64] [1, 1, 1, 1] : tensor<1x224x224x64xf32> to tensor<1x223x223x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 1, 0] [1, 223, 223, 64] [1, 1, 1, 1] : tensor<1x224x224x64xf32> to tensor<1x223x223x64xf32>
    %extracted_slice_1 = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 223, 223, 64] [1, 1, 1, 1] : tensor<1x224x224x64xf32> to tensor<1x223x223x64xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, 1, 1, 0] [1, 223, 223, 64] [1, 1, 1, 1] : tensor<1x224x224x64xf32> to tensor<1x223x223x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 : tensor<1x223x223x64xf32>, tensor<1x223x223x64xf32>, tensor<1x223x223x64xf32>, tensor<1x223x223x64xf32>) outs(%0 : tensor<1x112x112x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
      %cst_6 = arith.constant 0xFF800000 : f32
      %2 = arith.cmpf ogt, %in, %cst_6 : f32
      %3 = arith.select %2, %in, %cst_6 : f32
      %4 = arith.maximumf %3, %cst : f32
      %5 = arith.maxnumf %4, %in_3 : f32
      %6 = arith.maxnumf %5, %in_4 : f32
      %7 = arith.maxnumf %6, %in_5 : f32
      linalg.yield %7 : f32
    } -> tensor<1x112x112x64xf32>
    return %1 : tensor<1x112x112x64xf32>
  }
}

