// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s

// Every float predicate as tags, exact on NaN: Less, Greater and Equal are false on NaN, so an ordered predicate
// holds where one of its comparisons does (x >= k: Greater or Equal, a slot a set bit) and fails where all are
// clear; an unordered one is its ordered complement's opposite (x >= k unordered: Less clear; x > k unordered:
// Less and Equal clear). ord and uno test x < inf or x == inf. Each select of constants is one Max in Mode11 whose
// guarded slots take the side with fewer patterns; `0 >= x` is `x <= 0`.

// CHECK-LABEL: func.func @preds
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["1---", 1.000000e\+00 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["1---", 2.000000e\+00 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Less(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["1---", 4.000000e\+00 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Equal(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", 8.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", 1.600000e\+01 : f32\]\]}}, mode = "Mode11", tags = ["Less(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", 3.200000e\+01 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "Less(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 6.400000e\+01 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Less(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 1.280000e\+02 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 2.560000e\+02 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "Less(0.0E+0f32)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["0---", 5.120000e\+02 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Less(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["0---", 1.024000e\+03 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Greater(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["0---", 2.048000e\+03 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Equal(0.0E+0f32)", "False", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", 4.096000e\+03 : f32\]\]}}, mode = "Mode11", tags = ["Less(f32::INFINITY)", "Equal(f32::INFINITY)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 8.192000e\+03 : f32\], \["all", 0.000000e\+00 : f32\]\]}}, mode = "Mode11", tags = ["Less(f32::INFINITY)", "Equal(f32::INFINITY)", "False", "False"]
// CHECK: fopt.ve "vector_clip" "ClipBinaryOpF32::Max" {{.*}} {branches = {{\[\["00--", 0.000000e\+00 : f32\], \["all", 1.638400e\+04 : f32\]\]}}, mode = "Mode11", tags = ["Less(0.0E+0f32)", "Equal(0.0E+0f32)", "False", "False"]

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}]}"} {
  func.func @preds(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<64x64xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %m = arith.mulf %a, %b : f32
      %s = arith.addf %out, %m : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.000000e+00 : f32
      %k0 = arith.constant 1.000000e+00 : f32
      %c0 = arith.cmpf ogt, %in, %zero : f32
      %v0 = arith.select %c0, %k0, %zero : f32
      %s0 = arith.addf %zero, %v0 : f32
      %k1 = arith.constant 2.000000e+00 : f32
      %c1 = arith.cmpf olt, %in, %zero : f32
      %v1 = arith.select %c1, %k1, %zero : f32
      %s1 = arith.addf %s0, %v1 : f32
      %k2 = arith.constant 4.000000e+00 : f32
      %c2 = arith.cmpf oeq, %in, %zero : f32
      %v2 = arith.select %c2, %k2, %zero : f32
      %s2 = arith.addf %s1, %v2 : f32
      %k3 = arith.constant 8.000000e+00 : f32
      %c3 = arith.cmpf oge, %in, %zero : f32
      %v3 = arith.select %c3, %k3, %zero : f32
      %s3 = arith.addf %s2, %v3 : f32
      %k4 = arith.constant 1.600000e+01 : f32
      %c4 = arith.cmpf ole, %in, %zero : f32
      %v4 = arith.select %c4, %k4, %zero : f32
      %s4 = arith.addf %s3, %v4 : f32
      %k5 = arith.constant 3.200000e+01 : f32
      %c5 = arith.cmpf one, %in, %zero : f32
      %v5 = arith.select %c5, %k5, %zero : f32
      %s5 = arith.addf %s4, %v5 : f32
      %k6 = arith.constant 6.400000e+01 : f32
      %c6 = arith.cmpf ugt, %in, %zero : f32
      %v6 = arith.select %c6, %k6, %zero : f32
      %s6 = arith.addf %s5, %v6 : f32
      %k7 = arith.constant 1.280000e+02 : f32
      %c7 = arith.cmpf ult, %in, %zero : f32
      %v7 = arith.select %c7, %k7, %zero : f32
      %s7 = arith.addf %s6, %v7 : f32
      %k8 = arith.constant 2.560000e+02 : f32
      %c8 = arith.cmpf ueq, %in, %zero : f32
      %v8 = arith.select %c8, %k8, %zero : f32
      %s8 = arith.addf %s7, %v8 : f32
      %k9 = arith.constant 5.120000e+02 : f32
      %c9 = arith.cmpf uge, %in, %zero : f32
      %v9 = arith.select %c9, %k9, %zero : f32
      %s9 = arith.addf %s8, %v9 : f32
      %k10 = arith.constant 1.024000e+03 : f32
      %c10 = arith.cmpf ule, %in, %zero : f32
      %v10 = arith.select %c10, %k10, %zero : f32
      %s10 = arith.addf %s9, %v10 : f32
      %k11 = arith.constant 2.048000e+03 : f32
      %c11 = arith.cmpf une, %in, %zero : f32
      %v11 = arith.select %c11, %k11, %zero : f32
      %s11 = arith.addf %s10, %v11 : f32
      %k12 = arith.constant 4.096000e+03 : f32
      %c12 = arith.cmpf ord, %in, %zero : f32
      %v12 = arith.select %c12, %k12, %zero : f32
      %s12 = arith.addf %s11, %v12 : f32
      %k13 = arith.constant 8.192000e+03 : f32
      %c13 = arith.cmpf uno, %in, %zero : f32
      %v13 = arith.select %c13, %k13, %zero : f32
      %s13 = arith.addf %s12, %v13 : f32
      %k14 = arith.constant 1.638400e+04 : f32
      %c14 = arith.cmpf oge, %zero, %in : f32
      %v14 = arith.select %c14, %k14, %zero : f32
      %s14 = arith.addf %s13, %v14 : f32
      linalg.yield %s14 : f32
    } -> tensor<64x64xf32>
    return %4 : tensor<64x64xf32>
  }
}
