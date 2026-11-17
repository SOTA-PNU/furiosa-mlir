// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not='tensor<16xi8>'

// A dot whose K tile (16) is masked to K = 3 on both operands, with other = 0, reduces over the live
// extent 3 only.

// CHECK-LABEL: func.func @small_k
// CHECK: %[[K:.*]] = tensor.empty() : tensor<3xi8>
// CHECK: linalg.generic {{.*}}"reduction"{{.*}} ins(%[[K]] : tensor<3xi8>)

// Triton source:
// def small_k(a_ptr, b_ptr, c_ptr, K, BLOCK_K: tl.constexpr):
//     m = tl.arange(0, 16)
//     n = tl.arange(0, 16)
//     k = tl.arange(0, BLOCK_K)
//     a = tl.load(a_ptr + m[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.0)
//     b = tl.load(b_ptr + k[:, None] * 16 + n[None, :], mask=k[:, None] < K, other=0.0)
//     tl.store(c_ptr + m[:, None] * 16 + n[None, :], tl.dot(a, b, input_precision="ieee"))
// launch grid [1, 1, 1], buffers [('fp32', [16, 3]), ('fp32', [3, 16]), ('fp32', [16, 16])]

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [16, 3]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [3, 16]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [16, 16]}]}"} {
  tt.func public @small_k(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_1 = arith.constant dense<3> : tensor<16x1xi32>
    %cst_2 = arith.constant dense<3> : tensor<1x16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %2 = arith.cmpi slt, %1, %cst_2 : tensor<1x16xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<16x1xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %7 = tt.broadcast %6 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %8 = tt.broadcast %1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %10 = tt.broadcast %2 : tensor<1x16xi1> -> tensor<16x16xi1>
    %11 = tt.load %9, %10, %cst_0 : tensor<16x16x!tt.ptr<f32>>
    %12 = arith.cmpi slt, %3, %cst_1 : tensor<16x1xi32>
    %13 = arith.muli %3, %cst : tensor<16x1xi32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %16 = tt.broadcast %15 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %17 = tt.addptr %16, %8 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %18 = tt.broadcast %12 : tensor<16x1xi1> -> tensor<16x16xi1>
    %19 = tt.load %17, %18, %cst_0 : tensor<16x16x!tt.ptr<f32>>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %21 = tt.addptr %20, %13 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %22 = tt.broadcast %21 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %23 = tt.addptr %22, %8 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %24 = tt.dot %11, %19, %cst_0 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.store %23, %24 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

