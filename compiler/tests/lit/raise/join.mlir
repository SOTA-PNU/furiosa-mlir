// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.join

// tl.join of two 64x32 halves along a new last axis of 2, reshaped to 64x64: the new axis selects the
// operand (0: lhs, 1: rhs), read at the result's column floordiv 2.

// CHECK-LABEL: func.func @extra_kernel
// CHECK: %[[FIRST:.*]] = arith.cmpi eq
// CHECK: arith.select %[[FIRST]]

// Triton source: harness/ops extra_kernel, OP=13 (join) on a 64x64 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 2, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}]}"} {
  tt.func public @extra_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %3 = arith.muli %1, %cst : tensor<64x1xi32>
    %4 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x64xi32>
    %5 = tt.broadcast %2 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = arith.addi %4, %5 : tensor<64x64xi32>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %3 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %11 = tt.broadcast %10 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x32x!tt.ptr<f32>>
    %12 = tt.broadcast %8 : tensor<1x32xi32> -> tensor<64x32xi32>
    %13 = tt.addptr %11, %12 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
    %14 = tt.load %13 : tensor<64x32x!tt.ptr<f32>>
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %16 = tt.addptr %15, %3 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %17 = tt.broadcast %16 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x32x!tt.ptr<f32>>
    %18 = tt.addptr %17, %12 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
    %19 = tt.load %18 : tensor<64x32x!tt.ptr<f32>>
    %20 = tt.join %14, %19 : tensor<64x32xf32> -> tensor<64x32x2xf32>
    %21 = tt.reshape %20 : tensor<64x32x2xf32> -> tensor<64x64xf32>
    %22 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %23 = tt.addptr %22, %6 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    tt.store %23, %21 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}

