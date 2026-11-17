// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.histogram

// tl.histogram(x, 64): a reduction over the elements of (x_i == bin), one count per bin.

// CHECK-LABEL: func.func @extra_kernel
// CHECK: linalg.generic {{.*}}["parallel", "reduction"]{{.*}} outs({{.*}} : tensor<64xi32>)
// CHECK: arith.cmpi eq
// CHECK: arith.addi %out

// Triton source: harness/ops extra_kernel, OP=20 (histogram) on a 64x64 block of int32 in [0, 64).

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [64, 64]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [64, 64]}, {\22arg\22: 2, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [64, 64]}]}"} {
  tt.func public @extra_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %3 = arith.muli %1, %cst : tensor<64x1xi32>
    %4 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x64xi32>
    %5 = tt.broadcast %2 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = arith.addi %4, %5 : tensor<64x64xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x64x!tt.ptr<i32>>
    %8 = tt.addptr %7, %6 : tensor<64x64x!tt.ptr<i32>>, tensor<64x64xi32>
    %9 = tt.load %8 : tensor<64x64x!tt.ptr<i32>>
    %10 = tt.reshape %9 allow_reorder : tensor<64x64xi32> -> tensor<4096xi32>
    %11 = tt.histogram %10 : tensor<4096xi32> -> tensor<64xi32>
    %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %13 = tt.broadcast %12 : tensor<1x64xi32> -> tensor<64x64xi32>
    %14 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<64x64x!tt.ptr<i32>>
    %15 = tt.addptr %14, %6 : tensor<64x64x!tt.ptr<i32>>, tensor<64x64xi32>
    tt.store %15, %13 : tensor<64x64x!tt.ptr<i32>>
    tt.return
  }
}

