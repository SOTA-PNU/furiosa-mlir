// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.map_elementwise --implicit-check-not=cf.cond_br

// tl.map_elementwise with a scalar function that branches (if x < y: -1 elif x == y: 0 else 1): its
// blocks are pure, so each runs, and the value a block receives is selected by the branch taken.

// CHECK-LABEL: func.func @extra_kernel
// CHECK: %[[LT:.*]] = arith.cmpf olt
// CHECK: %[[EQ:.*]] = arith.cmpf oeq
// CHECK: arith.select
// CHECK: linalg.yield

// Triton source: harness/ops extra_kernel, OP=2 (map_elementwise) on a 64x64 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 2, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}]}"} {
  tt.func public @extra_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %3 = arith.muli %1, %cst : tensor<64x1xi32>
    %4 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x64xi32>
    %5 = tt.broadcast %2 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = arith.addi %4, %5 : tensor<64x64xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %9 = tt.load %8 : tensor<64x64x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %11 = tt.addptr %10, %6 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %12 = tt.load %11 : tensor<64x64x!tt.ptr<f32>>
    %13 = "tt.map_elementwise"(%9, %12) <{pack = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %17 = arith.cmpf olt, %arg3, %arg4 : f32
      cf.cond_br %17, ^bb2(%c-1_i32 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %18 = arith.cmpf oeq, %arg3, %arg4 : f32
      cf.cond_br %18, ^bb2(%c0_i32 : i32), ^bb2(%c1_i32 : i32)
    ^bb2(%19: i32):  // 3 preds: ^bb0, ^bb1, ^bb1
      tt.map_elementwise.return %19 : i32
    }) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xi32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %15 = tt.addptr %14, %6 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %16 = arith.sitofp %13 : tensor<64x64xi32> to tensor<64x64xf32>
    tt.store %15, %16 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}

