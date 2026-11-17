// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.floordivsi --implicit-check-not=arith.remsi

// A flattened pixel index tiled across output rows iterates the output's own dims [1, OH, OW, C];
// the dims' ranges resolve every div/mod of it.

// CHECK-LABEL: func.func @pool_pairs
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<1x4x4x4xf32>)

// Triton source:
// def pool_pairs(x_ptr, y_ptr, OH, OW, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
//     p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
//     c = tl.arange(0, BLOCK_C)
//     oh = p // OW
//     ow = p % OW
//     m = (p < OH * OW)[:, None] & (c < C)[None, :]
//     src = x_ptr + ((2 * oh * (2 * OW) + 2 * ow) * C)[:, None] + c[None, :]
//     v = tl.load(src, mask=m, other=0.0) + tl.load(src + C, mask=m, other=0.0)
//     tl.store(y_ptr + (p * C)[:, None] + c[None, :], v, mask=m)
// launch grid [2, 1, 1], buffers [('fp32', [1, 8, 8, 4]), ('fp32', [1, 4, 4, 4])]

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 8, 8, 4]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 4, 4, 4]}]}"} {
  tt.func public @pool_pairs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<8x4xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8x4xf32>
    %cst_1 = arith.constant dense<8> : tensor<8xi32>
    %cst_2 = arith.constant dense<2> : tensor<8xi32>
    %cst_3 = arith.constant dense<4> : tensor<4xi32>
    %cst_4 = arith.constant dense<16> : tensor<8xi32>
    %cst_5 = arith.constant dense<4> : tensor<8xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c8_i32 : i32
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.splat %1 : i32 -> tensor<8xi32>
    %4 = arith.addi %3, %2 : tensor<8xi32>
    %5 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %6 = arith.divsi %4, %cst_5 : tensor<8xi32>
    %7 = arith.remsi %4, %cst_5 : tensor<8xi32>
    %8 = arith.cmpi slt, %4, %cst_4 : tensor<8xi32>
    %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
    %10 = arith.cmpi slt, %5, %cst_3 : tensor<4xi32>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %12 = tt.broadcast %9 : tensor<8x1xi1> -> tensor<8x4xi1>
    %13 = tt.broadcast %11 : tensor<1x4xi1> -> tensor<8x4xi1>
    %14 = arith.andi %12, %13 : tensor<8x4xi1>
    %15 = arith.muli %6, %cst_2 : tensor<8xi32>
    %16 = arith.muli %15, %cst_1 : tensor<8xi32>
    %17 = arith.muli %7, %cst_2 : tensor<8xi32>
    %18 = arith.addi %16, %17 : tensor<8xi32>
    %19 = arith.muli %18, %cst_5 : tensor<8xi32>
    %20 = tt.expand_dims %19 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %21 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %22 = tt.addptr %21, %20 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %24 = tt.broadcast %22 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<8x4xi32>
    %26 = tt.addptr %24, %25 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    %27 = tt.load %26, %14, %cst_0 : tensor<8x4x!tt.ptr<f32>>
    %28 = tt.addptr %26, %cst : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    %29 = tt.load %28, %14, %cst_0 : tensor<8x4x!tt.ptr<f32>>
    %30 = arith.addf %27, %29 : tensor<8x4xf32>
    %31 = arith.muli %4, %cst_5 : tensor<8xi32>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %33 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %34 = tt.addptr %33, %32 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %35 = tt.broadcast %34 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x4x!tt.ptr<f32>>
    %36 = tt.addptr %35, %25 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %36, %30, %14 : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }
}

