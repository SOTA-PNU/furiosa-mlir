// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg --furiosa-structure %s | FileCheck %s --implicit-check-not='tensor.extract %'

// furiosa-structure promotes the raised gather (tensor.extract at a transposed address) to an
// indexing map: a transpose reads its input through (d1, d0).

// CHECK: #[[T:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: linalg.generic {indexing_maps = [#[[T]], {{.*}} ins(%{{.*}} : tensor<40x24xf32>) outs(%{{.*}} : tensor<24x40xf32>)

// Triton source:
// def transpose_tiles(x_ptr, y_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
//     rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
//     cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
//     mask = (rows[:, None] < R) & (cols[None, :] < C)
//     tile = tl.load(x_ptr + rows[:, None] * C + cols[None, :], mask=mask)
//     tl.store(y_ptr + cols[:, None] * R + rows[None, :], tl.trans(tile), mask=tl.trans(mask))
// launch grid [3, 2, 1], buffers [('fp32', [40, 24]), ('fp32', [24, 40])]

module attributes {furiosa.launch = "{\22grid\22: [3, 2, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [40, 24]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [24, 40]}]}"} {
  tt.func public @transpose_tiles(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<24> : tensor<16x1xi32>
    %cst_0 = arith.constant dense<24> : tensor<1x16xi32>
    %cst_1 = arith.constant dense<40> : tensor<16x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = tt.get_program_id y : i32
    %6 = arith.muli %5, %c16_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<16xi32>
    %8 = arith.addi %7, %2 : tensor<16xi32>
    %9 = tt.expand_dims %4 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %10 = arith.cmpi slt, %9, %cst_1 : tensor<16x1xi32>
    %11 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = arith.cmpi slt, %11, %cst_0 : tensor<1x16xi32>
    %13 = tt.broadcast %10 : tensor<16x1xi1> -> tensor<16x16xi1>
    %14 = tt.broadcast %12 : tensor<1x16xi1> -> tensor<16x16xi1>
    %15 = arith.andi %13, %14 : tensor<16x16xi1>
    %16 = arith.muli %9, %cst : tensor<16x1xi32>
    %17 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %19 = tt.broadcast %18 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %20 = tt.broadcast %11 : tensor<1x16xi32> -> tensor<16x16xi32>
    %21 = tt.addptr %19, %20 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %22 = tt.load %21, %15 : tensor<16x16x!tt.ptr<f32>>
    %23 = tt.trans %15 {order = array<i32: 1, 0>} : tensor<16x16xi1> -> tensor<16x16xi1>
    %24 = tt.expand_dims %8 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %25 = arith.muli %24, %cst_1 : tensor<16x1xi32>
    %26 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %27 = tt.addptr %26, %25 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %28 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %29 = tt.broadcast %27 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %30 = tt.broadcast %28 : tensor<1x16xi32> -> tensor<16x16xi32>
    %31 = tt.addptr %29, %30 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %32 = tt.trans %22 {order = array<i32: 1, 0>} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.store %31, %32, %23 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

