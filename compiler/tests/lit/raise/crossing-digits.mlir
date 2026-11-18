// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.floordivsi --implicit-check-not=arith.remsi

// Rows n * (P + 1) + 1 + p of a [N, P + 1] buffer (a class token row per image), n = m // P and p = m % P of
// a 4-row block of m: blocks that cross a multiple of P keep the digits other blocks resolve, so every block
// keeps them and the store's axes are the digits [N, P] past the leading row: the view [3, 7, 4], written at
// rows 1..6.

// CHECK-LABEL: func.func @class_rows
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<3x6x4xf32>)
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 1, 0] [3, 6, 4] [1, 1, 1] : tensor<3x6x4xf32> into tensor<3x7x4xf32>

// Triton source:
// def class_rows(x_ptr, y_ptr, M, P: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
//     m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
//     d = tl.arange(0, D)
//     n = m // P
//     p = m % P
//     mask = (m < M)[:, None]
//     v = tl.load(x_ptr + m[:, None] * D + d[None, :], mask=mask)
//     tl.store(y_ptr + (n * (P + 1) + 1 + p)[:, None] * D + d[None, :], v, mask=mask)
// launch grid [5, 1, 1], M=18, P=6, D=4, BLOCK_M=4, buffers [('fp32', [18, 4]), ('fp32', [3, 7, 4])]

module attributes {furiosa.launch = "{\22grid\22: [5, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [18, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [3, 7, 4]}], \22scalars\22: []}"} {
  tt.func public @class_rows(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<4xi32>
    %cst_0 = arith.constant dense<7> : tensor<4xi32>
    %cst_1 = arith.constant dense<4> : tensor<4x1xi32>
    %cst_2 = arith.constant dense<18> : tensor<4xi32>
    %cst_3 = arith.constant dense<6> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = arith.divsi %4, %cst_3 : tensor<4xi32>
    %6 = arith.remsi %4, %cst_3 : tensor<4xi32>
    %7 = arith.cmpi slt, %4, %cst_2 : tensor<4xi32>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %9 = tt.expand_dims %4 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %10 = arith.muli %9, %cst_1 : tensor<4x1xi32>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %14 = tt.broadcast %12 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.broadcast %13 : tensor<1x4xi32> -> tensor<4x4xi32>
    %16 = tt.addptr %14, %15 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %17 = tt.broadcast %8 : tensor<4x1xi1> -> tensor<4x4xi1>
    %18 = tt.load %16, %17 : tensor<4x4x!tt.ptr<f32>>
    %19 = arith.muli %5, %cst_0 : tensor<4xi32>
    %20 = arith.addi %19, %cst : tensor<4xi32>
    %21 = arith.addi %20, %6 : tensor<4xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %23 = arith.muli %22, %cst_1 : tensor<4x1xi32>
    %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %26 = tt.broadcast %25 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %27 = tt.addptr %26, %15 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %27, %18, %17 : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}
