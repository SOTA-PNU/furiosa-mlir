// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// A store of window-partitioned rows (Swin's): 16 lanes hold a 3 x 3 window's 9 tokens, masked, at image row
// (window row * 3 + t floordiv 3) mod 6. Over all 16 lanes the row passes the image's last and the mod stays, so
// the address is no sum of digits; over the lanes the mask keeps it resolves, and the windows' tokens cover the
// image: one store over its rows and columns.

// CHECK-LABEL: func.func @window_store
// CHECK: tensor.empty() : tensor<6x6x4xf32>

// Triton source (launch grid [4, 1, 1], H = W = 6, WS = 3, BLOCK_T = 16, D = 4; y: fp32 [36, 4]):
// def window_store(y_ptr, H: tl.constexpr, W: tl.constexpr, WS: tl.constexpr, BLOCK_T: tl.constexpr, D: tl.constexpr):
//     win = tl.program_id(0)
//     t = tl.arange(0, BLOCK_T)
//     ry = (win // (W // WS)) * WS + t // WS
//     rx = (win % (W // WS)) * WS + t % WS
//     rows = (ry % H) * W + rx % W
//     d = tl.arange(0, D)
//     tl.store(y_ptr + rows[:, None] * D + d[None, :], tl.full((BLOCK_T, D), 1.0, tl.float32), mask=(t < WS * WS)[:, None])

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [36, 4]}], \22scalars\22: []}"} {
  tt.func public @window_store(%y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<16x4xf32>
    %cst_0 = arith.constant dense<4> : tensor<16x1xi32>
    %cst_1 = arith.constant dense<9> : tensor<16xi32>
    %cst_2 = arith.constant dense<6> : tensor<16xi32>
    %cst_3 = arith.constant dense<3> : tensor<16xi32>
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %win = tt.get_program_id x : i32
    %t = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ry = arith.divsi %win, %c2_i32 : i32
    %ry_4 = arith.muli %ry, %c3_i32 : i32
    %ry_5 = arith.divsi %t, %cst_3 : tensor<16xi32>
    %ry_6 = tt.splat %ry_4 : i32 -> tensor<16xi32>
    %ry_7 = arith.addi %ry_6, %ry_5 : tensor<16xi32>
    %rx = arith.remsi %win, %c2_i32 : i32
    %rx_8 = arith.muli %rx, %c3_i32 : i32
    %rx_9 = arith.remsi %t, %cst_3 : tensor<16xi32>
    %rx_10 = tt.splat %rx_8 : i32 -> tensor<16xi32>
    %rx_11 = arith.addi %rx_10, %rx_9 : tensor<16xi32>
    %rows = arith.remsi %ry_7, %cst_2 : tensor<16xi32>
    %rows_12 = arith.muli %rows, %cst_2 : tensor<16xi32>
    %rows_13 = arith.remsi %rx_11, %cst_2 : tensor<16xi32>
    %rows_14 = arith.addi %rows_12, %rows_13 : tensor<16xi32>
    %d = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %0 = arith.cmpi slt, %t, %cst_1 : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %2 = tt.expand_dims %rows_14 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %3 = arith.muli %2, %cst_0 : tensor<16x1xi32>
    %4 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %6 = tt.expand_dims %d {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %7 = tt.broadcast %5 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x4x!tt.ptr<f32>>
    %8 = tt.broadcast %6 : tensor<1x4xi32> -> tensor<16x4xi32>
    %9 = tt.addptr %7, %8 : tensor<16x4x!tt.ptr<f32>>, tensor<16x4xi32>
    %10 = tt.broadcast %1 : tensor<16x1xi1> -> tensor<16x4xi1>
    tt.store %9, %cst, %10 : tensor<16x4x!tt.ptr<f32>>
    tt.return
  }
}
