// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.remsi

// A read of window-partitioned rows (Swin's q): the stored token axis is a digit axis over the window's rows and
// columns, whose tile origin mixes both, so token t floordiv 3 is one expression of the coordinates only within a
// window's tile. Rewritten there, every window agrees up to a constant affine in its tile's first coordinates,
// which the coordinates determine: the read is a plain function of the image position.

// CHECK-LABEL: func.func @window_rows
// CHECK: tensor.empty() : tensor<2x3x2x3x4xf32>
// CHECK: tensor.extract %{{.*}} : tensor<144xf32>

// Triton source (launch grid [4, 1, 1], H = W = 6, WS = 3, BLOCK_T = 16, D = 4; x, y: fp32 [36, 4]):
// def window_rows(x_ptr, y_ptr, H: tl.constexpr, W: tl.constexpr, WS: tl.constexpr, BLOCK_T: tl.constexpr,
//                 D: tl.constexpr):
//     win = tl.program_id(0)
//     t = tl.arange(0, BLOCK_T)
//     m = t < WS * WS
//     ry = (win // (W // WS)) * WS + t // WS
//     rx = (win % (W // WS)) * WS + t % WS
//     rows = (ry % H) * W + rx % W
//     d = tl.arange(0, D)
//     x = tl.load(x_ptr + rows[:, None] * D + d[None, :], mask=m[:, None], other=0.0)
//     tl.store(y_ptr + rows[:, None] * D + d[None, :], x * 2.0, mask=m[:, None])

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [36, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [36, 4]}], \22scalars\22: []}"} {
  tt.func public @window_rows(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2.000000e+00> : tensor<16x4xf32>
    %x = arith.constant dense<0.000000e+00> : tensor<16x4xf32>
    %cst_0 = arith.constant dense<4> : tensor<16x1xi32>
    %cst_1 = arith.constant dense<6> : tensor<16xi32>
    %cst_2 = arith.constant dense<3> : tensor<16xi32>
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %m = arith.constant dense<9> : tensor<16xi32>
    %win = tt.get_program_id x : i32
    %t = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %m_3 = arith.cmpi slt, %t, %m : tensor<16xi32>
    %ry = arith.divsi %win, %c2_i32 : i32
    %ry_4 = arith.muli %ry, %c3_i32 : i32
    %ry_5 = arith.divsi %t, %cst_2 : tensor<16xi32>
    %ry_6 = tt.splat %ry_4 : i32 -> tensor<16xi32>
    %ry_7 = arith.addi %ry_6, %ry_5 : tensor<16xi32>
    %rx = arith.remsi %win, %c2_i32 : i32
    %rx_8 = arith.muli %rx, %c3_i32 : i32
    %rx_9 = arith.remsi %t, %cst_2 : tensor<16xi32>
    %rx_10 = tt.splat %rx_8 : i32 -> tensor<16xi32>
    %rx_11 = arith.addi %rx_10, %rx_9 : tensor<16xi32>
    %rows = arith.remsi %ry_7, %cst_1 : tensor<16xi32>
    %rows_12 = arith.muli %rows, %cst_1 : tensor<16xi32>
    %rows_13 = arith.remsi %rx_11, %cst_1 : tensor<16xi32>
    %rows_14 = arith.addi %rows_12, %rows_13 : tensor<16xi32>
    %d = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %x_15 = tt.expand_dims %m_3 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %x_16 = tt.expand_dims %rows_14 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %x_17 = arith.muli %x_16, %cst_0 : tensor<16x1xi32>
    %x_18 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %x_19 = tt.addptr %x_18, %x_17 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %x_20 = tt.expand_dims %d {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %x_21 = tt.broadcast %x_19 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x4x!tt.ptr<f32>>
    %x_22 = tt.broadcast %x_20 : tensor<1x4xi32> -> tensor<16x4xi32>
    %x_23 = tt.addptr %x_21, %x_22 : tensor<16x4x!tt.ptr<f32>>, tensor<16x4xi32>
    %x_24 = tt.broadcast %x_15 : tensor<16x1xi1> -> tensor<16x4xi1>
    %x_25 = tt.load %x_23, %x_24, %x : tensor<16x4x!tt.ptr<f32>>
    %0 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %x_17 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x4x!tt.ptr<f32>>
    %3 = tt.addptr %2, %x_22 : tensor<16x4x!tt.ptr<f32>>, tensor<16x4xi32>
    %4 = arith.mulf %x_25, %cst : tensor<16x4xf32>
    tt.store %3, %4, %x_24 : tensor<16x4x!tt.ptr<f32>>
    tt.return
  }
}
