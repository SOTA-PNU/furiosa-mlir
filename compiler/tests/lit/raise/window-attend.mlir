// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// Window attention (Swin's): along the 16 key lanes of the score matrix the key's row (window row * 3 +
// t floordiv 3) mod 6 passes the image's last, so the key read's address is one expression only over the 9 lanes
// the mask keeps. It is rewritten there; the other lanes read `other` through a clamped address.

// CHECK-LABEL: func.func @window_attend
// CHECK: tensor.empty() : tensor<2x3x2x3x16xf32>
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: arith.select

// Triton source (launch grid [4, 1, 1], H = W = 6, WS = 3, BLOCK_T = 16, D = 16; x, y: fp32 [36, 16]):
// def window_attend(x_ptr, y_ptr, H: tl.constexpr, W: tl.constexpr, WS: tl.constexpr, BLOCK_T: tl.constexpr,
//                   D: tl.constexpr):
//     win = tl.program_id(0)
//     t = tl.arange(0, BLOCK_T)
//     m = t < WS * WS
//     ry = (win // (W // WS)) * WS + t // WS
//     rx = (win % (W // WS)) * WS + t % WS
//     rows = (ry % H) * W + rx % W
//     d = tl.arange(0, D)
//     q = tl.load(x_ptr + rows[:, None] * D + d[None, :], mask=m[:, None], other=0.0)
//     k = tl.load(x_ptr + rows[None, :] * D + d[:, None], mask=m[None, :], other=0.0)
//     s = tl.where(m[None, :], tl.dot(q, k, input_precision="ieee"), float("-inf"))
//     p = tl.exp(s - tl.max(s, axis=1)[:, None])
//     v = tl.load(x_ptr + rows[:, None] * D + d[None, :], mask=m[:, None], other=0.0)
//     o = tl.dot(p, v, input_precision="ieee") / tl.sum(p, axis=1)[:, None]
//     tl.store(y_ptr + rows[:, None] * D + d[None, :], o, mask=m[:, None])

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [36, 16]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [36, 16]}], \22scalars\22: []}"} {
  tt.func public @window_attend(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %s = arith.constant dense<0xFF800000> : tensor<16x16xf32>
    %k = arith.constant dense<16> : tensor<1x16xi32>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32>
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
    %q = tt.expand_dims %m_3 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %q_15 = tt.expand_dims %rows_14 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %q_16 = arith.muli %q_15, %cst_0 : tensor<16x1xi32>
    %q_17 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %q_18 = tt.addptr %q_17, %q_16 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %q_19 = tt.expand_dims %t {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %q_20 = tt.broadcast %q_18 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %q_21 = tt.broadcast %q_19 : tensor<1x16xi32> -> tensor<16x16xi32>
    %q_22 = tt.addptr %q_20, %q_21 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %q_23 = tt.broadcast %q : tensor<16x1xi1> -> tensor<16x16xi1>
    %q_24 = tt.load %q_22, %q_23, %cst : tensor<16x16x!tt.ptr<f32>>
    %k_25 = tt.expand_dims %m_3 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %k_26 = tt.expand_dims %rows_14 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %k_27 = arith.muli %k_26, %k : tensor<1x16xi32>
    %k_28 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %k_29 = tt.addptr %k_28, %k_27 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %k_30 = tt.expand_dims %t {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %k_31 = tt.broadcast %k_29 : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %k_32 = tt.broadcast %k_30 : tensor<16x1xi32> -> tensor<16x16xi32>
    %k_33 = tt.addptr %k_31, %k_32 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %k_34 = tt.broadcast %k_25 : tensor<1x16xi1> -> tensor<16x16xi1>
    %k_35 = tt.load %k_33, %k_34, %cst : tensor<16x16x!tt.ptr<f32>>
    %s_36 = tt.dot %q_24, %k_35, %cst : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %s_37 = arith.select %k_34, %s_36, %s : tensor<16x16xi1>, tensor<16x16xf32>
    %p = "tt.reduce"(%s_37) <{axis = 1 : i32}> ({
    ^bb0(%p_46: f32, %p_47: f32):
      %p_48 = arith.maxnumf %p_46, %p_47 : f32
      tt.reduce.return %p_48 : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %p_38 = tt.expand_dims %p {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %p_39 = tt.broadcast %p_38 : tensor<16x1xf32> -> tensor<16x16xf32>
    %p_40 = arith.subf %s_37, %p_39 : tensor<16x16xf32>
    %p_41 = math.exp %p_40 : tensor<16x16xf32>
    %o = tt.dot %p_41, %q_24, %cst : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %o_42 = "tt.reduce"(%p_41) <{axis = 1 : i32}> ({
    ^bb0(%o_46: f32, %o_47: f32):
      %o_48 = arith.addf %o_46, %o_47 : f32
      tt.reduce.return %o_48 : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %o_43 = tt.expand_dims %o_42 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %o_44 = tt.broadcast %o_43 : tensor<16x1xf32> -> tensor<16x16xf32>
    %o_45 = arith.divf %o, %o_44 : tensor<16x16xf32>
    %0 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %q_16 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %3 = tt.addptr %2, %q_21 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %3, %o_45, %q_23 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
