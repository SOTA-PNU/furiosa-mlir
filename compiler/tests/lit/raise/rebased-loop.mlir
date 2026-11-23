// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// A loop whose first iteration differs between programs as no expression of the program ids (a window's first key
// block, max(start - W + 1, 0) // BLOCK * BLOCK: blocks 0, 0, 0, 8 for programs at 0, 8, 16, 24) counts from the
// least of them (0) in every program, which runs the iterations from its own lower bound on: the accumulation over
// the 4 blocks of the longest trip keeps an iteration where counter * 8 >= the lower bound (computed from the
// coordinates: max, divsi, muli) and counter < the trip of its program ((row floordiv 8) + 1).

// CHECK-LABEL: func.func @window_sum
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]} ins(%{{.*}} : tensor<4xi8>)
// CHECK: %[[MAX:.*]] = arith.maxsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[DIV:.*]] = arith.divsi %[[MAX]], %{{.*}} : i32
// CHECK: %[[LO:.*]] = arith.muli %[[DIV]], %{{.*}} : i32
// CHECK: %[[LOI:.*]] = arith.index_cast %[[LO]] : i32 to index
// CHECK: %[[AT:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[FIRST:.*]] = arith.cmpi sge, %[[AT]], %[[LOI]] : index
// CHECK: %[[LAST:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[ACTIVE:.*]] = arith.andi %[[FIRST]], %[[LAST]] : i1
// CHECK: arith.select %[[ACTIVE]], %{{.*}}, %{{.*}} : f32
// CHECK: return %{{.*}} : tensor<30xf32>

// Triton source:
// def window_sum(x_ptr, y_ptr, N, W, BLOCK: tl.constexpr):
//     start = tl.program_id(0) * BLOCK
//     rows = start + tl.arange(0, BLOCK)
//     acc = tl.zeros((BLOCK,), tl.float32)
//     lo = tl.maximum(start - W + 1, 0) // BLOCK * BLOCK
//     hi = tl.minimum(start + BLOCK, N)
//     for k0 in range(lo, hi, BLOCK):
//         cols = k0 + tl.arange(0, BLOCK)
//         x = tl.load(x_ptr + cols, mask=cols < N, other=0.0)
//         d = rows[:, None] - cols[None, :]
//         acc += tl.sum(tl.where((d >= 0) & (d < W), x[None, :], 0.0), axis=1)
//     tl.store(y_ptr + rows, acc, mask=rows < N)
// launch grid [4, 1, 1], N=30, W=10, BLOCK=8, buffers [('fp32', [30]), ('fp32', [30])]

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [30]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [30]}], \22scalars\22: []}"} {
  tt.func public @window_sum(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %lo = arith.constant -9 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %cst_0 = arith.constant dense<10> : tensor<8x8xi32>
    %cst_1 = arith.constant dense<0> : tensor<8x8xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %cst_3 = arith.constant dense<30> : tensor<8xi32>
    %c30_i32 = arith.constant 30 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %start = tt.get_program_id x : i32
    %start_4 = arith.muli %start, %c8_i32 : i32
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %rows_5 = tt.splat %start_4 : i32 -> tensor<8xi32>
    %rows_6 = arith.addi %rows_5, %rows : tensor<8xi32>
    %lo_7 = arith.addi %start_4, %lo : i32
    %lo_8 = arith.maxsi %lo_7, %c0_i32 : i32
    %lo_9 = arith.divsi %lo_8, %c8_i32 : i32
    %lo_10 = arith.muli %lo_9, %c8_i32 : i32
    %hi = arith.addi %start_4, %c8_i32 : i32
    %hi_11 = arith.minsi %hi, %c30_i32 : i32
    %acc = scf.for %k0 = %lo_10 to %hi_11 step %c8_i32 iter_args(%acc_12 = %cst_2) -> (tensor<8xf32>)  : i32 {
      %cols = tt.splat %k0 : i32 -> tensor<8xi32>
      %cols_13 = arith.addi %cols, %rows : tensor<8xi32>
      %x = arith.cmpi slt, %cols_13, %cst_3 : tensor<8xi32>
      %x_14 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %x_15 = tt.addptr %x_14, %cols_13 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %x_16 = tt.load %x_15, %x, %cst_2 : tensor<8x!tt.ptr<f32>>
      %d = tt.expand_dims %rows_6 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %d_17 = tt.expand_dims %cols_13 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %d_18 = tt.broadcast %d : tensor<8x1xi32> -> tensor<8x8xi32>
      %d_19 = tt.broadcast %d_17 : tensor<1x8xi32> -> tensor<8x8xi32>
      %d_20 = arith.subi %d_18, %d_19 : tensor<8x8xi32>
      %acc_21 = arith.cmpi sge, %d_20, %cst_1 : tensor<8x8xi32>
      %acc_22 = arith.cmpi slt, %d_20, %cst_0 : tensor<8x8xi32>
      %acc_23 = arith.andi %acc_21, %acc_22 : tensor<8x8xi1>
      %acc_24 = tt.expand_dims %x_16 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
      %acc_25 = tt.broadcast %acc_24 : tensor<1x8xf32> -> tensor<8x8xf32>
      %acc_26 = arith.select %acc_23, %acc_25, %cst : tensor<8x8xi1>, tensor<8x8xf32>
      %acc_27 = "tt.reduce"(%acc_26) <{axis = 1 : i32}> ({
      ^bb0(%acc_29: f32, %acc_30: f32):
        %acc_31 = arith.addf %acc_29, %acc_30 : f32
        tt.reduce.return %acc_31 : f32
      }) : (tensor<8x8xf32>) -> tensor<8xf32>
      %acc_28 = arith.addf %acc_12, %acc_27 : tensor<8xf32>
      scf.yield %acc_28 : tensor<8xf32>
    }
    %0 = arith.cmpi slt, %rows_6, %cst_3 : tensor<8xi32>
    %1 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %2 = tt.addptr %1, %rows_6 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %2, %acc, %0 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
