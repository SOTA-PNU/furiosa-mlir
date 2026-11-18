// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tensor.insert_slice

// A loop over column tiles whose width (4) divides neither the row (10) nor is a dim of the buffer: the tile
// counter and the lanes are the digits of the column under the tile width (view [2, 10], tile = col floordiv 4).
// The store reads the loop's running carry at the current tile: the accumulation of the tiles before it, a
// reduction over the 3 iterations that adds only those before the column's own (cmpi slt of the counters).

// CHECK-LABEL: func.func @exclusive_rows
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<3xi8>) outs(%{{.*}} : tensor<2x10xi32>)
// CHECK: arith.floordivsi %{{.*}}, %c4
// CHECK: arith.cmpi slt
// CHECK: return %{{.*}} : tensor<2x10xi32>

// Triton source:
// def exclusive_rows(x_ptr, y_ptr, C, BLOCK_C: tl.constexpr):
//     row = tl.program_id(0)
//     carry = 0
//     for c0 in range(0, C, BLOCK_C):
//         cols = c0 + tl.arange(0, BLOCK_C)
//         mask = cols < C
//         x = tl.load(x_ptr + row * C + cols, mask=mask, other=0)
//         tl.store(y_ptr + row * C + cols, carry + tl.cumsum(x, axis=0) - x, mask=mask)
//         carry += tl.sum(x, axis=0)
// launch grid [2, 1, 1], C=10, BLOCK_C=4, buffers [('i32', [2, 10]), ('i32', [2, 10])]

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [2, 10]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [2, 10]}], \22scalars\22: []}"} {
  tt.func public @exclusive_rows(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0> : tensor<4xi32>
    %cst_0 = arith.constant dense<10> : tensor<4xi32>
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = scf.for %arg2 = %c0_i32 to %c10_i32 step %c4_i32 iter_args(%arg3 = %c0_i32) -> (i32)  : i32 {
      %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %3 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %4 = arith.addi %3, %2 : tensor<4xi32>
      %5 = arith.cmpi slt, %4, %cst_0 : tensor<4xi32>
      %6 = arith.muli %0, %c10_i32 : i32
      %7 = tt.addptr %arg0, %6 : !tt.ptr<i32>, i32
      %8 = tt.splat %7 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %10 = tt.load %9, %5, %cst : tensor<4x!tt.ptr<i32>>
      %11 = tt.addptr %arg1, %6 : !tt.ptr<i32>, i32
      %12 = tt.splat %11 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %14 = "tt.scan"(%10) <{axis = 0 : i32, reverse = false}> ({
      ^bb0(%arg4: i32, %arg5: i32):
        %20 = arith.addi %arg4, %arg5 : i32
        tt.scan.return %20 : i32
      }) : (tensor<4xi32>) -> tensor<4xi32>
      %15 = tt.splat %arg3 : i32 -> tensor<4xi32>
      %16 = arith.addi %15, %14 : tensor<4xi32>
      %17 = arith.subi %16, %10 : tensor<4xi32>
      tt.store %13, %17, %5 : tensor<4x!tt.ptr<i32>>
      %18 = "tt.reduce"(%10) <{axis = 0 : i32}> ({
      ^bb0(%arg4: i32, %arg5: i32):
        %20 = arith.addi %arg4, %arg5 : i32
        tt.reduce.return %20 : i32
      }) : (tensor<4xi32>) -> i32
      %19 = arith.addi %arg3, %18 : i32
      scf.yield %19 : i32
    }
    tt.return
  }
}

