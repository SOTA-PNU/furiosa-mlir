// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// A min and a max of indices whose ranges overlap (a window's in-range taps) are data: computed from their operands
// where the value is read, here as a divisor. Each resolves in one program and not in the other (min in block 1,
// max in block 0), so their difference is no one expression of the coordinates in every program either.

// CHECK-LABEL: func.func @window_count
// CHECK: %[[MIN:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[MAX:.*]] = arith.maxsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[TAPS:.*]] = arith.subi %[[MIN]], %[[MAX]] : i32
// CHECK: arith.sitofp %[[TAPS]] : i32 to f32
// CHECK: return %{{.*}} : tensor<12xf32>

// Triton source:
// def window_count(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
//     i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
//     taps = tl.minimum(i + 2, N) - tl.maximum(i - 1, 0)
//     x = tl.load(x_ptr + i, mask=i < N, other=0.0)
//     tl.store(y_ptr + i, x / taps.to(tl.float32), mask=i < N)
// launch grid [2, 1, 1], N=12, BLOCK=8, buffers [('fp32', [12]), ('fp32', [12])]

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [12]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [12]}], \22scalars\22: []}"} {
  tt.func public @window_count(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %taps = arith.constant dense<0> : tensor<8xi32>
    %taps_0 = arith.constant dense<1> : tensor<8xi32>
    %cst = arith.constant dense<12> : tensor<8xi32>
    %taps_1 = arith.constant dense<2> : tensor<8xi32>
    %i = arith.constant 8 : i32
    %i_2 = tt.get_program_id x : i32
    %i_3 = arith.muli %i_2, %i : i32
    %i_4 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %i_5 = tt.splat %i_3 : i32 -> tensor<8xi32>
    %i_6 = arith.addi %i_5, %i_4 : tensor<8xi32>
    %taps_7 = arith.addi %i_6, %taps_1 : tensor<8xi32>
    %taps_8 = arith.minsi %taps_7, %cst : tensor<8xi32>
    %taps_9 = arith.subi %i_6, %taps_0 : tensor<8xi32>
    %taps_10 = arith.maxsi %taps_9, %taps : tensor<8xi32>
    %taps_11 = arith.subi %taps_8, %taps_10 : tensor<8xi32>
    %x_12 = arith.cmpi slt, %i_6, %cst : tensor<8xi32>
    %x_13 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %x_14 = tt.addptr %x_13, %i_6 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %x_15 = tt.load %x_14, %x_12, %x : tensor<8x!tt.ptr<f32>>
    %0 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %i_6 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %2 = arith.sitofp %taps_11 : tensor<8xi32> to tensor<8xf32>
    %3 = arith.divf %x_15, %2 : tensor<8xf32>
    tt.store %1, %3, %x_12 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
