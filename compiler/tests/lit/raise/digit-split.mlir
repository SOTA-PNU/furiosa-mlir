// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.floordivsi

// A head index read as head // 2 (grouped-query attention) splits the store view's head dim into its
// digits [H / 2, 2], so the read needs no division.

// CHECK-LABEL: func.func @grouped_heads
// CHECK: tensor.empty() : tensor<4x2x2x8xf32>

// Triton source:
// def grouped_heads(x_ptr, y_ptr, H: tl.constexpr, D: tl.constexpr):
//     t = tl.program_id(0)
//     h = tl.program_id(1)
//     d = tl.arange(0, D)
//     v = tl.load(x_ptr + t * (H // 2) * D + (h // 2) * D + d)
//     tl.store(y_ptr + t * H * D + h * D + d, v)
// launch grid [4, 4, 1], buffers [('fp32', [4, 16]), ('fp32', [4, 32])]

module attributes {furiosa.launch = "{\22grid\22: [4, 4, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [4, 16]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [4, 32]}]}"} {
  tt.func public @grouped_heads(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = arith.muli %0, %c16_i32 : i32
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f32>, i32
    %5 = arith.divsi %1, %c2_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = tt.addptr %4, %6 : !tt.ptr<f32>, i32
    %8 = tt.splat %7 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %9 = tt.addptr %8, %2 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %10 = tt.load %9 : tensor<8x!tt.ptr<f32>>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = tt.addptr %arg1, %11 : !tt.ptr<f32>, i32
    %13 = arith.muli %1, %c8_i32 : i32
    %14 = tt.addptr %12, %13 : !tt.ptr<f32>, i32
    %15 = tt.splat %14 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %16 = tt.addptr %15, %2 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %16, %10 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

