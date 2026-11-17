// RUN: not furiosa-mlir-opt --furiosa-raise-to-linalg %s 2>&1 | FileCheck %s

// A kernel outside the supported subset fails with a diagnostic naming what is unsupported.

// CHECK: error: furiosa: atomics are not raised

// Triton source:
// def atomic_sum(x_ptr, y_ptr, BLOCK: tl.constexpr):
//     tl.atomic_add(y_ptr, tl.sum(tl.load(x_ptr + tl.arange(0, BLOCK)), axis=0))
// launch grid [1, 1, 1], buffers [('fp32', [8]), ('fp32', [1])]

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [8]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [1]}]}"} {
  tt.func public @atomic_sum(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<f32>>
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %6 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<8xf32>) -> f32
    %5 = tt.atomic_rmw fadd, acq_rel, gpu, %arg1, %4, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}

