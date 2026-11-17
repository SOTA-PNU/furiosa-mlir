// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not='tensor<16x16'

// A loop carrying (best, index) state is kept as scf.for; its states span only the lanes they depend
// on, not the reduction dims of the generics that consume them.

// CHECK-LABEL: func.func @first_argmax
// CHECK: scf.for {{.*}} -> (tensor<16xf32>, tensor<16xi32>)

// Triton source:
// def first_argmax(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
//     offs = tl.arange(0, BLOCK)
//     best = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
//     best_i = tl.zeros((BLOCK,), dtype=tl.int32)
//     for n0 in range(0, N, BLOCK):
//         v = tl.load(x_ptr + n0 + offs, mask=n0 + offs < N, other=float("-inf"))
//         better = v > best
//         best = tl.where(better, v, best)
//         best_i = tl.where(better, n0 + offs, best_i)
//     top = tl.max(best, axis=0)
//     tl.store(out_ptr, tl.min(tl.where(best == top, best_i, N), axis=0))
// launch grid [1, 1, 1], buffers [('fp32', [1, 64]), ('i32', [1])]

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 64]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22i32\22, \22shape\22: [1]}]}"} {
  tt.func public @first_argmax(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<16xi32>
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<64> : tensor<16xi32>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c16_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst) -> (tensor<16xf32>, tensor<16xi32>)  : i32 {
      %7 = tt.splat %arg2 : i32 -> tensor<16xi32>
      %8 = arith.addi %7, %0 : tensor<16xi32>
      %9 = arith.cmpi slt, %8, %cst_0 : tensor<16xi32>
      %10 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      %11 = tt.splat %10 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %12 = tt.addptr %11, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %13 = tt.load %12, %9, %cst_1 : tensor<16x!tt.ptr<f32>>
      %14 = arith.cmpf ogt, %13, %arg3 : tensor<16xf32>
      %15 = arith.select %14, %13, %arg3 : tensor<16xi1>, tensor<16xf32>
      %16 = arith.select %14, %8, %arg4 : tensor<16xi1>, tensor<16xi32>
      scf.yield %15, %16 : tensor<16xf32>, tensor<16xi32>
    }
    %2 = "tt.reduce"(%1#0) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %7 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16xf32>) -> f32
    %3 = tt.splat %2 : f32 -> tensor<16xf32>
    %4 = arith.cmpf oeq, %1#0, %3 : tensor<16xf32>
    %5 = arith.select %4, %1#1, %cst_0 : tensor<16xi1>, tensor<16xi32>
    %6 = "tt.reduce"(%5) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %7 = arith.minsi %arg2, %arg3 : i32
      tt.reduce.return %7 : i32
    }) : (tensor<16xi32>) -> i32
    tt.store %arg1, %6 : !tt.ptr<i32>
    tt.return
  }
}

