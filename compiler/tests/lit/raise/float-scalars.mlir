// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=1.000000e-03

// A float scalar argument stays an fp32 argument, as Triton's JIT passes it, and the launch carries its value: the
// raise binds it and folds what that fixes in fp32, so 1 - beta2 at beta2 = 0.999 is 1 - 0.999f = 9.99987125E-4,
// not 1e-3 (what a constexpr would fold in double precision). The argument leaves the function.

// CHECK-LABEL: func.func @decay(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32>
// CHECK: arith.constant 9.99987125E-4 : f32
// CHECK: arith.mulf

// Triton source:
// def decay(x_ptr, out_ptr, beta2):
//     offs = tl.arange(0, 8)
//     tl.store(out_ptr + offs, (1.0 - beta2) * tl.load(x_ptr + offs))
// launch grid [1, 1, 1], buffers [('fp32', [8]), ('fp32', [8])], scalars [0.999]

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [8]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [8]}], \22scalars\22: [0.999]}"} {
  tt.func public @decay(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %beta2: f32) attributes {noinline = false} {
    %one = arith.constant 1.000000e+00 : f32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<f32>>
    %c = arith.subf %one, %beta2 : f32
    %cs = tt.splat %c : f32 -> tensor<8xf32>
    %y = arith.mulf %cs, %v : tensor<8xf32>
    %os = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %op = tt.addptr %os, %offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %op, %y : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
