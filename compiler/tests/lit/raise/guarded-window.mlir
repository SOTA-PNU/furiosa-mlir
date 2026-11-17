// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.maxsi --implicit-check-not=arith.minsi

// A load in a loop whose trip count varies by program (program p sums rows p + i for i < 2 - p) leaves its buffer
// only in iterations the shorter trip masks off, where its value is discarded. It reads the window of addresses it
// spans, holding what the clamped address would (the buffer, then its last element repeated: a box copy and a
// broadcast), at an affine address rather than a clamped one: furiosa-structure promotes the read.

// CHECK-LABEL: func.func @window
// CHECK: %[[W0:.*]] = tensor.insert_slice %arg0 into %{{.*}}[0] [32] [1] : tensor<32xf32> into tensor<48xf32>
// CHECK: %[[LAST:.*]] = tensor.extract_slice %arg0[31] [1] [1] : tensor<32xf32> to tensor<1xf32>
// CHECK: %[[REP:.*]] = linalg.generic {{.*}} ins(%[[LAST]] : tensor<1xf32>) outs(%{{.*}} : tensor<16xf32>)
// CHECK: %[[W:.*]] = tensor.insert_slice %[[REP]] into %[[W0]][32] [16] [1] : tensor<16xf32> into tensor<48xf32>
// CHECK: tensor.extract %[[W]][%{{.*}}] : tensor<48xf32>

// Triton source:
// def window(x_ptr, out_ptr):
//     p = tl.program_id(0)
//     offs = tl.arange(0, 16)
//     acc = tl.zeros((16,), dtype=tl.float32)
//     for i in range(0, 2 - p):
//         acc += tl.load(x_ptr + (p + i) * 16 + offs)
//     tl.store(out_ptr + p * 16 + offs, acc)
// launch grid [2, 1, 1], buffers [('fp32', [32]), ('fp32', [32])]

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [32]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [32]}]}"} {
  tt.func public @window(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %p = tt.get_program_id x : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = arith.subi %c2, %p : i32
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %zero) -> (tensor<16xf32>)  : i32 {
      %pi = arith.addi %p, %i : i32
      %off = arith.muli %pi, %c16 : i32
      %base = tt.addptr %x, %off : !tt.ptr<f32>, i32
      %bs = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %ptrs = tt.addptr %bs, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %v = tt.load %ptrs : tensor<16x!tt.ptr<f32>>
      %s = arith.addf %a, %v : tensor<16xf32>
      scf.yield %s : tensor<16xf32>
    }
    %po = arith.muli %p, %c16 : i32
    %ob = tt.addptr %out, %po : !tt.ptr<f32>, i32
    %os = tt.splat %ob : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %optrs = tt.addptr %os, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %optrs, %acc : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
