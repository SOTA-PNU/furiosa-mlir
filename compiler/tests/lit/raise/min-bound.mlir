// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// A loop over a band of rows up to min(start + 3, M): its trip (3, 3, 1 for programs 0..2) is not affine in the
// program id, but the distance to each term of the min is (3, and 7 - 3 * pid), so an iteration runs where it is
// below every term that some iteration can reach (here 7 - 3 * pid alone). A masked load inside that leaves its
// buffer only in iterations past the trip reads `other` there too: its mask gains the same bound, so it bounds
// every index (furiosa-structure then promotes the read to a padded buffer).

// CHECK-LABEL: func.func @band_sum
// The per-row max over (program, iteration, lane): the lane mask and the trip bound.
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<3x3xf32>)
// CHECK: %[[LANE:.*]] = arith.cmpi sge
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[B:.*]] = arith.addi %{{.*}}, %[[C6]] : index
// CHECK: %[[TRIP:.*]] = arith.cmpi sge, %[[B]]
// CHECK: %[[MASK:.*]] = arith.andi %[[LANE]], %[[TRIP]] : i1
// CHECK: arith.select %[[MASK]]
// The band sum: iterations past 7 - 3 * pid leave the accumulator unchanged.
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<3x4xf32>)
// CHECK: %[[C7:.*]] = arith.constant 7 : index
// CHECK: %[[END:.*]] = arith.addi %{{.*}}, %[[C7]] : index
// CHECK: %[[RUNS:.*]] = arith.cmpi slt, %{{.*}}, %[[END]] : index
// CHECK-NOT: arith.cmpi
// CHECK: arith.select %[[RUNS]]

// Triton source:
// def band_sum(x_ptr, out_ptr):
//     pid = tl.program_id(0)
//     cols = tl.arange(0, 8)
//     acc = tl.zeros((8,), dtype=tl.float32)
//     for row in range(pid * 3, tl.minimum(pid * 3 + 3, 7)):
//         v = tl.load(x_ptr + row * 4 + cols, mask=cols < 4, other=0.0)
//         acc += v * tl.max(v, axis=0)
//     tl.store(out_ptr + pid * 4 + cols, acc, mask=cols < 4)
// launch grid [3, 1, 1], buffers [('fp32', [7, 4]), ('fp32', [3, 4])]

module attributes {furiosa.launch = "{\22grid\22: [3, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [7, 4]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [3, 4]}]}"} {
  tt.func public @band_sum(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c7 = arith.constant 7 : i32
    %n = arith.constant dense<4> : tensor<8xi32>
    %pid = tt.get_program_id x : i32
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %mask = arith.cmpi slt, %cols, %n : tensor<8xi32>
    %start = arith.muli %pid, %c3 : i32
    %next = arith.addi %start, %c3 : i32
    %end = arith.minsi %next, %c7 : i32
    %acc = scf.for %row = %start to %end step %c1 iter_args(%a = %zero) -> (tensor<8xf32>)  : i32 {
      %off = arith.muli %row, %c4 : i32
      %base = tt.addptr %x, %off : !tt.ptr<f32>, i32
      %bs = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %ptrs = tt.addptr %bs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %v = tt.load %ptrs, %mask, %zero : tensor<8x!tt.ptr<f32>>
      %ss = "tt.reduce"(%v) <{axis = 0 : i32}> ({
      ^bb0(%l: f32, %r: f32):
        %t = arith.maxnumf %l, %r : f32
        tt.reduce.return %t : f32
      }) : (tensor<8xf32>) -> f32
      %sp = tt.splat %ss : f32 -> tensor<8xf32>
      %w = arith.mulf %v, %sp : tensor<8xf32>
      %s = arith.addf %a, %w : tensor<8xf32>
      scf.yield %s : tensor<8xf32>
    }
    %po = arith.muli %pid, %c4 : i32
    %ob = tt.addptr %out, %po : !tt.ptr<f32>, i32
    %os = tt.splat %ob : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %optrs = tt.addptr %os, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %optrs, %acc, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
