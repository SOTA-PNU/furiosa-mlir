// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// Tiles of 4 lanes masked to their first 3 (a group's channels in a power-of-two block) start every 3 elements:
// their origins are not aligned to the block, but are to the live prefix, so the origin of a stored element is
// recovered modulo 3. A value of the whole tile (the group's sum) depends on that origin, so the store view splits
// the buffer into [group, lane].

// CHECK-LABEL: func.func @group_center
// CHECK: %[[SUM:.*]] = linalg.generic {{.*}}["parallel", "reduction"]{{.*}} outs(%{{.*}} : tensor<4xf32>)
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<4x3xf32>)
// CHECK: tensor.extract %[[SUM]][%{{.*}}] : tensor<4xf32>
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]] : tensor<4x3xf32> into tensor<12xf32>

// Triton source:
// def group_center(x_ptr, out_ptr):
//     g = tl.program_id(0)
//     ci = tl.arange(0, 4)
//     x = tl.load(x_ptr + g * 3 + ci, mask=ci < 3, other=0.0)
//     tl.store(out_ptr + g * 3 + ci, x - tl.sum(x, axis=0), mask=ci < 3)
// launch grid [4, 1, 1], buffers [('fp32', [12]), ('fp32', [12])]

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [12]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [12]}]}"} {
  tt.func public @group_center(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %c3 = arith.constant 3 : i32
    %three = arith.constant dense<3> : tensor<4xi32>
    %g = tt.get_program_id x : i32
    %ci = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %m = arith.cmpi slt, %ci, %three : tensor<4xi32>
    %base = arith.muli %g, %c3 : i32
    %bs = tt.splat %base : i32 -> tensor<4xi32>
    %c = arith.addi %bs, %ci : tensor<4xi32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %c : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %xp, %m, %zero : tensor<4x!tt.ptr<f32>>
    %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%l: f32, %r: f32):
      %t = arith.addf %l, %r : f32
      tt.reduce.return %t : f32
    }) : (tensor<4xf32>) -> f32
    %sp = tt.splat %s : f32 -> tensor<4xf32>
    %y = arith.subf %v, %sp : tensor<4xf32>
    %os = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %op = tt.addptr %os, %c : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %op, %y, %m : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
