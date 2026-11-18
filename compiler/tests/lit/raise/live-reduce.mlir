// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.select --implicit-check-not=arith.minsi

// A reduction over a padded tile (8 lanes, masked to the row's 5 with the combiner's identity as `other`, here
// through a product) reduces over the live prefix: 5 lanes, where the mask always holds and the address stays in
// the buffer, so neither a select nor a clamp is left.

// CHECK-LABEL: func.func @row_energy
// CHECK: tensor.empty() : tensor<5xi8>
// CHECK: linalg.generic {{.*}}["parallel", "reduction"]{{.*}} ins(%{{.*}} : tensor<5xi8>)

// Triton source:
// def row_energy(x_ptr, out_ptr):
//     r = tl.program_id(0)
//     c = tl.arange(0, 8)
//     v = tl.load(x_ptr + r * 5 + c, mask=c < 5, other=0.0)
//     tl.store(out_ptr + r, tl.sum(v * v, axis=0))
// launch grid [4, 1, 1], buffers [('fp32', [4, 5]), ('fp32', [4])]

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [4, 5]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [4]}]}"} {
  tt.func public @row_energy(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c5 = arith.constant 5 : i32
    %five = arith.constant dense<5> : tensor<8xi32>
    %r = tt.get_program_id x : i32
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %m = arith.cmpi slt, %c, %five : tensor<8xi32>
    %base = arith.muli %r, %c5 : i32
    %bp = tt.addptr %x, %base : !tt.ptr<f32>, i32
    %xs = tt.splat %bp : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %c : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %xp, %m, %zero : tensor<8x!tt.ptr<f32>>
    %sq = arith.mulf %v, %v : tensor<8xf32>
    %s = "tt.reduce"(%sq) <{axis = 0 : i32}> ({
    ^bb0(%l: f32, %rr: f32):
      %t = arith.addf %l, %rr : f32
      tt.reduce.return %t : f32
    }) : (tensor<8xf32>) -> f32
    %op = tt.addptr %out, %r : !tt.ptr<f32>, i32
    tt.store %op, %s : !tt.ptr<f32>
    tt.return
  }
}
