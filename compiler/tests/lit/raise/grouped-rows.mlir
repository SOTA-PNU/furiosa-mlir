// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// Rows that pack two program ids' tiles (grouped-query attention's t * GROUP + j over [T, heads], t's block from
// program x and j's group from program y) store along one digit axis whose tile origin mixes both ids, so no
// origin depends on program x alone. The tile origin of the view dim its block moves along does: program x is
// recovered from that dim (t floordiv BLOCK_T), which then splits into its digits.

// CHECK-LABEL: func.func @grouped_rows
// CHECK: tensor.empty() : tensor<2x2x2x2x8xf32>
// CHECK: %[[PID:.*]] = linalg.index 0 : index
// CHECK: %[[I:.*]] = arith.index_cast %[[PID]] : index to i32
// CHECK: %[[F:.*]] = arith.sitofp %[[I]] : i32 to f32
// CHECK: arith.addf %{{.*}}, %[[F]] : f32

// Triton source:
// def grouped_rows(x_ptr, y_ptr, BLOCK_T: tl.constexpr, GROUP: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr):
//     r = tl.arange(0, BLOCK_T * GROUP)
//     t = tl.program_id(0) * BLOCK_T + r // GROUP
//     head = tl.program_id(1) * GROUP + r % GROUP
//     d = tl.arange(0, D)
//     rows = t[:, None] * (HEADS * D) + head[:, None] * D + d[None, :]
//     x = tl.load(x_ptr + rows)
//     tl.store(y_ptr + rows, x + tl.program_id(0).to(tl.float32))
// launch grid [2, 2, 1], BLOCK_T=2, GROUP=2, HEADS=4, D=8, buffers [('fp32', [4, 32]), ('fp32', [4, 32])]

module attributes {furiosa.launch = "{\22grid\22: [2, 2, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [4, 32]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [4, 32]}]}"} {
  tt.func public @grouped_rows(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8 = arith.constant dense<8> : tensor<4x1xi32>
    %c32 = arith.constant dense<32> : tensor<4x1xi32>
    %c2v = arith.constant dense<2> : tensor<4xi32>
    %c2_i32 = arith.constant 2 : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %px = tt.get_program_id x : i32
    %t0 = arith.muli %px, %c2_i32 : i32
    %tq = arith.divsi %r, %c2v : tensor<4xi32>
    %t0s = tt.splat %t0 : i32 -> tensor<4xi32>
    %t = arith.addi %t0s, %tq : tensor<4xi32>
    %py = tt.get_program_id y : i32
    %h0 = arith.muli %py, %c2_i32 : i32
    %hr = arith.remsi %r, %c2v : tensor<4xi32>
    %h0s = tt.splat %h0 : i32 -> tensor<4xi32>
    %head = arith.addi %h0s, %hr : tensor<4xi32>
    %d = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %te = tt.expand_dims %t {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %ta = arith.muli %te, %c32 : tensor<4x1xi32>
    %he = tt.expand_dims %head {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %ha = arith.muli %he, %c8 : tensor<4x1xi32>
    %row = arith.addi %ta, %ha : tensor<4x1xi32>
    %de = tt.expand_dims %d {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %rowb = tt.broadcast %row : tensor<4x1xi32> -> tensor<4x8xi32>
    %deb = tt.broadcast %de : tensor<1x8xi32> -> tensor<4x8xi32>
    %rows = arith.addi %rowb, %deb : tensor<4x8xi32>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %rows : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %x = tt.load %xp : tensor<4x8x!tt.ptr<f32>>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %yp = tt.addptr %ys, %rows : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %pf = arith.sitofp %px : i32 to f32
    %pfs = tt.splat %pf : f32 -> tensor<4x8xf32>
    %y = arith.addf %x, %pfs : tensor<4x8xf32>
    tt.store %yp, %y : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }
}
