// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// Rows of sequence b, block i stored at b * T + i * BLOCK (BERT's attention over [B * T, D] token rows): the
// rows' tile origin mixes the two program ids in mixed radix, so the sequence's per-program length read
// (len[b], b = program y) recovers b as the origin's digit above T, (row floordiv T); the row dim then splits
// into [B, T] and the read is a projection of the sequence dim.

// CHECK-LABEL: func.func @seq_rows
// CHECK: tensor.empty() : tensor<2x8x8xf32>
// CHECK: %[[B:.*]] = linalg.index 0 : index
// CHECK: tensor.extract %{{.*}}[%[[B]]] : tensor<2xi32>

// Triton source:
// def seq_rows(x_ptr, len_ptr, y_ptr, T: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
//     b = tl.program_id(1)
//     rows = b * T + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
//     d = tl.arange(0, D)
//     n = tl.load(len_ptr + b)
//     x = tl.load(x_ptr + rows[:, None] * D + d[None, :])
//     tl.store(y_ptr + rows[:, None] * D + d[None, :], x + n.to(tl.float32))
// launch grid [2, 2, 1], T=8, BLOCK=4, D=8, buffers [('fp32', [16, 8]), ('i32', [2]), ('fp32', [16, 8])]

module attributes {furiosa.launch = "{\22grid\22: [2, 2, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [16, 8]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22i32\22, \22shape\22: [2]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [16, 8]}]}"} {
  tt.func public @seq_rows(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %len_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8v = arith.constant dense<8> : tensor<4x1xi32>
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %b = tt.get_program_id y : i32
    %r0 = arith.muli %b, %c8_i32 : i32
    %i = tt.get_program_id x : i32
    %r1 = arith.muli %i, %c4_i32 : i32
    %r2 = arith.addi %r0, %r1 : i32
    %ar = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %r3 = tt.splat %r2 : i32 -> tensor<4xi32>
    %rows = arith.addi %r3, %ar : tensor<4xi32>
    %d = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %np = tt.addptr %len_ptr, %b : !tt.ptr<i32>, i32
    %n = tt.load %np : !tt.ptr<i32>
    %re = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %ra = arith.muli %re, %c8v : tensor<4x1xi32>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %xr = tt.addptr %xs, %ra : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %de = tt.expand_dims %d {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %xb = tt.broadcast %xr : tensor<4x1x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
    %db = tt.broadcast %de : tensor<1x8xi32> -> tensor<4x8xi32>
    %xp = tt.addptr %xb, %db : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %x = tt.load %xp : tensor<4x8x!tt.ptr<f32>>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %yr = tt.addptr %ys, %ra : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %yb = tt.broadcast %yr : tensor<4x1x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
    %yp = tt.addptr %yb, %db : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %nf = arith.sitofp %n : i32 to f32
    %ns = tt.splat %nf : f32 -> tensor<4x8xf32>
    %y = arith.addf %x, %ns : tensor<4x8xf32>
    tt.store %yp, %y : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }
}
