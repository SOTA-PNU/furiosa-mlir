// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// A loop whose bounds are data (a segment's rows, from off[s] to off[s + 1], by 4) runs by 1 over every counter value
// u its load can reach in x[20] (u + lane in [0, 20) for a lane in [0, 4): u in [-3, 19], 23 iterations), each kept
// where off[s] <= u < off[s + 1] and (u - off[s]) rem 4 == 0. Past those values the load is masked off in a valid
// program and yields 0.0, the identity of the accumulation, so they leave the sum as it is.

// CHECK-LABEL: func.func @ranges_sum
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}} : tensor<23xi8>)
// CHECK: %[[END:.*]] = tensor.extract %arg1[%{{.*}}] : tensor<4xi32>
// CHECK: %[[LIVE:.*]] = arith.cmpi slt, %{{.*}}, %[[END]] : i32
// CHECK: %[[X:.*]] = tensor.extract %arg0[%{{.*}}] : tensor<20xf32>
// CHECK: arith.select %[[LIVE]], %[[X]], %{{.*}} : f32
// CHECK: %[[START:.*]] = tensor.extract %arg1[%{{.*}}] : tensor<4xi32>
// CHECK: %[[LO:.*]] = arith.index_cast %[[START]] : i32 to index
// CHECK: %[[HI:.*]] = arith.index_cast %[[END]] : i32 to index
// CHECK: arith.cmpi slt, %[[U:.*]], %[[HI]] : index
// CHECK: arith.cmpi sge, %[[U]], %[[LO]] : index
// CHECK: %[[PAST:.*]] = arith.subi %[[U]], %[[LO]] : index
// CHECK: arith.remsi %[[PAST]], %c4 : index
// CHECK: return %{{.*}} : tensor<3xf32>

// Triton source:
// def ranges_sum(x_ptr, off_ptr, y_ptr, BLOCK: tl.constexpr):
//     s = tl.program_id(0)
//     start = tl.load(off_ptr + s)
//     end = tl.load(off_ptr + s + 1)
//     acc = tl.zeros((BLOCK,), tl.float32)
//     for r0 in range(start, end, BLOCK):
//         rows = r0 + tl.arange(0, BLOCK)
//         acc += tl.load(x_ptr + rows, mask=rows < end, other=0.0)
//     tl.store(y_ptr + s, tl.sum(acc))
// launch grid [3, 1, 1], BLOCK=4, buffers [('fp32', [20]), ('i32', [4]), ('fp32', [3])]

module attributes {furiosa.launch = "{\22grid\22: [3, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [20]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [4]}, {\22arg\22: 2, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [3]}], \22scalars\22: []}"} {
  tt.func public @ranges_sum(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %off_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %end = arith.constant 1 : i32
    %s = tt.get_program_id x : i32
    %start = tt.addptr %off_ptr, %s : !tt.ptr<i32>, i32
    %start_0 = tt.load %start : !tt.ptr<i32>
    %end_1 = tt.addptr %start, %end : !tt.ptr<i32>, i32
    %end_2 = tt.load %end_1 : !tt.ptr<i32>
    %acc = scf.for %r0 = %start_0 to %end_2 step %c4_i32 iter_args(%acc_3 = %cst) -> (tensor<4xf32>)  : i32 {
      %rows = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %rows_4 = tt.splat %r0 : i32 -> tensor<4xi32>
      %rows_5 = arith.addi %rows_4, %rows : tensor<4xi32>
      %acc_6 = tt.splat %end_2 : i32 -> tensor<4xi32>
      %acc_7 = arith.cmpi slt, %rows_5, %acc_6 : tensor<4xi32>
      %acc_8 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %acc_9 = tt.addptr %acc_8, %rows_5 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %acc_10 = tt.load %acc_9, %acc_7, %cst : tensor<4x!tt.ptr<f32>>
      %acc_11 = arith.addf %acc_3, %acc_10 : tensor<4xf32>
      scf.yield %acc_11 : tensor<4xf32>
    }
    %0 = tt.addptr %y_ptr, %s : !tt.ptr<f32>, i32
    %1 = tt.reshape %acc allow_reorder : tensor<4xf32> -> tensor<4xf32>
    %2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %3 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<4xf32>) -> f32
    tt.store %0, %2 : !tt.ptr<f32>
    tt.return
  }
}
