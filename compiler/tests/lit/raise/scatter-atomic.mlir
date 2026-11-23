// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// An atomic whose address adds data (out + ids[t] * D + d): in each program, the data selects the row of [5, 4] and
// the index part the column, so each element folds, from its old value, the program's token lanes whose row lands
// on it (ids[t] * 4 == row * 4, and the mask). The two programs write the same columns, so they are two rounds.

// CHECK-LABEL: func.func @scatter_rows
// CHECK: %[[OLD:.*]] = linalg.generic {{.*}} outs(%{{.*}} : tensor<5x4xf32>)
// CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<20xf32>
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}} : tensor<4xi8>) outs(%[[OLD]] : tensor<5x4xf32>)
// CHECK: %[[ID:.*]] = tensor.extract %arg0[%{{.*}}] : tensor<6xi32>
// CHECK: %[[DATA:.*]] = arith.muli %[[ID]], %{{.*}} : i32
// CHECK: %[[DELTA:.*]] = arith.index_cast %[[DATA]] : i32 to index
// CHECK: %[[HIT:.*]] = arith.cmpi eq, %[[DELTA]], %{{.*}} : index
// CHECK: %[[ON:.*]] = arith.andi %[[HIT]], %{{.*}} : i1
// CHECK: %[[ADD:.*]] = arith.addf %out, %{{.*}} : f32
// CHECK: arith.select %[[ON]], %[[ADD]], %out : f32
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}} : tensor<4xi8>)
// CHECK: arith.cmpi eq
// CHECK: return %{{.*}} : tensor<5x4xf32>

// Triton source:
// def scatter_rows(ids_ptr, x_ptr, out_ptr, T, D, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
//     t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
//     d = tl.arange(0, BLOCK_D)
//     m = (t < T)[:, None] & (d < D)[None, :]
//     rows = tl.load(ids_ptr + t, mask=t < T, other=0)
//     x = tl.load(x_ptr + t[:, None] * D + d[None, :], mask=m, other=0.0)
//     tl.atomic_add(out_ptr + rows[:, None] * D + d[None, :], x, mask=m)
// launch grid [2, 1, 1], T=6, D=4, BLOCK_T=4, BLOCK_D=4, buffers [('i32', [6]), ('fp32', [6, 4]), ('fp32', [5, 4])]

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [6]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [6, 4]}, {\22arg\22: 2, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [5, 4]}], \22scalars\22: []}"} {
  tt.func public @scatter_rows(%ids_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst = arith.constant dense<4> : tensor<4x1xi32>
    %rows = arith.constant dense<0> : tensor<4xi32>
    %m = arith.constant dense<4> : tensor<4xi32>
    %cst_0 = arith.constant dense<6> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %t = tt.get_program_id x : i32
    %t_1 = arith.muli %t, %c4_i32 : i32
    %t_2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %t_3 = tt.splat %t_1 : i32 -> tensor<4xi32>
    %t_4 = arith.addi %t_3, %t_2 : tensor<4xi32>
    %m_5 = arith.cmpi slt, %t_4, %cst_0 : tensor<4xi32>
    %m_6 = tt.expand_dims %m_5 {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %m_7 = arith.cmpi slt, %t_2, %m : tensor<4xi32>
    %m_8 = tt.expand_dims %m_7 {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %m_9 = tt.broadcast %m_6 : tensor<4x1xi1> -> tensor<4x4xi1>
    %m_10 = tt.broadcast %m_8 : tensor<1x4xi1> -> tensor<4x4xi1>
    %m_11 = arith.andi %m_9, %m_10 : tensor<4x4xi1>
    %rows_12 = tt.splat %ids_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %rows_13 = tt.addptr %rows_12, %t_4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %rows_14 = tt.load %rows_13, %m_5, %rows : tensor<4x!tt.ptr<i32>>
    %x_15 = tt.expand_dims %t_4 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %x_16 = arith.muli %x_15, %cst : tensor<4x1xi32>
    %x_17 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %x_18 = tt.addptr %x_17, %x_16 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %x_19 = tt.expand_dims %t_2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %x_20 = tt.broadcast %x_18 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %x_21 = tt.broadcast %x_19 : tensor<1x4xi32> -> tensor<4x4xi32>
    %x_22 = tt.addptr %x_20, %x_21 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %x_23 = tt.load %x_22, %m_11, %x : tensor<4x4x!tt.ptr<f32>>
    %0 = tt.expand_dims %rows_14 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %1 = arith.muli %0, %cst : tensor<4x1xi32>
    %2 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %4 = tt.broadcast %3 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %x_21 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %x_23, %m_11 : (tensor<4x4x!tt.ptr<f32>>, tensor<4x4xf32>, tensor<4x4xi1>) -> tensor<4x4xf32>
    tt.return
  }
}
