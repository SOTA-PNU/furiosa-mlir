// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// tl.atomic_max on floats: Triton bitcasts the pointer to i32 and issues a signed max for x >= 0 and
// an unsigned min for x < 0, each masked by the sign of the data. The bitcast pointer keeps the
// address; the write reads the f32 buffer as i32, combines, selects on the data mask, and casts back.

// CHECK-LABEL: func.func @atomic_kernel
// CHECK: linalg.generic {{.*}} outs(%arg1 : tensor<4xf32>)
// CHECK: %[[OLD:.*]] = arith.bitcast %out : f32 to i32
// CHECK: %[[M:.*]] = arith.maxsi %[[OLD]], %{{.*}} : i32
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %[[M]], %[[OLD]] : i32
// CHECK: arith.bitcast %[[S]] : i32 to f32
// CHECK: arith.minui

// Triton source (harness/ops atomic_kernel, OP=1): tl.atomic_max(out_ptr + c, x[row]) on grid (2,).

module attributes {furiosa.launch = "{\22grid\22: [2, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [4]}]}"} {
  tt.func public @atomic_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<4xi32>
    %cst_0 = arith.constant dense<31> : tensor<4xi32>
    %cst_1 = arith.constant dense<true> : tensor<4xi1>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = arith.muli %0, %c4_i32 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %9 = tt.bitcast %6 : tensor<4xf32> -> tensor<4xi32>
    %10 = tt.bitcast %8 : tensor<4x!tt.ptr<f32>> -> tensor<4x!tt.ptr<i32>>
    %11 = arith.shrui %9, %cst_0 : tensor<4xi32>
    %12 = arith.cmpi ne, %11, %cst : tensor<4xi32>
    %13 = arith.xori %12, %cst_1 : tensor<4xi1>
    %14 = tt.atomic_rmw max, acq_rel, gpu, %10, %9, %13 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    %15 = tt.atomic_rmw umin, acq_rel, gpu, %10, %9, %12 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    tt.return
  }
}

