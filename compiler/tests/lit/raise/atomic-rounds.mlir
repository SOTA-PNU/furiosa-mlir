// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// tl.atomic_add from 3 programs into the same 4 elements: their footprints overlap, so they apply in
// rounds, each a store of old + val that starts from the buffer's current value.

// CHECK-LABEL: func.func @atomic_kernel
// CHECK: %[[R0:.*]] = linalg.generic {{.*}} outs(%arg1 : tensor<4xf32>)
// CHECK: arith.addf %out, %{{.*}} : f32
// CHECK: %[[R1:.*]] = linalg.generic {{.*}} outs(%[[R0]] : tensor<4xf32>)
// CHECK: arith.addf %out
// CHECK: %[[R2:.*]] = linalg.generic {{.*}} outs(%[[R1]] : tensor<4xf32>)
// CHECK: arith.addf %out
// CHECK: return %[[R2]]

// Triton source (harness/ops atomic_kernel, OP=0): tl.atomic_add(out_ptr + c, x[row]) on grid (3,).

module attributes {furiosa.launch = "{\22grid\22: [3, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [3, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [4]}]}"} {
  tt.func public @atomic_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<4xi1>
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
    %9 = tt.atomic_rmw fadd, acq_rel, gpu, %8, %6, %cst : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
    tt.return
  }
}

