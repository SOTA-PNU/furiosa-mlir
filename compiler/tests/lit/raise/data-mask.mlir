// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// tl.store(out, x, mask=m & (x > 0)): the mask's index part (m) bounds the written box, its data part
// (x > 0) selects between the new and the old value, so the store starts from the buffer.

// CHECK-LABEL: func.func @masked_store_kernel
// CHECK: linalg.generic {{.*}} outs(%arg1 : tensor<2x4xf32>)
// CHECK: %[[X:.*]] = tensor.extract %{{.*}} : tensor<8xf32>
// CHECK: %[[POS:.*]] = arith.cmpf ogt, %[[X]]
// CHECK: arith.select %[[POS]], %[[X]], %out : f32

// Triton source (harness/ops masked_store_kernel) on a 2x4 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 4]}]}"} {
  tt.func public @masked_store_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %cst_0 = arith.constant dense<4> : tensor<2x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<1x4xi32>
    %cst_2 = arith.constant dense<2> : tensor<2x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = arith.cmpi slt, %1, %cst_2 : tensor<2x1xi32>
    %5 = arith.cmpi slt, %3, %cst_1 : tensor<1x4xi32>
    %6 = tt.broadcast %4 : tensor<2x1xi1> -> tensor<2x4xi1>
    %7 = tt.broadcast %5 : tensor<1x4xi1> -> tensor<2x4xi1>
    %8 = arith.andi %6, %7 : tensor<2x4xi1>
    %9 = arith.muli %1, %cst_0 : tensor<2x1xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %12 = tt.broadcast %11 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %13 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<2x4xi32>
    %14 = tt.addptr %12, %13 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %15 = tt.load %14, %8, %cst : tensor<2x4x!tt.ptr<f32>>
    %16 = arith.cmpf ogt, %15, %cst : tensor<2x4xf32>
    %17 = arith.andi %8, %16 : tensor<2x4xi1>
    %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %9 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %20 = tt.broadcast %19 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %13 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %21, %15, %17 : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

