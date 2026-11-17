// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// tl.argmax: a two-operand reduction (value, index) whose combiner has no identity. Element 0 seeds
// both accumulators in a parallel generic, and the reduction runs over elements 1..63 (index + 1).

// CHECK-LABEL: func.func @reduce_kernel
// CHECK: %[[SEED:.*]]:2 = linalg.generic {{.*}}["parallel"]{{.*}} outs({{.*}} : tensor<14xf32>, tensor<14xi32>)
// CHECK: linalg.generic {{.*}}["parallel", "reduction"]{{.*}} ins({{.*}} : tensor<63xi8>) outs(%[[SEED]]#0, %[[SEED]]#1 : tensor<14xf32>, tensor<14xi32>)
// CHECK: %[[J:.*]] = arith.addi %{{.*}}, %c1
// CHECK: arith.index_cast %[[J]] : index to i32
// CHECK: linalg.yield %{{.*}}, %{{.*}} : f32, i32

// Triton source (harness/ops reduce_kernel, OP=3): y = tl.argmax(x, axis=1), launch grid [1, 1, 1],
// buffers [('fp32', [14, 33]), ('i32', [14])], BR=16, BC=64.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [14, 33]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22i32\22, \22shape\22: [14]}]}"} {
  tt.func public @reduce_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<14> : tensor<16xi32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<16x64xf32>
    %cst_1 = arith.constant dense<33> : tensor<16x1xi32>
    %cst_2 = arith.constant dense<33> : tensor<1x64xi32>
    %cst_3 = arith.constant dense<14> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = arith.cmpi slt, %1, %cst_3 : tensor<16x1xi32>
    %5 = arith.cmpi slt, %3, %cst_2 : tensor<1x64xi32>
    %6 = tt.broadcast %4 : tensor<16x1xi1> -> tensor<16x64xi1>
    %7 = tt.broadcast %5 : tensor<1x64xi1> -> tensor<16x64xi1>
    %8 = arith.andi %6, %7 : tensor<16x64xi1>
    %9 = arith.muli %1, %cst_1 : tensor<16x1xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %12 = tt.broadcast %11 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %13 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<16x64xi32>
    %14 = tt.addptr %12, %13 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %15 = tt.load %14, %8, %cst_0 : tensor<16x64x!tt.ptr<f32>>
    %16:2 = "tt.reduce"(%15, %13) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: i32):
      %20 = arith.cmpf oeq, %arg2, %arg4 : f32
      %21 = arith.cmpi slt, %arg3, %arg5 : i32
      %22 = arith.andi %20, %21 : i1
      %23 = arith.cmpf ogt, %arg2, %arg4 : f32
      %24 = arith.ori %23, %22 : i1
      %25 = arith.select %24, %arg2, %arg4 : f32
      %26 = arith.select %24, %arg3, %arg5 : i32
      tt.reduce.return %25, %26 : f32, i32
    }) : (tensor<16x64xf32>, tensor<16x64xi32>) -> (tensor<16xf32>, tensor<16xi32>)
    %17 = arith.cmpi slt, %0, %cst : tensor<16xi32>
    %18 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %19, %16#1, %17 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

