// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.reshape

// tl.flip on 2x4 rows: Triton reshapes each row to 2x2, xor-reduces over each 2-axis, and reshapes
// back, moving data between dims. Each source axis of a reshape is a mixed-radix digit of the
// result's linear index: the row stays the row (no floordiv by 4), the 2x2 digits of c become
// c floordiv 2 and c mod 2.

// CHECK-LABEL: func.func @scan_kernel
// CHECK: linalg.generic {{.*}}["parallel", "parallel", "reduction"]
// CHECK: arith.xori
// CHECK-NOT: arith.floordivsi {{.*}}, %c4
// CHECK: return

// Triton source (harness/ops scan_kernel, OP=2): y = tl.flip(x, dim=1) on a 2x4 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 4]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 4]}]}"} {
  tt.func public @scan_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %cst_0 = arith.constant dense<4> : tensor<2x1xi32>
    %cst_1 = arith.constant dense<2> : tensor<2x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = arith.cmpi slt, %1, %cst_1 : tensor<2x1xi32>
    %5 = arith.muli %1, %cst_0 : tensor<2x1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %7 = tt.addptr %6, %5 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %8 = tt.broadcast %7 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %9 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<2x4xi32>
    %10 = tt.addptr %8, %9 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %11 = tt.broadcast %4 : tensor<2x1xi1> -> tensor<2x4xi1>
    %12 = tt.load %10, %11, %cst : tensor<2x4x!tt.ptr<f32>>
    %13 = tt.bitcast %12 : tensor<2x4xf32> -> tensor<2x4xi32>
    %14 = tt.reshape %13 : tensor<2x4xi32> -> tensor<2x2x2xi32>
    %15 = "tt.reduce"(%14) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %29 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %29 : i32
    }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<2x2xi32> -> tensor<2x1x2xi32>
    %17 = tt.broadcast %16 : tensor<2x1x2xi32> -> tensor<2x2x2xi32>
    %18 = arith.xori %14, %17 : tensor<2x2x2xi32>
    %19 = "tt.reduce"(%18) <{axis = 2 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %29 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %29 : i32
    }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
    %20 = tt.expand_dims %19 {axis = 2 : i32} : tensor<2x2xi32> -> tensor<2x2x1xi32>
    %21 = tt.broadcast %20 : tensor<2x2x1xi32> -> tensor<2x2x2xi32>
    %22 = arith.xori %18, %21 : tensor<2x2x2xi32>
    %23 = tt.reshape %22 : tensor<2x2x2xi32> -> tensor<2x4xi32>
    %24 = tt.bitcast %23 : tensor<2x4xi32> -> tensor<2x4xf32>
    %25 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %26 = tt.addptr %25, %5 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %27 = tt.broadcast %26 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %28 = tt.addptr %27, %9 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %28, %24, %11 : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

