// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.scan --implicit-check-not='xi1>'

// tl.cumsum: a scan becomes a reduction over a new axis j next to the scanned axis i. The combiner
// (addf) has an identity, so the reduction adds x_j where j <= i and 0 elsewhere: one accumulator.

// CHECK-LABEL: func.func @scan_kernel
// CHECK: %[[INIT:.*]] = linalg.generic {{.*}} outs({{.*}} : tensor<64x64xf32>)
// CHECK: linalg.yield %cst : f32
// CHECK: linalg.generic {{.*}}["parallel", "parallel", "reduction"]{{.*}} ins({{.*}} : tensor<64xi8>) outs(%[[INIT]] : tensor<64x64xf32>)
// CHECK: %[[IN:.*]] = arith.cmpi sge
// CHECK: %[[X:.*]] = arith.select %{{.*}}, %extracted, %cst : f32
// CHECK: arith.addf %out, %[[X]] : f32

// Triton source (harness/ops scan_kernel, OP=0): y = tl.cumsum(x, axis=1) on a 64x64 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [64, 64]}]}"} {
  tt.func public @scan_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %3 = arith.cmpi slt, %1, %cst_0 : tensor<64x1xi32>
    %4 = arith.muli %1, %cst_0 : tensor<64x1xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %7 = tt.broadcast %6 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
    %8 = tt.broadcast %2 : tensor<1x64xi32> -> tensor<64x64xi32>
    %9 = tt.addptr %7, %8 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %10 = tt.broadcast %3 : tensor<64x1xi1> -> tensor<64x64xi1>
    %11 = tt.load %9, %10, %cst : tensor<64x64x!tt.ptr<f32>>
    %12 = "tt.scan"(%11) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %17 = arith.addf %arg2, %arg3 : f32
      tt.scan.return %17 : f32
    }) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %14 = tt.addptr %13, %4 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %15 = tt.broadcast %14 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
    %16 = tt.addptr %15, %8 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    tt.store %16, %12, %10 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}

