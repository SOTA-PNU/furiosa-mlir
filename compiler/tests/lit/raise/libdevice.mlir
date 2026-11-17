// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.extern_elementwise

// Triton tutorial 07: a libdevice call. The declaration names the function (__nv_asinf, the fp32
// variant); with no library to link, the raising pass maps known libdevice functions to the math
// dialect.

// CHECK-LABEL: func.func @asin_kernel
// CHECK: math.asin %{{.*}} : f32

// Triton source: harness/cases/tutorial_extern.py (n = 98432, BLOCK_SIZE = 1024).

module attributes {furiosa.launch = "{\22grid\22: [97, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [98432]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [98432]}]}"} {
  tt.func public @asin_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<98432> : tensor<1024xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = arith.cmpi slt, %4, %cst : tensor<1024xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %7, %5 : tensor<1024x!tt.ptr<f32>>
    %9 = tt.extern_elementwise %8 {libname = "", libpath = "", pure = true, symbol = "__nv_asinf"} : (tensor<1024xf32>) -> tensor<1024xf32>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %11, %9, %5 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

