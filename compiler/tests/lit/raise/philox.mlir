// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.mulhiui

// Triton tutorial 04's seeded dropout: tl.rand runs Philox on the element offsets. Its math.umulhi
// (tt.mulhiui) raises to the high half of arith.mului_extended, inside the store's generic.

// CHECK-LABEL: func.func @_seeded_dropout
// CHECK: linalg.generic
// CHECK: %{{.*}}, %[[HI:.*]] = arith.mului_extended
// CHECK: arith.xori %[[HI]]

// Triton source: harness/cases/tutorial_dropout.py (n = 98432, BLOCK_SIZE = 1024, p = 0.5, seed = 123).

module attributes {furiosa.launch = "{\22grid\22: [97, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp32\22, \22shape\22: [98432]}, {\22arg\22: 1, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [98432]}]}"} {
  tt.func public @_seeded_dropout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-766435501_i32 = arith.constant -766435501 : i32
    %c123_i32 = arith.constant 123 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-845247145_i32 = arith.constant -845247145 : i32
    %cst = arith.constant dense<-845247145> : tensor<1024xi32>
    %cst_0 = arith.constant dense<-766435501> : tensor<1024xi32>
    %cst_1 = arith.constant dense<-1640531404> : tensor<1024xi32>
    %cst_2 = arith.constant dense<-1150833019> : tensor<1024xi32>
    %cst_3 = arith.constant dense<1013904365> : tensor<1024xi32>
    %cst_4 = arith.constant dense<1993301258> : tensor<1024xi32>
    %cst_5 = arith.constant dense<-626627162> : tensor<1024xi32>
    %cst_6 = arith.constant dense<842468239> : tensor<1024xi32>
    %cst_7 = arith.constant dense<2027808607> : tensor<1024xi32>
    %cst_8 = arith.constant dense<-308364780> : tensor<1024xi32>
    %cst_9 = arith.constant dense<387277080> : tensor<1024xi32>
    %cst_10 = arith.constant dense<-1459197799> : tensor<1024xi32>
    %cst_11 = arith.constant dense<-1253254447> : tensor<1024xi32>
    %cst_12 = arith.constant dense<1684936478> : tensor<1024xi32>
    %cst_13 = arith.constant dense<1401181322> : tensor<1024xi32>
    %cst_14 = arith.constant dense<534103459> : tensor<1024xi32>
    %cst_15 = arith.constant dense<-616729560> : tensor<1024xi32>
    %cst_16 = arith.constant dense<-1879881732> : tensor<1024xi32>
    %cst_17 = arith.constant dense<0> : tensor<1024xi32>
    %cst_18 = arith.constant dense<4.6566126E-10> : tensor<1024xf32>
    %cst_19 = arith.constant dense<-1> : tensor<1024xi32>
    %cst_20 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_21 = arith.constant dense<5.000000e-01> : tensor<1024xf32>
    %cst_22 = arith.constant dense<98432> : tensor<1024xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = arith.cmpi slt, %4, %cst_22 : tensor<1024xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %7, %5 : tensor<1024x!tt.ptr<f32>>
    %9 = tt.mulhiui %c-845247145_i32, %c0_i32 : i32
    %10 = arith.xori %9, %c123_i32 : i32
    %11 = tt.mulhiui %cst_0, %4 : tensor<1024xi32>
    %12 = arith.muli %4, %cst_0 : tensor<1024xi32>
    %13 = tt.mulhiui %cst, %11 : tensor<1024xi32>
    %14 = arith.xori %13, %cst_1 : tensor<1024xi32>
    %15 = tt.mulhiui %c-766435501_i32, %10 : i32
    %16 = tt.splat %15 : i32 -> tensor<1024xi32>
    %17 = arith.xori %16, %12 : tensor<1024xi32>
    %18 = arith.xori %17, %cst_2 : tensor<1024xi32>
    %19 = arith.muli %11, %cst : tensor<1024xi32>
    %20 = arith.muli %10, %c-766435501_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<1024xi32>
    %22 = tt.mulhiui %cst, %18 : tensor<1024xi32>
    %23 = arith.xori %22, %19 : tensor<1024xi32>
    %24 = arith.xori %23, %cst_3 : tensor<1024xi32>
    %25 = tt.mulhiui %cst_0, %14 : tensor<1024xi32>
    %26 = arith.xori %25, %21 : tensor<1024xi32>
    %27 = arith.xori %26, %cst_4 : tensor<1024xi32>
    %28 = arith.muli %18, %cst : tensor<1024xi32>
    %29 = arith.muli %14, %cst_0 : tensor<1024xi32>
    %30 = tt.mulhiui %cst, %27 : tensor<1024xi32>
    %31 = arith.xori %30, %28 : tensor<1024xi32>
    %32 = arith.xori %31, %cst_5 : tensor<1024xi32>
    %33 = tt.mulhiui %cst_0, %24 : tensor<1024xi32>
    %34 = arith.xori %33, %29 : tensor<1024xi32>
    %35 = arith.xori %34, %cst_6 : tensor<1024xi32>
    %36 = arith.muli %27, %cst : tensor<1024xi32>
    %37 = arith.muli %24, %cst_0 : tensor<1024xi32>
    %38 = tt.mulhiui %cst, %35 : tensor<1024xi32>
    %39 = arith.xori %38, %36 : tensor<1024xi32>
    %40 = arith.xori %39, %cst_7 : tensor<1024xi32>
    %41 = tt.mulhiui %cst_0, %32 : tensor<1024xi32>
    %42 = arith.xori %41, %37 : tensor<1024xi32>
    %43 = arith.xori %42, %cst_8 : tensor<1024xi32>
    %44 = arith.muli %35, %cst : tensor<1024xi32>
    %45 = arith.muli %32, %cst_0 : tensor<1024xi32>
    %46 = tt.mulhiui %cst, %43 : tensor<1024xi32>
    %47 = arith.xori %46, %44 : tensor<1024xi32>
    %48 = arith.xori %47, %cst_9 : tensor<1024xi32>
    %49 = tt.mulhiui %cst_0, %40 : tensor<1024xi32>
    %50 = arith.xori %49, %45 : tensor<1024xi32>
    %51 = arith.xori %50, %cst_10 : tensor<1024xi32>
    %52 = arith.muli %43, %cst : tensor<1024xi32>
    %53 = arith.muli %40, %cst_0 : tensor<1024xi32>
    %54 = tt.mulhiui %cst, %51 : tensor<1024xi32>
    %55 = arith.xori %54, %52 : tensor<1024xi32>
    %56 = arith.xori %55, %cst_11 : tensor<1024xi32>
    %57 = tt.mulhiui %cst_0, %48 : tensor<1024xi32>
    %58 = arith.xori %57, %53 : tensor<1024xi32>
    %59 = arith.xori %58, %cst_12 : tensor<1024xi32>
    %60 = arith.muli %51, %cst : tensor<1024xi32>
    %61 = arith.muli %48, %cst_0 : tensor<1024xi32>
    %62 = tt.mulhiui %cst, %59 : tensor<1024xi32>
    %63 = arith.xori %62, %60 : tensor<1024xi32>
    %64 = arith.xori %63, %cst_13 : tensor<1024xi32>
    %65 = tt.mulhiui %cst_0, %56 : tensor<1024xi32>
    %66 = arith.xori %65, %61 : tensor<1024xi32>
    %67 = arith.xori %66, %cst_14 : tensor<1024xi32>
    %68 = arith.muli %56, %cst_0 : tensor<1024xi32>
    %69 = tt.mulhiui %cst_0, %64 : tensor<1024xi32>
    %70 = arith.xori %69, %68 : tensor<1024xi32>
    %71 = arith.xori %70, %cst_15 : tensor<1024xi32>
    %72 = arith.muli %67, %cst : tensor<1024xi32>
    %73 = tt.mulhiui %cst, %71 : tensor<1024xi32>
    %74 = arith.xori %73, %72 : tensor<1024xi32>
    %75 = arith.xori %74, %cst_16 : tensor<1024xi32>
    %76 = arith.cmpi slt, %75, %cst_17 : tensor<1024xi32>
    %77 = arith.xori %75, %cst_19 : tensor<1024xi32>
    %78 = arith.select %76, %77, %75 : tensor<1024xi1>, tensor<1024xi32>
    %79 = arith.sitofp %78 : tensor<1024xi32> to tensor<1024xf32>
    %80 = arith.mulf %79, %cst_18 : tensor<1024xf32>
    %81 = arith.cmpf ogt, %80, %cst_21 : tensor<1024xf32>
    %82 = arith.divf %8, %cst_21 : tensor<1024xf32>
    %83 = arith.select %81, %82, %cst_20 : tensor<1024xi1>, tensor<1024xf32>
    %84 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %85 = tt.addptr %84, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %85, %83, %5 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

