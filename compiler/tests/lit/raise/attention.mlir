// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s

// Triton tutorial 06's causal flash attention (BLOCK_M = BLOCK_N = 64, N_CTX = 256, 2 heads). Its
// off-band K/V loop runs start_m times, a trip that varies with the program (start_m = origin / 64,
// fitted as an origin divided by its step): instances past their trip keep their state. The row max
// m_i is stored on its own (M); built for that store, the loop leaves out the accumulator, whose
// head-dim axis has nothing to do with M's coordinates: two states (m_i and l_i), not three.

// CHECK-LABEL: func.func @_attn_fwd
// CHECK: scf.for %[[IV:[a-z0-9]+]] = {{.*}} -> (tensor<2x4x64xf32>, tensor<2x4x64xf32>)
// CHECK: %[[LIVE:.*]] = arith.cmpi slt, %[[IV]], %{{.*}} : index
// CHECK: arith.select %[[LIVE]]
// CHECK: scf.for {{.*}} -> (tensor<1x2x4x64x64xf32>, tensor<2x4x64xf32>, tensor<2x4x64xf32>)

// Triton source: harness/cases/tutorial_fused_attention.py.

module attributes {furiosa.launch = "{\22grid\22: [4, 2, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22output\22, \22dtype\22: \22fp32\22, \22shape\22: [1, 2, 256]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [1, 2, 256, 64]}, {\22arg\22: 2, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [1, 2, 256, 64]}, {\22arg\22: 3, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [1, 2, 256, 64]}, {\22arg\22: 4, \22kind\22: \22output\22, \22dtype\22: \22fp16\22, \22shape\22: [1, 2, 256, 64]}]}"} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi64>
    %cst_0 = arith.constant dense<64> : tensor<1x64xi64>
    %cst_1 = arith.constant dense<0> : tensor<1x64xi64>
    %cst_2 = arith.constant dense<512> : tensor<64x1xi64>
    %cst_3 = arith.constant dense<0> : tensor<64x1xi64>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %cst_5 = arith.constant dense<0.72134751> : tensor<64x64xf32>
    %cst_6 = arith.constant dense<0.72134751> : tensor<64xf32>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_8 = arith.constant dense<0xFF800000> : tensor<64xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c2_i32 : i32
    %3 = arith.remsi %1, %c2_i32 : i32
    %4 = arith.muli %2, %c512_i32 : i32
    %5 = arith.muli %3, %c256_i32 : i32
    %6 = arith.addi %4, %5 : i32
    %7 = arith.muli %0, %c64_i32 : i32
    %8 = arith.addi %6, %7 : i32
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %10 = tt.splat %7 : i32 -> tensor<64xi32>
    %11 = arith.addi %10, %9 : tensor<64xi32>
    %12 = arith.extsi %8 : i32 to i64
    %13 = tt.splat %12 : i64 -> tensor<64xi64>
    %14 = arith.extsi %9 : tensor<64xi32> to tensor<64xi64>
    %15 = arith.addi %13, %14 : tensor<64xi64>
    %16 = tt.reshape %15 : tensor<64xi64> -> tensor<64x1xi64>
    %17 = tt.reshape %14 : tensor<64xi64> -> tensor<1x64xi64>
    %18 = arith.cmpi sge, %16, %cst_3 : tensor<64x1xi64>
    %19 = arith.cmpi slt, %16, %cst_2 : tensor<64x1xi64>
    %20 = arith.andi %18, %19 : tensor<64x1xi1>
    %21 = tt.broadcast %20 : tensor<64x1xi1> -> tensor<64x64xi1>
    %22 = arith.cmpi sge, %17, %cst_1 : tensor<1x64xi64>
    %23 = arith.cmpi slt, %17, %cst_0 : tensor<1x64xi64>
    %24 = arith.andi %22, %23 : tensor<1x64xi1>
    %25 = tt.broadcast %24 : tensor<1x64xi1> -> tensor<64x64xi1>
    %26 = arith.andi %21, %25 : tensor<64x64xi1>
    %27 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
    %28 = arith.muli %16, %cst : tensor<64x1xi64>
    %29 = tt.broadcast %28 : tensor<64x1xi64> -> tensor<64x64xi64>
    %30 = tt.broadcast %17 : tensor<1x64xi64> -> tensor<64x64xi64>
    %31 = arith.addi %29, %30 : tensor<64x64xi64>
    %32 = tt.addptr %27, %31 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
    %33 = tt.load %32, %26, %cst_4 : tensor<64x64x!tt.ptr<f16>>
    %34 = arith.muli %1, %c256_i32 : i32
    %35:5 = scf.for %arg5 = %c0_i32 to %7 step %c64_i32 iter_args(%arg6 = %cst_10, %arg7 = %cst_7, %arg8 = %cst_8, %arg9 = %34, %arg10 = %34) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32)  : i32 {
      %52 = arith.extsi %arg9 : i32 to i64
      %53 = tt.splat %52 : i64 -> tensor<64xi64>
      %54 = arith.addi %53, %14 : tensor<64xi64>
      %55 = tt.reshape %54 : tensor<64xi64> -> tensor<64x1xi64>
      %56 = arith.cmpi sge, %55, %cst_3 : tensor<64x1xi64>
      %57 = arith.cmpi slt, %55, %cst_2 : tensor<64x1xi64>
      %58 = arith.andi %56, %57 : tensor<64x1xi1>
      %59 = tt.broadcast %58 : tensor<64x1xi1> -> tensor<64x64xi1>
      %60 = arith.andi %59, %25 : tensor<64x64xi1>
      %61 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
      %62 = arith.muli %55, %cst : tensor<64x1xi64>
      %63 = tt.broadcast %62 : tensor<64x1xi64> -> tensor<64x64xi64>
      %64 = arith.addi %63, %30 : tensor<64x64xi64>
      %65 = tt.addptr %61, %64 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
      %66 = tt.load %65, %60, %cst_4 : tensor<64x64x!tt.ptr<f16>>
      %67 = tt.trans %66 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
      %68 = tt.dot %33, %67, %cst_10, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %69 = "tt.reduce"(%68) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %104 = arith.maxnumf %arg11, %arg12 : f32
        tt.reduce.return %104 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %70 = arith.mulf %69, %cst_6 : tensor<64xf32>
      %71 = arith.maxnumf %arg8, %70 : tensor<64xf32>
      %72 = arith.mulf %68, %cst_5 : tensor<64x64xf32>
      %73 = tt.expand_dims %71 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %74 = tt.broadcast %73 : tensor<64x1xf32> -> tensor<64x64xf32>
      %75 = arith.subf %72, %74 : tensor<64x64xf32>
      %76 = math.exp2 %75 : tensor<64x64xf32>
      %77 = arith.subf %arg8, %71 : tensor<64xf32>
      %78 = math.exp2 %77 : tensor<64xf32>
      %79 = "tt.reduce"(%76) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %104 = arith.addf %arg11, %arg12 : f32
        tt.reduce.return %104 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %80 = tt.expand_dims %78 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %81 = tt.broadcast %80 : tensor<64x1xf32> -> tensor<64x64xf32>
      %82 = arith.mulf %arg6, %81 : tensor<64x64xf32>
      %83 = arith.extsi %arg10 : i32 to i64
      %84 = tt.splat %83 : i64 -> tensor<64xi64>
      %85 = arith.addi %84, %14 : tensor<64xi64>
      %86 = tt.reshape %85 : tensor<64xi64> -> tensor<64x1xi64>
      %87 = arith.cmpi sge, %86, %cst_3 : tensor<64x1xi64>
      %88 = arith.cmpi slt, %86, %cst_2 : tensor<64x1xi64>
      %89 = arith.andi %87, %88 : tensor<64x1xi1>
      %90 = tt.broadcast %89 : tensor<64x1xi1> -> tensor<64x64xi1>
      %91 = arith.andi %90, %25 : tensor<64x64xi1>
      %92 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
      %93 = arith.muli %86, %cst : tensor<64x1xi64>
      %94 = tt.broadcast %93 : tensor<64x1xi64> -> tensor<64x64xi64>
      %95 = arith.addi %94, %30 : tensor<64x64xi64>
      %96 = tt.addptr %92, %95 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
      %97 = tt.load %96, %91, %cst_4 : tensor<64x64x!tt.ptr<f16>>
      %98 = arith.truncf %76 : tensor<64x64xf32> to tensor<64x64xf16>
      %99 = tt.dot %98, %97, %82, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %100 = arith.mulf %arg7, %78 : tensor<64xf32>
      %101 = arith.addf %100, %79 : tensor<64xf32>
      %102 = arith.addi %arg9, %c64_i32 : i32
      %103 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %99, %101, %71, %102, %103 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32
    }
    %36 = arith.muli %0, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
    %37 = arith.addi %0, %c1_i32 : i32
    %38 = arith.muli %37, %c64_i32 : i32
    %39 = arith.addi %34, %36 : i32
    %40:5 = scf.for %arg5 = %36 to %38 step %c64_i32 iter_args(%arg6 = %35#0, %arg7 = %35#1, %arg8 = %35#2, %arg9 = %39, %arg10 = %39) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32)  : i32 {
      %52 = arith.extsi %arg9 : i32 to i64
      %53 = tt.splat %52 : i64 -> tensor<64xi64>
      %54 = arith.addi %53, %14 : tensor<64xi64>
      %55 = tt.reshape %54 : tensor<64xi64> -> tensor<64x1xi64>
      %56 = arith.cmpi sge, %55, %cst_3 : tensor<64x1xi64>
      %57 = arith.cmpi slt, %55, %cst_2 : tensor<64x1xi64>
      %58 = arith.andi %56, %57 : tensor<64x1xi1>
      %59 = tt.broadcast %58 : tensor<64x1xi1> -> tensor<64x64xi1>
      %60 = arith.andi %59, %25 : tensor<64x64xi1>
      %61 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
      %62 = arith.muli %55, %cst : tensor<64x1xi64>
      %63 = tt.broadcast %62 : tensor<64x1xi64> -> tensor<64x64xi64>
      %64 = arith.addi %63, %30 : tensor<64x64xi64>
      %65 = tt.addptr %61, %64 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
      %66 = tt.load %65, %60, %cst_4 : tensor<64x64x!tt.ptr<f16>>
      %67 = tt.trans %66 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
      %68 = tt.dot %33, %67, %cst_10, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %69 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %70 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %71 = tt.splat %arg5 : i32 -> tensor<1x64xi32>
      %72 = arith.addi %71, %70 : tensor<1x64xi32>
      %73 = tt.broadcast %69 : tensor<64x1xi32> -> tensor<64x64xi32>
      %74 = tt.broadcast %72 : tensor<1x64xi32> -> tensor<64x64xi32>
      %75 = arith.cmpi sge, %73, %74 : tensor<64x64xi32>
      %76 = arith.mulf %68, %cst_5 : tensor<64x64xf32>
      %77 = arith.select %75, %cst_10, %cst_9 : tensor<64x64xi1>, tensor<64x64xf32>
      %78 = arith.addf %76, %77 : tensor<64x64xf32>
      %79 = "tt.reduce"(%78) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %112 = arith.maxnumf %arg11, %arg12 : f32
        tt.reduce.return %112 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %80 = arith.maxnumf %arg8, %79 : tensor<64xf32>
      %81 = tt.expand_dims %80 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %82 = tt.broadcast %81 : tensor<64x1xf32> -> tensor<64x64xf32>
      %83 = arith.subf %78, %82 : tensor<64x64xf32>
      %84 = math.exp2 %83 : tensor<64x64xf32>
      %85 = arith.subf %arg8, %80 : tensor<64xf32>
      %86 = math.exp2 %85 : tensor<64xf32>
      %87 = "tt.reduce"(%84) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %112 = arith.addf %arg11, %arg12 : f32
        tt.reduce.return %112 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %88 = tt.expand_dims %86 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %89 = tt.broadcast %88 : tensor<64x1xf32> -> tensor<64x64xf32>
      %90 = arith.mulf %arg6, %89 : tensor<64x64xf32>
      %91 = arith.extsi %arg10 : i32 to i64
      %92 = tt.splat %91 : i64 -> tensor<64xi64>
      %93 = arith.addi %92, %14 : tensor<64xi64>
      %94 = tt.reshape %93 : tensor<64xi64> -> tensor<64x1xi64>
      %95 = arith.cmpi sge, %94, %cst_3 : tensor<64x1xi64>
      %96 = arith.cmpi slt, %94, %cst_2 : tensor<64x1xi64>
      %97 = arith.andi %95, %96 : tensor<64x1xi1>
      %98 = tt.broadcast %97 : tensor<64x1xi1> -> tensor<64x64xi1>
      %99 = arith.andi %98, %25 : tensor<64x64xi1>
      %100 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
      %101 = arith.muli %94, %cst : tensor<64x1xi64>
      %102 = tt.broadcast %101 : tensor<64x1xi64> -> tensor<64x64xi64>
      %103 = arith.addi %102, %30 : tensor<64x64xi64>
      %104 = tt.addptr %100, %103 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
      %105 = tt.load %104, %99, %cst_4 : tensor<64x64x!tt.ptr<f16>>
      %106 = arith.truncf %84 : tensor<64x64xf32> to tensor<64x64xf16>
      %107 = tt.dot %106, %105, %90, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %108 = arith.mulf %arg7, %86 : tensor<64xf32>
      %109 = arith.addf %108, %87 : tensor<64xf32>
      %110 = arith.addi %arg9, %c64_i32 : i32
      %111 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %107, %109, %80, %110, %111 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32
    }
    %41 = math.log2 %40#1 : tensor<64xf32>
    %42 = arith.addf %40#2, %41 : tensor<64xf32>
    %43 = tt.expand_dims %40#1 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %44 = tt.broadcast %43 : tensor<64x1xf32> -> tensor<64x64xf32>
    %45 = arith.divf %40#0, %44 : tensor<64x64xf32>
    %46 = tt.addptr %arg0, %34 : !tt.ptr<f32>, i32
    %47 = tt.splat %46 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %48 = tt.addptr %47, %11 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %48, %42 : tensor<64x!tt.ptr<f32>>
    %49 = arith.truncf %45 : tensor<64x64xf32> to tensor<64x64xf16>
    %50 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
    %51 = tt.addptr %50, %31 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
    tt.store %51, %49, %26 : tensor<64x64x!tt.ptr<f16>>
    tt.return
  }
}

