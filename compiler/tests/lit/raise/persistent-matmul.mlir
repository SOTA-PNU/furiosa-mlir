// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=arith.maxsi
// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg --furiosa-structure %s | FileCheck %s --check-prefix=STRUCT --implicit-check-not='tensor.extract %'

// Triton tutorial 09's persistent matmul: 4 programs stride over the 16 output tiles in grouped order
// (min(num_pid_m - first_pid_m, GROUP_SIZE_M) decided by the ranges of the tile loop; pid_n =
// (tile_id mod 32) floordiv 4 resolved to the loop counter over its range), each tile a K loop of
// masked dots, the store inside the tile loop. Raised, the schedule is gone: one contraction over the
// 512x512 output (tile columns as their own dim) reducing over the 4 K tiles of 64. Its rows go through
// tl.where(offs_am < M, offs_am, 0), which the ranges decide (every row is below M): no clamp, so the
// reads structure into indexing maps of A and B.

// CHECK-LABEL: func.func @matmul_kernel_persistent
// CHECK: linalg.generic {{.*}}["parallel", "parallel", "parallel", "reduction", "reduction"]{{.*}} ins({{.*}} : tensor<4x64xi8>) outs({{.*}} : tensor<512x4x128xf32>)
// CHECK: arith.mulf
// CHECK: return {{.*}} : tensor<512x512xf16>

// STRUCT: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<512x256xf16>, tensor<256x512xf16>)

// Triton source: harness/cases/tutorial_persistent_matmul.py (M = N = 512, K = 256, 128x128x64 tiles).

module attributes {furiosa.launch = "{\22grid\22: [4, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [512, 256]}, {\22arg\22: 1, \22kind\22: \22input\22, \22dtype\22: \22fp16\22, \22shape\22: [256, 512]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22fp16\22, \22shape\22: [512, 512]}]}"} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant dense<512> : tensor<1x128xi32>
    %cst_3 = arith.constant dense<512> : tensor<128x1xi32>
    %cst_4 = arith.constant dense<512> : tensor<64x1xi32>
    %cst_5 = arith.constant dense<256> : tensor<128x1xi32>
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_6 = arith.constant dense<0> : tensor<128xi32>
    %cst_7 = arith.constant dense<512> : tensor<128xi32>
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.subi %0, %c4_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = scf.for %arg3 = %0 to %c16_i32 step %c4_i32 iter_args(%arg4 = %1) -> (i32)  : i32 {
      %4 = arith.divsi %arg3, %c32_i32 : i32
      %5 = arith.muli %4, %c8_i32 : i32
      %6 = arith.subi %c4_i32, %5 : i32
      %7 = arith.minsi %6, %c8_i32 : i32
      %8 = arith.remsi %arg3, %7 : i32
      %9 = arith.addi %5, %8 : i32
      %10 = arith.remsi %arg3, %c32_i32 : i32
      %11 = arith.divsi %10, %7 : i32
      %12 = arith.muli %9, %c128_i32 : i32
      %13 = arith.muli %11, %c128_i32 : i32
      %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %15 = tt.splat %12 : i32 -> tensor<128xi32>
      %16 = arith.addi %15, %14 : tensor<128xi32>
      %17 = tt.splat %13 : i32 -> tensor<128xi32>
      %18 = arith.addi %17, %14 : tensor<128xi32>
      %19 = arith.cmpi slt, %16, %cst_7 : tensor<128xi32>
      %20 = arith.select %19, %16, %cst_6 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
      %21 = arith.cmpi slt, %18, %cst_7 : tensor<128xi32>
      %22 = arith.select %21, %18, %cst_6 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
      %23 = scf.for %arg5 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<128x128xf32>)  : i32 {
        %53 = arith.muli %arg5, %c64_i32 : i32
        %54 = tt.splat %53 : i32 -> tensor<64xi32>
        %55 = arith.addi %54, %2 : tensor<64xi32>
        %56 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %57 = arith.muli %56, %cst_5 : tensor<128x1xi32>
        %58 = tt.expand_dims %55 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %59 = tt.broadcast %57 : tensor<128x1xi32> -> tensor<128x64xi32>
        %60 = tt.broadcast %58 : tensor<1x64xi32> -> tensor<128x64xi32>
        %61 = arith.addi %59, %60 : tensor<128x64xi32>
        %62 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
        %63 = tt.addptr %62, %61 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %64 = tt.expand_dims %55 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %65 = arith.muli %64, %cst_4 : tensor<64x1xi32>
        %66 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %67 = tt.broadcast %65 : tensor<64x1xi32> -> tensor<64x128xi32>
        %68 = tt.broadcast %66 : tensor<1x128xi32> -> tensor<64x128xi32>
        %69 = arith.addi %67, %68 : tensor<64x128xi32>
        %70 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
        %71 = tt.addptr %70, %69 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
        %72 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %73 = arith.subi %c256_i32, %53 : i32
        %74 = tt.splat %73 : i32 -> tensor<1x64xi32>
        %75 = arith.cmpi slt, %72, %74 : tensor<1x64xi32>
        %76 = tt.broadcast %75 : tensor<1x64xi1> -> tensor<128x64xi1>
        %77 = tt.load %63, %76, %cst_1 : tensor<128x64x!tt.ptr<f16>>
        %78 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %79 = tt.splat %73 : i32 -> tensor<64x1xi32>
        %80 = arith.cmpi slt, %78, %79 : tensor<64x1xi32>
        %81 = tt.broadcast %80 : tensor<64x1xi1> -> tensor<64x128xi1>
        %82 = tt.load %71, %81, %cst_0 : tensor<64x128x!tt.ptr<f16>>
        %83 = tt.dot %77, %82, %arg6, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        scf.yield %83 : tensor<128x128xf32>
      }
      %24 = arith.addi %arg4, %c4_i32 : i32
      %25 = arith.divsi %24, %c32_i32 : i32
      %26 = arith.muli %25, %c8_i32 : i32
      %27 = arith.subi %c4_i32, %26 : i32
      %28 = arith.minsi %27, %c8_i32 : i32
      %29 = arith.remsi %24, %28 : i32
      %30 = arith.addi %26, %29 : i32
      %31 = arith.remsi %24, %c32_i32 : i32
      %32 = arith.divsi %31, %28 : i32
      %33 = arith.muli %30, %c128_i32 : i32
      %34 = tt.splat %33 : i32 -> tensor<128xi32>
      %35 = arith.addi %34, %14 : tensor<128xi32>
      %36 = arith.muli %32, %c128_i32 : i32
      %37 = tt.splat %36 : i32 -> tensor<128xi32>
      %38 = arith.addi %37, %14 : tensor<128xi32>
      %39 = tt.expand_dims %35 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %40 = arith.muli %39, %cst_3 : tensor<128x1xi32>
      %41 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.addptr %41, %40 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %43 = tt.expand_dims %38 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %44 = tt.broadcast %42 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
      %45 = tt.broadcast %43 : tensor<1x128xi32> -> tensor<128x128xi32>
      %46 = tt.addptr %44, %45 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
      %47 = arith.cmpi slt, %39, %cst_3 : tensor<128x1xi32>
      %48 = arith.cmpi slt, %43, %cst_2 : tensor<1x128xi32>
      %49 = tt.broadcast %47 : tensor<128x1xi1> -> tensor<128x128xi1>
      %50 = tt.broadcast %48 : tensor<1x128xi1> -> tensor<128x128xi1>
      %51 = arith.andi %49, %50 : tensor<128x128xi1>
      %52 = arith.truncf %23 : tensor<128x128xf32> to tensor<128x128xf16>
      tt.store %46, %52, %51 : tensor<128x128x!tt.ptr<f16>>
      scf.yield %24 : i32
    } {tt.flatten}
    tt.return
  }
}

