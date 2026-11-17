// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=VE
// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt --furiosa-dma --furiosa-ve-fuse %s | FileCheck %s

// A 3x3 convolution's epilogue, relu((conv - m) * s + b + r) with m, s, b per column and r a second tile (a batch
// norm and a residual): the tiled class emits one abstract Vector Engine op (fopt.ve) per operation, and
// furiosa-ve-fuse groups them into passes in the engine's stage order. The subtraction (FMA unit), the scaling
// (Mul0) and the bias add, moved to the clip stage since the FMA unit is taken, share one pass; the residual's zip
// starts the next, which also takes the relu (clip stage).

// VE-LABEL: func.func @epilogue
// VE: fopt.ve "vector_fp_binary" "FpBinaryOp::SubF"
// VE: fopt.ve "vector_fp_binary" "FpBinaryOp::MulF(FpMulAlu::Mul0)"
// VE: fopt.ve "vector_fp_binary" "FpBinaryOp::AddF"
// VE: fopt.ve "vector_fp_zip" "FpBinaryOp::AddF" {{.*}} zip
// VE: fopt.ve "vector_clip" "ClipBinaryOpF32::Max"

// CHECK-LABEL: func.func @epilogue
// CHECK: fopt.contract_outer
// CHECK: fopt.begin main
// CHECK-NOT: fopt.commit
// CHECK: fopt.vector_alu "vector_fp_binary" "FpBinaryOp::SubF"
// CHECK-NOT: fopt.commit
// CHECK: fopt.vector_alu "vector_fp_binary" "FpBinaryOp::MulF(FpMulAlu::Mul0)"
// CHECK-NOT: fopt.commit
// CHECK: fopt.vector_widen_concat
// CHECK-NEXT: fopt.vector_alu "vector_clip" "ClipBinaryOpF32::Add"
// CHECK: fopt.commit
// CHECK: fopt.begin_interleaved main
// CHECK: fopt.vector_alu "vector_fp_zip" "FpBinaryOp::AddF"
// CHECK-NOT: fopt.commit
// CHECK: fopt.vector_alu "vector_clip" "ClipBinaryOpF32::Max"
// CHECK: fopt.commit
// CHECK-NOT: fopt.vector_alu

#x = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 + d3, d1 + d4, d5)>
#w = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d5, d3, d4)>
#y = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#t = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#c = affine_map<(d0, d1, d2) -> (d2)>
#i = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
module {
  func.func @epilogue(%arg0: tensor<1x6x6x8xf32>, %arg1: tensor<16x8x3x3xf32>, %m: tensor<16xf32>, %s: tensor<16xf32>,
                      %b: tensor<16xf32>, %r: tensor<1x6x6x16xf32>, %arg2: tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32> attributes {furiosa.written = [6]} {
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<6x6x16xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<6x6x16xf32>) -> tensor<6x6x16xf32>
    %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%a: index, %bb: index, %cc: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x6x6x8xf32> to tensor<1x8x8x8xf32>
    %acc = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<1x8x8x8xf32>, tensor<16x8x3x3xf32>) outs(%f : tensor<6x6x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %p = arith.mulf %in, %in_0 : f32
      %q = arith.addf %out, %p : f32
      linalg.yield %q : f32
    } -> tensor<6x6x16xf32>
    %o = tensor.empty() : tensor<1x6x6x16xf32>
    %y = linalg.generic {indexing_maps = [#t, #c, #c, #c, #i, #i], iterator_types = ["parallel", "parallel", "parallel"]} ins(%acc, %m, %s, %b, %r : tensor<6x6x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<1x6x6x16xf32>) outs(%o : tensor<1x6x6x16xf32>) {
    ^bb0(%v: f32, %mv: f32, %sv: f32, %bv: f32, %rv: f32, %out: f32):
      %dd = arith.subf %v, %mv : f32
      %k = arith.mulf %dd, %sv : f32
      %l = arith.addf %k, %bv : f32
      %n = arith.addf %l, %rv : f32
      %u = arith.maximumf %n, %cst : f32
      linalg.yield %u : f32
    } -> tensor<1x6x6x16xf32>
    return %y : tensor<1x6x6x16xf32>
  }
}
