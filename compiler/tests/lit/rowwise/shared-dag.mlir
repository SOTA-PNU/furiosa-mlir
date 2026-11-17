// RUN: timeout 60 furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// A value two ops of the streamed shape share (v + exp(v), forty deep, as tl.sort's stages read the previous one
// twice) is computed once, and choosing which operand to stream costs time linear in the DAG, not in its paths
// (2^40 here).

// CHECK-LABEL: func.func @shared
// CHECK-COUNT-40: "FpUnaryOp::Exp"
// CHECK-NOT: "FpUnaryOp::Exp"
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @shared(%x: tensor<64x64xf32>, %r: tensor<64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %y = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %r : tensor<64x64xf32>, tensor<64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v0 = arith.addf %a, %b : f32
      %e0 = math.exp %v0 : f32
      %v1 = arith.addf %v0, %e0 : f32
      %e1 = math.exp %v1 : f32
      %v2 = arith.addf %v1, %e1 : f32
      %e2 = math.exp %v2 : f32
      %v3 = arith.addf %v2, %e2 : f32
      %e3 = math.exp %v3 : f32
      %v4 = arith.addf %v3, %e3 : f32
      %e4 = math.exp %v4 : f32
      %v5 = arith.addf %v4, %e4 : f32
      %e5 = math.exp %v5 : f32
      %v6 = arith.addf %v5, %e5 : f32
      %e6 = math.exp %v6 : f32
      %v7 = arith.addf %v6, %e6 : f32
      %e7 = math.exp %v7 : f32
      %v8 = arith.addf %v7, %e7 : f32
      %e8 = math.exp %v8 : f32
      %v9 = arith.addf %v8, %e8 : f32
      %e9 = math.exp %v9 : f32
      %v10 = arith.addf %v9, %e9 : f32
      %e10 = math.exp %v10 : f32
      %v11 = arith.addf %v10, %e10 : f32
      %e11 = math.exp %v11 : f32
      %v12 = arith.addf %v11, %e11 : f32
      %e12 = math.exp %v12 : f32
      %v13 = arith.addf %v12, %e12 : f32
      %e13 = math.exp %v13 : f32
      %v14 = arith.addf %v13, %e13 : f32
      %e14 = math.exp %v14 : f32
      %v15 = arith.addf %v14, %e14 : f32
      %e15 = math.exp %v15 : f32
      %v16 = arith.addf %v15, %e15 : f32
      %e16 = math.exp %v16 : f32
      %v17 = arith.addf %v16, %e16 : f32
      %e17 = math.exp %v17 : f32
      %v18 = arith.addf %v17, %e17 : f32
      %e18 = math.exp %v18 : f32
      %v19 = arith.addf %v18, %e18 : f32
      %e19 = math.exp %v19 : f32
      %v20 = arith.addf %v19, %e19 : f32
      %e20 = math.exp %v20 : f32
      %v21 = arith.addf %v20, %e20 : f32
      %e21 = math.exp %v21 : f32
      %v22 = arith.addf %v21, %e21 : f32
      %e22 = math.exp %v22 : f32
      %v23 = arith.addf %v22, %e22 : f32
      %e23 = math.exp %v23 : f32
      %v24 = arith.addf %v23, %e23 : f32
      %e24 = math.exp %v24 : f32
      %v25 = arith.addf %v24, %e24 : f32
      %e25 = math.exp %v25 : f32
      %v26 = arith.addf %v25, %e25 : f32
      %e26 = math.exp %v26 : f32
      %v27 = arith.addf %v26, %e26 : f32
      %e27 = math.exp %v27 : f32
      %v28 = arith.addf %v27, %e27 : f32
      %e28 = math.exp %v28 : f32
      %v29 = arith.addf %v28, %e28 : f32
      %e29 = math.exp %v29 : f32
      %v30 = arith.addf %v29, %e29 : f32
      %e30 = math.exp %v30 : f32
      %v31 = arith.addf %v30, %e30 : f32
      %e31 = math.exp %v31 : f32
      %v32 = arith.addf %v31, %e31 : f32
      %e32 = math.exp %v32 : f32
      %v33 = arith.addf %v32, %e32 : f32
      %e33 = math.exp %v33 : f32
      %v34 = arith.addf %v33, %e33 : f32
      %e34 = math.exp %v34 : f32
      %v35 = arith.addf %v34, %e34 : f32
      %e35 = math.exp %v35 : f32
      %v36 = arith.addf %v35, %e35 : f32
      %e36 = math.exp %v36 : f32
      %v37 = arith.addf %v36, %e36 : f32
      %e37 = math.exp %v37 : f32
      %v38 = arith.addf %v37, %e37 : f32
      %e38 = math.exp %v38 : f32
      %v39 = arith.addf %v38, %e38 : f32
      %e39 = math.exp %v39 : f32
      %v40 = arith.addf %v39, %e39 : f32
      linalg.yield %v40 : f32
    } -> tensor<64x64xf32>
    return %y : tensor<64x64xf32>
  }
}
