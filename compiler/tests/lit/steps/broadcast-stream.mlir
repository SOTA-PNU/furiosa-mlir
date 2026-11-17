// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// tl.histogram: count[b] = sum_i (x[i] == b), x varying along the reduced loop alone and the bins along the
// other. No input spans the tile the row-wise class streams, so x is broadcast over the bins first (a relayout
// step, one DMA) and the count reads it whole.

// CHECK-LABEL: func.func private @histogram__s0
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<4096xi32>) outs(%{{.*}} : tensor<64x4096xi32>)
// CHECK-LABEL: func.func private @histogram__s1
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]} ins(%{{.*}}, %{{.*}} : tensor<64x4096xi32>, tensor<64xi32>)
#col = affine_map<(d0, d1) -> (d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @histogram(%x: tensor<4096xi32>, %b: tensor<64xi32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [2]} {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<64xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<64xi32>) -> tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#col, #row, #row], iterator_types = ["parallel", "reduction"]} ins(%x, %b : tensor<4096xi32>, tensor<64xi32>) outs(%z : tensor<64xi32>) {
    ^bb0(%v: i32, %k: i32, %acc: i32):
      %eq = arith.cmpi eq, %v, %k : i32
      %one = arith.select %eq, %c1, %c0 : i32
      %s = arith.addi %acc, %one : i32
      linalg.yield %s : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
