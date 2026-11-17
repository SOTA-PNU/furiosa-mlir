// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// A reduction loop of extent 1 (tl.sum over an axis tl.expand_dims made) reduces nothing: the loop goes, its index
// reads as 0, and the generic is elementwise (the zero start read as an input).

// CHECK-DAG: #[[UNIT:.*]] = affine_map<(d0, d1) -> (d0, d1, 0)>
// CHECK-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @unit_sum
// CHECK: linalg.generic {indexing_maps = [#[[UNIT]], #[[ID]], #[[ID]]], iterator_types = ["parallel", "parallel"]}
// CHECK: arith.addi
#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @unit_sum(%x: tensor<64x32x1xi32>, %o: tensor<64x32xi32>) -> tensor<64x32xi32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<64x32xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<64x32xi32>) -> tensor<64x32xi32>
    %r = linalg.generic {indexing_maps = [#x, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<64x32x1xi32>) outs(%z : tensor<64x32xi32>) {
    ^bb0(%v: i32, %acc: i32):
      %s = arith.addi %acc, %v : i32
      linalg.yield %s : i32
    } -> tensor<64x32xi32>
    return %r : tensor<64x32xi32>
  }
}
