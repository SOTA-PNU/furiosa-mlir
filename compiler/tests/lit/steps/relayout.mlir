// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// An elementwise generic that reads an operand through the loops out of their order, y[i, j] = a[i, j] ^ b[j, i],
// maps once that operand is relaid out in loop order: a transpose step (one DMA), then the xor reading in order.

// CHECK-LABEL: func.func private @transposed__s0
// CHECK: linalg.transpose ins(%{{.*}} : tensor<64x32xi32>) outs(%{{.*}} : tensor<32x64xi32>) permutation = [1, 0]
// CHECK-LABEL: func.func private @transposed__s1
// CHECK: linalg.generic {indexing_maps = [#[[ID:map[0-9]*]], #[[ID]], #[[ID]]]
// CHECK: arith.xori
#id = affine_map<(d0, d1) -> (d0, d1)>
#t = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @transposed(%a: tensor<32x64xi32>, %b: tensor<64x32xi32>, %o: tensor<32x64xi32>) -> tensor<32x64xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<32x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #t, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x64xi32>, tensor<64x32xi32>) outs(%e : tensor<32x64xi32>) {
    ^bb0(%x: i32, %y: i32, %out: i32):
      %z = arith.xori %x, %y : i32
      linalg.yield %z : i32
    } -> tensor<32x64xi32>
    return %r : tensor<32x64xi32>
  }
}
