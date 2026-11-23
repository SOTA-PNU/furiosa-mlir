// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// An atomic add whose lanes data selects (dw[ids[t], d] += dy[t, d]) is raised as a fold that adds a term only
// where the token's id is the row: select(ids[t] == v, acc + dy[t, d], acc). It folds the term or the add's
// identity instead, acc + select(ids[t] == v, dy[t, d], -0.0): exact (x + -0.0 = x for every x, -0.0 too), and
// a plain sum the row-wise class reduces. A float term under a condition of the indices alone keeps its select.

// CHECK-LABEL: func.func private @scatter__s
// CHECK: %[[Z:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: %[[T:.*]] = arith.select %{{.*}}, %{{.*}}, %[[Z]] : f32
// CHECK-NEXT: %[[S:.*]] = arith.addf %{{.*}}, %[[T]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
#dy = affine_map<(d0, d1, d2) -> (d1, d2)>
#ids = affine_map<(d0, d1, d2) -> (d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @scatter(%dy: tensor<64x32xf32>, %ids: tensor<32xi32>, %dw: tensor<8x64xf32>) -> tensor<8x64xf32> attributes {furiosa.written = [2]} {
    %r = linalg.generic {indexing_maps = [#dy, #ids, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%dy, %ids : tensor<64x32xf32>, tensor<32xi32>) outs(%dw : tensor<8x64xf32>) {
    ^bb0(%x: f32, %id: i32, %acc: f32):
      %v = linalg.index 0 : index
      %vi = arith.index_cast %v : index to i32
      %hit = arith.cmpi eq, %id, %vi : i32
      %s = arith.addf %acc, %x : f32
      %y = arith.select %hit, %s, %acc : f32
      linalg.yield %y : f32
    } -> tensor<8x64xf32>
    return %r : tensor<8x64xf32>
  }
}
