// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A segment sum over [segments, columns, rows of the segment] masked by each segment's end: its rows, 13 x 48 x 208,
// have no row dim whose others, each padded to a power of two, fit the row-wise class's 256 slices. The end, read
// along the segments alone, is broadcast over the columns (a relayout step, one DMA), and the first two loops
// collapse into one of 624 rows, 208 rows inside it.

// CHECK-LABEL: func.func private @segments__s0
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<13xi32>) outs(%{{.*}} : tensor<13x48xi32>)
// CHECK-LABEL: func.func private @segments__s1
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<624xi32>, tensor<624x208x8xf32>, tensor<208x8xi32>) outs(%{{.*}} : tensor<624x208xf32>)
#end = affine_map<(d0, d1, d2, d3) -> (d0)>
#x = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#o = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @segments(%end: tensor<13xi32>, %x: tensor<13x48x208x8xf32>, %o: tensor<13x48x208xf32>) -> tensor<13x48x208xf32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<13x48x208xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<13x48x208xf32>) -> tensor<13x48x208xf32>
    %r = linalg.generic {indexing_maps = [#end, #x, #o], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%end, %x : tensor<13xi32>, tensor<13x48x208x8xf32>) outs(%f : tensor<13x48x208xf32>) {
    ^bb0(%n: i32, %v: f32, %acc: f32):
      %u = linalg.index 2 : index
      %i = linalg.index 3 : index
      %row = arith.addi %u, %i : index
      %ri = arith.index_cast %row : index to i32
      %in = arith.cmpi slt, %ri, %n : i32
      %t = arith.select %in, %v, %zero : f32
      %s = arith.addf %acc, %t : f32
      linalg.yield %s : f32
    } -> tensor<13x48x208xf32>
    return %r : tensor<13x48x208xf32>
  }
}
