// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A loop whose reads take it apart by a place that does not divide its extent (an exclusive scan's in-tile prefix:
// columns c of 10 read their 4-column tile, c floordiv 4, of the row padded to 12), in a generic no class maps as it
// is, pads to whole digits [3, 4] where every read stays in its tensor: the reduction runs over [2, 3, 4] rows from a
// fill of that shape, and its result is sliced back to the 10 columns.

// CHECK-LABEL: func.func @tile_scan
// CHECK: %[[S:.*]] = call @tile_scan__s1(%{{.*}}, %{{.*}}) : (tensor<2x3x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
// CHECK: %[[C:.*]] = tensor.collapse_shape %[[S]] {{\[}}[0], [1, 2]] : tensor<2x3x4xi32> into tensor<2x12xi32>
// CHECK: call @tile_scan__s2(%[[C]], %arg1)
// CHECK-LABEL: func.func private @tile_scan__s1
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: outs(%{{.*}} : tensor<2x3x4xi32>)
// CHECK-LABEL: func.func private @tile_scan__s2
// CHECK: tensor.extract_slice %arg0[0, 0] [2, 10] [1, 1] : tensor<2x12xi32> to tensor<2x10xi32>

#c = affine_map<(d0, d1, d2) -> (d2)>
#x = affine_map<(d0, d1, d2) -> (d0, d2 + (d1 floordiv 4) * 4)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @tile_scan(%x: tensor<2x10xi32>, %o: tensor<2x10xi32>) -> tensor<2x10xi32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %k = tensor.empty() : tensor<4xi8>
    %e = tensor.empty() : tensor<2x10xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<2x10xi32>) -> tensor<2x10xi32>
    %p = tensor.pad %x low[0, 0] high[0, 2] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : i32
    } : tensor<2x10xi32> to tensor<2x12xi32>
    %r = linalg.generic {indexing_maps = [#c, #x, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%k, %p : tensor<4xi8>, tensor<2x12xi32>) outs(%f : tensor<2x10xi32>) {
    ^bb0(%u: i8, %a: i32, %acc: i32):
      %col = linalg.index 1 : index
      %kk = linalg.index 2 : index
      %t = arith.floordivsi %col, %c4 : index
      %start = arith.muli %t, %c4 : index
      %in = arith.subi %col, %start : index
      %d = arith.subi %in, %kk : index
      %on = arith.cmpi sge, %d, %c0 : index
      %v = arith.select %on, %a, %zero : i32
      %s = arith.addi %acc, %v : i32
      linalg.yield %s : i32
    } -> tensor<2x10xi32>
    return %r : tensor<2x10xi32>
  }
}
