// RUN: furiosa-mlir-opt --furiosa-split-steps --split-input-file %s | FileCheck %s

// An integer reduction whose step a condition guards, select(p, acc + t, acc) (an exclusive scan's carry: the sums
// of the tiles before a column's), in a generic no class maps as it is, adds the identity where p does not hold:
// acc + select(p, t, 0), a masked term the row-wise class maps (p a table once hoisted).

// CHECK-LABEL: func.func private @carry__s0
// CHECK: %[[P:.*]] = arith.cmpi ne, %{{.*}}, %c0_i32 : i32
// CHECK: %[[T:.*]] = arith.select %[[P]], %{{.*}}, %c0_i32 : i32
// CHECK: arith.addi %{{.*}}, %[[T]] : i32

// A float one keeps its guard (a term the guard skips may be inf, which the mask's product would turn into NaN).

// CHECK-LABEL: func.func private @carry_f32__s1
// CHECK: arith.addf
// CHECK: arith.xori

#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @carry(%sums: tensor<2x3xi32>, %o: tensor<2x10xi32>) -> tensor<2x10xi32> attributes {furiosa.written = [1]} {
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2x10xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<2x10xi32>) -> tensor<2x10xi32>
    %r = linalg.generic {indexing_maps = [#x, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%sums : tensor<2x3xi32>) outs(%f : tensor<2x10xi32>) {
    ^bb0(%s: i32, %acc: i32):
      %c = linalg.index 1 : index
      %t = linalg.index 2 : index
      %ct = arith.floordivsi %c, %c4 : index
      %start = arith.muli %ct, %c4 : index
      %tstart = arith.muli %t, %c4 : index
      %before = arith.cmpi slt, %tstart, %start : index
      %sum = arith.addi %acc, %s : i32
      %g = arith.select %before, %sum, %acc : i32
      linalg.yield %g : i32
    } -> tensor<2x10xi32>
    return %r : tensor<2x10xi32>
  }
}

// -----

#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @carry_f32(%sums: tensor<2x3xf32>, %o: tensor<2x10xf32>) -> tensor<2x10xf32> attributes {furiosa.written = [1]} {
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2x10xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x10xf32>) -> tensor<2x10xf32>
    %r = linalg.generic {indexing_maps = [#x, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%sums : tensor<2x3xf32>) outs(%f : tensor<2x10xf32>) {
    ^bb0(%s: f32, %acc: f32):
      %c = linalg.index 1 : index
      %t = linalg.index 2 : index
      %ct = arith.floordivsi %c, %c4 : index
      %start = arith.muli %ct, %c4 : index
      %tstart = arith.muli %t, %c4 : index
      %before = arith.cmpi slt, %tstart, %start : index
      %sum = arith.addf %acc, %s : f32
      %g = arith.select %before, %sum, %acc : f32
      linalg.yield %g : f32
    } -> tensor<2x10xf32>
    return %r : tensor<2x10xf32>
  }
}
