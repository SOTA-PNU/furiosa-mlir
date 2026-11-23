// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=tensor.pad

// A padded row read as tiles of its columns (the sums of a masked row's 4-column tiles: the 10 columns padded to 12,
// reshaped [3, 4]): no class reads a reshape of a padding, so the padded row is the row inserted into a constant
// of the padding, a copy step, and the reduction reads that buffer reshaped.

// CHECK-LABEL: func.func @tile_sums
// CHECK: %[[P:.*]] = call @tile_sums__s0(%arg0, %{{.*}}) : (tensor<2x10xi32>, tensor<2x12xi32>) -> tensor<2x12xi32>
// CHECK: %[[X:.*]] = tensor.expand_shape %[[P]] {{\[}}[0], [1, 2]] output_shape [2, 3, 4]
// CHECK: call @tile_sums__s1(%[[X]], %arg1)
// CHECK-LABEL: func.func private @tile_sums__s0
// CHECK: tensor.insert_slice %arg0 into %{{.*}}[0, 0] [2, 10] [1, 1] : tensor<2x10xi32> into tensor<2x12xi32>

#x = affine_map<(d0, d1, d2) -> (d0, d1 * 4 + d2)>
#c = affine_map<(d0, d1, d2) -> (d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @tile_sums(%x: tensor<2x10xi32>, %o: tensor<2x3xi32>) -> tensor<2x3xi32> attributes {furiosa.written = [1]} {
    %zero = arith.constant 0 : i32
    %k = tensor.empty() : tensor<4xi8>
    %e = tensor.empty() : tensor<2x3xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
    %p = tensor.pad %x low[0, 0] high[0, 2] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : i32
    } : tensor<2x10xi32> to tensor<2x12xi32>
    %r = linalg.generic {indexing_maps = [#c, #x, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%k, %p : tensor<4xi8>, tensor<2x12xi32>) outs(%f : tensor<2x3xi32>) {
    ^bb0(%u: i8, %a: i32, %acc: i32):
      %s = arith.addi %acc, %a : i32
      linalg.yield %s : i32
    } -> tensor<2x3xi32>
    return %r : tensor<2x3xi32>
  }
}
